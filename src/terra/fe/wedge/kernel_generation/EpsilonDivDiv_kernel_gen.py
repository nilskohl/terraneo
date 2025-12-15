import os, sys
from integrands import *
from kernel_helpers import *
from ast_extensions import *
from sympy.codegen.ast import (
    Assignment,
    For,
    CodeBlock,
    Variable,
    Declaration,
    Pointer,
    AugmentedAssignment,
    aug_assign,
    integer,
    Comment,
    String,
    Element,
)
from sympy import symbols, IndexedBase, Idx
from sympy.tensor.indexed import IndexedBase

local_subdomain_id, x_cell, y_cell, r_cell = sp.symbols(
    "local_subdomain_id x_cell y_cell r_cell", integer=True
)


# initial decls and loads of coords and src
(
    quad_surface_coords_symbol,
    wedge_surf_phy_coords_symbol,
    wedge_array_declarations,
    wedge_assignments,
) = make_wedge_surface_physical_coord_assignments(local_subdomain_id, x_cell, y_cell)
rads, rad_assignments = make_rad_assignments(local_subdomain_id, r_cell)
src_symbol, src_array_declaration, src_assignments = (
    make_extract_local_wedge_vector_assignments(
        local_subdomain_id, x_cell, y_cell, r_cell, "src"
    )
)
k_symbol, k_array_declaration, k_assignments = (
    make_extract_local_wedge_scalar_assignments(
        local_subdomain_id, x_cell, y_cell, r_cell, "k"
    )
)

kernel_code = "\n"
kernel_code += wedge_array_declarations
kernel_code += terraneo_ccode(wedge_assignments)
kernel_code += "\n" + terraneo_ccode(rad_assignments)
kernel_code += src_array_declaration
kernel_code += terraneo_ccode(src_assignments)
kernel_code += k_array_declaration
kernel_code += terraneo_ccode(k_assignments)


# quadrature data initialization
qp_data = quad_points_1x1
qw_data = quad_weights_1x1
num_qps = 1
dim = 3
# have to do string injection here because sympy
qp_array_name = "qp_array"
qw_array_name = "qw_array"
qp_array = IndexedBase(qp_array_name, shape=(num_qps, 3), real=True)
qw_array = IndexedBase(qw_array_name, shape=(num_qps), real=True)


kernel_code += f"\ndouble {qp_array_name}[{num_qps}][{3}];\n"
kernel_code += f"double {qw_array_name}[{num_qps}];"
kernel_code += "\n" + terraneo_ccode(
    CodeBlock(
        *[
            Assignment(qp_array[q, d], qp_data[q][d])
            for d in range(dim)
            for q in range(num_qps)
        ]
    )
)
kernel_code += "\n" + terraneo_ccode(
    CodeBlock(*[Assignment(qw_array[q], qw_data[q]) for q in range(num_qps)])
)

num_wedges_per_hex_cell = 2
num_nodes_per_wedge_surface = 3
num_nodes_per_wedge = 6


# Produce conditional statements for BCs and diagonal kernels
# (could also generate exactly the same loop bounds, but to multiply with conds and unroll
#  enables a cse across loop iteration (essentially, cse finds loop invariants) which
# *might* result in fewer flops.
cmb_shift, surface_shift, at_surface_boundary, at_cmb_boundary = sp.symbols(
    "cmb_shift surface_shift at_surface_boundary at_cmb_boundary", integer=True
)
max_rad, treat_boundary, diagonal, postloop = sp.symbols(
    " max_rad treat_boundary_ diagonal_ postloop", integer=True
)
kernel_code += "\n" + terraneo_ccode(
    CodeBlock(
        *[
            Variable.deduced(at_cmb_boundary).as_Declaration(
                value=FunctionCall(
                    "has_flag",
                    [local_subdomain_id, x_cell, y_cell, r_cell, String("CMB")],
                )
            ),
            Variable.deduced(at_surface_boundary).as_Declaration(
                value=FunctionCall(
                    "has_flag",
                    [local_subdomain_id, x_cell, y_cell, r_cell + 1, String("SURFACE")],
                )
            ),
            Variable.deduced(cmb_shift).as_Declaration(
                value=sp.Piecewise(
                    (
                        3,
                        sp.Eq(diagonal, False)
                        & treat_boundary
                        & Ne(at_cmb_boundary, 0),
                    ),
                    (0, True),
                )
            ),
            Variable.deduced(max_rad).as_Declaration(
                value=FunctionCall("radii_.extent", [1]) - 1,
            ),
            Variable.deduced(surface_shift).as_Declaration(
                value=sp.Piecewise(
                    (
                        3,
                        sp.Eq(diagonal, False)
                        & treat_boundary
                        & Ne(at_surface_boundary, 0),
                    ),
                    (0, True),
                )
            ),
        ]
    )
)

# generate tmp destination array filled during quadrature loop
dst_array_name = "dst_array"
dst_symbol = IndexedBase(
    dst_array_name, shape=(dim, num_wedges_per_hex_cell, num_nodes_per_wedge)
)
kernel_code += f"\ndouble {dst_array_name}[{dim}][{num_wedges_per_hex_cell}][{num_nodes_per_wedge}] = {{0}};"
# construct quadrature loop
quadloop_body = []
quadloop_exprs = []

# assign current quadrature point and weight
# to make the kernel smaller/more readable it would be nice to directly generate array accesses
# and use them in the jacobian computation etc., but SymPy wont allow multiplication of element
# accesses with other stuff (symbols, matrices), so we have to generate tmp symbols which are
# assigned the array accesses and are used in symbolic computations.
# (those assigns and bridging tmps should be optimized away by backend compiler, so prolly not performance damaging)
q_symbol = sp.symbols("q", integer=True)
qp = [qp_array[q_symbol, d] for d in range(dim)]
qw = qw_array[q_symbol]

# wedge, dimi, dimj counter
w_symbol, dimi_symbol, dimj_symbol, dim_diagBC_symbol = sp.symbols(
    "w dimi dimj dim_diagBC", integer=True
)

# evaluate coefficient
k_eval_symbol = sp.symbols("k_eval", real=True)
k_eval = sum(
    [shape_vec(j, qp) * k_symbol[w_symbol, j] for j in range(num_nodes_per_wedge)]
)
k_eval_replacements, k_eval_reduced_exprs = sp.cse(
    exprs=k_eval, symbols=numbered_symbols(prefix=f"tmpcse_k_eval_", real=True)
)
quadloop_body += [
    Comment("Coefficient evaluation on current wedge w")
] + make_ast_from_exprs(k_eval_replacements)
quadloop_body.append(
    Variable.deduced(k_eval_symbol).as_Declaration(value=k_eval_reduced_exprs[0])
)


# Jacobian
jac_laterally_precomputed = False
if not jac_laterally_precomputed:
    jac_exprs = []
    J = jac_from_array(
        [
            [wedge_surf_phy_coords_symbol[w_symbol, j, d] for d in range(dim)]
            for j in range(dim)
        ],
        rads[0],
        rads[1],
        qp,
    )
    # CSE on jacobian
    J_cse_assignments, J_cse_exprs = flattened_cse(J, f"tmpcse_J_")
    jac_exprs += J_cse_assignments
    # replace Jacobian entries with tmp symbols to speed up inversion
    J_cse = sp.Matrix(3, 3, J_cse_exprs)
    J_cse_replaced_assignments, J_cse_replaced = replace_matrix(J_cse, f"J")
    jac_exprs += J_cse_replaced_assignments
    # compute abs determinant + CSE
    J_det = J_cse_replaced.det()
    J_det_replacements, J_det_reduced_exprs = sp.cse(
        exprs=J_det, symbols=numbered_symbols(prefix=f"tmpcse_det_", real=True)
    )
    jac_exprs += J_det_replacements
    J_det_symbol = sp.symbols(f"J_det", real=True)
    jac_exprs.append((J_det_symbol, J_det_reduced_exprs[0]))
    J_abs_det = abs(J_det_symbol)
    # invert + transpose jacobian
    J_inv = J_cse_replaced.adjugate() / J_det_symbol
    J_invT = J_inv.transpose()
    # second CSE after inversion
    J_invT_replacements, J_invT_reduced_exprs = flattened_cse(J_invT, f"tmpcse_J_invT_")
    jac_exprs += J_invT_replacements
    J_invT_cse = sp.Matrix(3, 3, J_invT_reduced_exprs)
    J_invT_cse_assignments, J_invT_cse_replaced = replace_matrix(
        J_invT_cse, f"J_invT_cse"
    )
    jac_exprs += J_invT_cse_assignments

    quadloop_body += [
        Comment("Computation + Inversion of the Jacobian")
    ] + make_ast_from_exprs(jac_exprs)
else:
    r_inv, g2, grad_r, grad_r_inv = symbols("r_inv g2 grad_r grad_r_inv", real=True)
    kernel_code += "\n" + terraneo_ccode(
        CodeBlock(
            Variable.deduced(grad_r).as_Declaration(
                value=grad_forward_map_rad(rads[0], rads[1])
            ),
            Variable.deduced(grad_r_inv).as_Declaration(value=1.0 / grad_r),
        )
    )

    J_invT = IndexedBase("J_invT", shape=(3, 3))
    factors = IndexedBase("factors", shape=(3))
    d1, d2 = symbols("d1 d2", integer=True)
    quadloop_body += [
        Comment(
            "\nLoad the radially constant parts of the Jacobial from storage,\n scale with radial parts of the current element."
        ),
        String(f"double J_invT[{dim}][{dim}] = {{0}};"),
        Variable.deduced(r_inv).as_Declaration(
            value=1.0 / (forward_map_rad(rads[0], rads[1], qp[2]))
        ),
        String(f"double factors[{dim}] = {{ {r_inv}, {r_inv}, {grad_r_inv} }};"),
        Variable.deduced(d1).as_Declaration(),
        Variable.deduced(d2).as_Declaration(),
        For(
            d1,
            range(0, dim),
            [
                For(
                    d2,
                    range(0, dim),
                    [
                        Assignment(
                            J_invT[d1, d2],
                            factors[d2]
                            * FunctionCall(
                                "g1_",
                                [
                                    local_subdomain_id,
                                    x_cell,
                                    y_cell,
                                    w_symbol,
                                    q_symbol,
                                    d1,
                                    d2,
                                ],
                            ),
                        )
                    ],
                )
            ],
        ),
    ]
    quadloop_body.append(
        Variable.deduced(g2).as_Declaration(
            value=FunctionCall(
                "g2_", [local_subdomain_id, x_cell, y_cell, w_symbol, q_symbol]
            )
        )
    )

    r = 1 / r_inv
    J_abs_det = r * r * grad_r * g2
    J_invT_cse_replaced = Matrix(
        [[J_invT[i, j] for j in range(dim)] for i in range(dim)]
    )

# precompute gradients + CSE
grad_exprs = []
scalar_grad_is = []
scalar_grad_name = "scalar_grad"
scalar_grad = IndexedBase(scalar_grad_name, shape=(num_nodes_per_wedge, dim), real=True)
grad_exprs.append(f"\ndouble {scalar_grad_name}[{num_nodes_per_wedge}][{dim}] = {{0}}")
for i in range(num_nodes_per_wedge):
    scalar_grad_is.append(J_invT_cse_replaced * grad_shape_vec(i, qp))
scalar_grad_i_replacements, scalar_grad_i_reduced_exprs = sp.cse(
    exprs=scalar_grad_is,
    symbols=numbered_symbols(prefix=f"tmpcse_grad_i_", real=True),
)
grad_exprs += scalar_grad_i_replacements
for i, grad_i_expr in enumerate(scalar_grad_i_reduced_exprs):
    for d, gie in enumerate(grad_i_expr):
        grad_exprs.append((scalar_grad[i, d], gie))

# up to here, we are component-invariant, and can already transform
# the symbolic computation to actual AST nodes and add it to the quadrature loop body
quadloop_body += [
    Comment(
        "Computation of the gradient of the scalar shape functions belonging to each DoF.\n "
        "In the Eps-component-loops, we insert the gradient at the entry of the\n "
        "vectorial gradient matrix corresponding to the Eps-component."
    )
] + make_ast_from_exprs(grad_exprs)

# from now on, statements go into the component/dimensions loops
dimloop_j_body = []
dimloop_j_exprs = []
dimloop_i_body = []
dimloop_i_exprs = []

# setup gradients of vectorial basis functions (which are 3x3 matrices)
# assemble src/trial gradient (sum over rows in local matrix)
E_trial_name = "E_grad_trial"
E_trial = IndexedBase(E_trial_name, shape=(3, 3), real=True)
g_symbol = symbols("node_idx", integer=True)
div_u = symbols("div_u", real=True)
u_grad_loop_exprs = []
u_grad_loop_exprs.append(f"\ndouble {E_trial_name}[3][3] = {{0}}")


def create_col_assigns(E, col, col_vec):
    return [(E[i, col], col_vec[i]) for i in range(dim)]


u_grad_loop_exprs += create_col_assigns(
    E_trial, dimj_symbol, [scalar_grad[g_symbol, d] for d in range(dim)]
)
E_trial_matrix = Matrix(3, 3, [E_trial[i, j] for i in range(dim) for j in range(dim)])
grad_u_name = "grad_u"
grad_u = IndexedBase(grad_u_name, shape=(3, 3), real=True)
grad_u_matrix = Matrix(3, 3, [grad_u[i, j] for i in range(dim) for j in range(dim)])
symm_grad_j = 0.5 * (E_trial_matrix + (E_trial_matrix).transpose())
symm_grad_j_replacements, symm_grad_j_reduced_exprs = sp.cse(
    exprs=symm_grad_j,
    symbols=numbered_symbols(prefix=f"tmpcse_symgrad_trial_", real=True),
)
grad_u_matrix = grad_u_matrix + Matrix(
    3, 3, symm_grad_j_reduced_exprs[0]
).multiply_elementwise(
    Matrix(
        3,
        3,
        [
            src_symbol[dimj_symbol, w_symbol, g_symbol]
            for i in range(dim)
            for j in range(dim)
        ],
    )
)
u_grad_loop_exprs += (
    symm_grad_j_replacements
    + [(grad_u[i, j], grad_u_matrix[i, j]) for i in range(dim) for j in range(dim)]
    + [
        (
            div_u,
            div_u
            + E_trial[dimj_symbol, dimj_symbol]
            * src_symbol[dimj_symbol, w_symbol, g_symbol],
        )
    ]
)

# pair symmetric gradient of u (test space) with symmetric gradients of the trial space
E_test_name = "E_grad_test"
E_test = IndexedBase(E_test_name, shape=(3, 3), real=True)
pairing_loop_exprs = []
pairing_loop_exprs.append(f"\ndouble {E_test_name}[3][3] = {{0}}")
pairing_loop_exprs += create_col_assigns(
    E_test, dimi_symbol, [scalar_grad[g_symbol, d] for d in range(dim)]
)
E_test_matrix = Matrix(3, 3, [E_test[i, j] for i in range(dim) for j in range(dim)])


def double_contract(A, B):
    T = A.multiply_elementwise(B)
    return sum([t for row in T.tolist() for t in row])


symm_grad_i = 0.5 * (E_test_matrix + (E_test_matrix).transpose())
symm_grad_i_replacements, symm_grad_i_reduced_exprs = sp.cse(
    exprs=symm_grad_i,
    symbols=numbered_symbols(prefix=f"tmpcse_symgrad_test_", real=True),
)

pairing_replacements, pairing_reduced_exprs = sp.cse(
    exprs=[
        qw
        * k_eval_symbol
        * J_abs_det
        * (
            2
            * double_contract(
                Matrix(3, 3, (symm_grad_i_reduced_exprs[0])),
                Matrix(3, 3, [grad_u[i, j] for i in range(dim) for j in range(dim)]),
            )
            - 2.0 / 3.0 * E_test[dimi_symbol, dimi_symbol] * div_u
        )
    ],
    symbols=numbered_symbols(prefix=f"tmpcse_pairing_", real=True),
)
pairing_loop_exprs += (
    symm_grad_i_replacements
    + pairing_replacements
    + [
        (
            dst_symbol[dimi_symbol, w_symbol, g_symbol],
            dst_symbol[dimi_symbol, w_symbol, g_symbol] + pairing_reduced_exprs[0],
        )
    ]
)

# boundary/diagonal loop
boundary_loop_exprs = []
boundary_loop_exprs.append(f"\ndouble {E_test_name}[3][3] = {{0}}")
boundary_loop_exprs += create_col_assigns(
    E_test, dim_diagBC_symbol, [scalar_grad[g_symbol, d] for d in range(dim)]
)
grad_u_diag_name = "grad_u_diag"
grad_u_diag = IndexedBase(grad_u_diag_name, shape=(3, 3), real=True)
boundary_loop_exprs.append(f"\ndouble {grad_u_diag_name}[3][3] = {{0}}")
grad_u_diag_matrix = Matrix(3, 3, symm_grad_i_reduced_exprs[0]).multiply_elementwise(
    Matrix(
        3,
        3,
        [
            src_symbol[dim_diagBC_symbol, w_symbol, g_symbol]
            for i in range(dim)
            for j in range(dim)
        ],
    )
)
symm_grad_i_reduced_matrix = Matrix(3, 3, symm_grad_i_reduced_exprs[0])
diag_pairing_replacements, diag_pairing_reduced_exprs = sp.cse(
    exprs=[
        qw
        * k_eval_symbol
        * J_abs_det
        * (
            2 * double_contract(symm_grad_i_reduced_matrix, grad_u_diag_matrix)
            - 2.0
            / 3.0
            * E_test[dim_diagBC_symbol, dim_diagBC_symbol]
            * E_test[dim_diagBC_symbol, dim_diagBC_symbol]
            * src_symbol[dim_diagBC_symbol, w_symbol, g_symbol]
        )
    ],
    symbols=numbered_symbols(prefix=f"tmpcse_pairing_", real=True),
)
boundary_loop_exprs += (
    symm_grad_i_replacements
    + [
        (grad_u_diag[i, j], grad_u_diag_matrix[i, j])
        for i in range(dim)
        for j in range(dim)
    ]
    + diag_pairing_replacements
    + [
        (
            dst_symbol[dim_diagBC_symbol, w_symbol, g_symbol],
            dst_symbol[dim_diagBC_symbol, w_symbol, g_symbol]
            + diag_pairing_reduced_exprs[0],
        )
    ]
)

# append all loop bodies
dimloop_j_body += [
    Conditional(
        Eq(diagonal, False),
        [
            For(
                g_symbol,
                [0 + cmb_shift, num_nodes_per_wedge - surface_shift, 1],
                make_ast_from_exprs(u_grad_loop_exprs),
            ),
        ],
    )
]

dimloop_i_body += [
    Conditional(
        Eq(diagonal, False),
        [
            For(
                g_symbol,
                [0 + cmb_shift, num_nodes_per_wedge - surface_shift, 1],
                make_ast_from_exprs(pairing_loop_exprs),
            ),
        ],
    )
]

dimloop_diagBC_body = [
    Conditional(
        (
            diagonal
            | (treat_boundary & (Ne(at_surface_boundary, 0) | Ne(at_cmb_boundary, 0)))
        ),
        [
            Variable.deduced(g_symbol).as_Declaration(),
            For(
                g_symbol,
                [0 + surface_shift, num_nodes_per_wedge - cmb_shift, 1],
                make_ast_from_exprs(boundary_loop_exprs),
            ),
        ],
    ),
]

# generate code for quadloop, wedgeloop and dimloop (eps + divdiv components)
kernel_code += (
    "\n"
    + terraneo_ccode(
        CodeBlock(
            *[
                Variable.deduced(w_symbol).as_Declaration(value=0),
                Comment(
                    "Apply local matrix for both wedges and accumulated for all quadrature points."
                ),
                For(
                    w_symbol,
                    range(0, 2),
                    [
                        Variable.deduced(q_symbol).as_Declaration(value=0),
                        For(
                            q_symbol,
                            range(0, num_qps),
                            [
                                *quadloop_body,
                                Variable.deduced(dimj_symbol).as_Declaration(),
                                String(f"\ndouble {grad_u}[3][3] = {{0}}"),
                                Variable.deduced(div_u).as_Declaration(value=0.0),
                                Variable.deduced(g_symbol).as_Declaration(),
                                Comment(
                                    "In the following, we exploit the outer-product-structure of the local MV both in \nthe components of the Epsilon operators and in the local DoFs."
                                ),
                                Comment("Loop to assemble the trial gradient."),
                                For(dimj_symbol, range(0, dim), *[dimloop_j_body]),
                                Variable.deduced(dimi_symbol).as_Declaration(),
                                Comment(
                                    "Loop to pair the assembled trial gradient with the test gradients."
                                ),
                                For(dimi_symbol, range(0, dim), *[dimloop_i_body]),
                                Variable.deduced(dim_diagBC_symbol).as_Declaration(),
                                Comment(
                                    "Loop to apply BCs or only the diagonal of the operator."
                                ),
                                For(
                                    dim_diagBC_symbol,
                                    range(0, dim),
                                    *[dimloop_diagBC_body],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
    )
    + "\n"
)

# atomic adds of local dsts to global dst
kernel_code += terraneo_ccode(
    make_atomic_add_local_wedge_vector_coefficients(
        local_subdomain_id, x_cell, y_cell, r_cell, dst_symbol
    )
)

print(kernel_code)
with open("EpsDivDiv_kernel", "w", encoding="utf-8") as f:
    f.write(kernel_code)
