from sympy import ccode
import os, sys
from integrands import *
from kernel_helpers import *
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
    Comment
)
from sympy.printing import ccode, print_ccode
from sympy import symbols, IndexedBase, Idx
from sympy.utilities.codegen import codegen
from sympy import ccode


local_subdomain_id, x_cell, y_cell, r_cell = sp.symbols(
    "local_subdomain_id x_cell y_cell r_cell", integer=True
)


def flattened_cse(M, prefix):

    exprs = M.tolist()
    exprs_flat = [item for row in exprs for item in row]

    return sp.cse(exprs=exprs_flat, symbols=numbered_symbols(prefix=prefix, real=True))


def replace_matrix(matrix, prefix):
    replace_assignments = []
    replaced_matrix = []
    row, col = matrix.shape
    for i in range(row):
        for j in range(col):
            tmp_ij = sp.symbols(f"{prefix}_{i}_{j}", real=True)
            replaced_matrix.append(tmp_ij)
            replace_assignments.append((tmp_ij, matrix[i, j]))
    replaced_matrix = sp.Matrix(row, col, replaced_matrix)
    return replace_assignments, replaced_matrix


kernel_code = "\n"

# initial decls and loads of coords and src
(
    quad_surface_coords_symbol,
    wedge_surf_phy_coords_symbol,
    wedge_array_declarations,
    wedge_assignments,
) = make_wedge_surface_physical_coord_assignments(local_subdomain_id, x_cell, y_cell)
rads, rad_assignments = make_rad_assignments(local_subdomain_id, r_cell)
src_symbol, src_array_declaration, src_assignments = (
    make_extract_local_wedge_scalar_assignments(
        local_subdomain_id, x_cell, y_cell, r_cell
    )
)
kernel_code += wedge_array_declarations
kernel_code += ccode(wedge_assignments)
kernel_code += "\n" + ccode(rad_assignments)
kernel_code += src_array_declaration
kernel_code += ccode(src_assignments)


# quadrature data initialization
qp_data = quad_points_3x2
qw_data = quad_weights_3x2
num_qps = 6
dim = 3
# have to do string injection here because sympy
kernel_code += f"\ndouble qp_array[{num_qps}][{3}];\n"
kernel_code += f"double qw_array[{num_qps}];"
qp_array_symbol = Pointer("qp_array")
qw_array_symbol = Pointer("qw_array")
q_idx = symbols("q_idx", integer=True)
kernel_code += "\n" + ccode(
    CodeBlock(
        *[
            Assignment(qp_array_symbol[q, d], qp_data[q][d])
            for d in range(dim)
            for q in range(num_qps)
        ]
    )
)
kernel_code += "\n" + ccode(
    CodeBlock(
        *[
            Assignment(qw_array_symbol[q], qw_data[q])
            for q in range(num_qps)
        ]
    )
)

num_wedges_per_hex_cell = 2
num_nodes_per_wedge_surface = 3
num_nodes_per_wedge = 6


# Produce conditional statements for BCs and diagonal kernels 
# (could also generate exactly the same loop bounds, but to multiply with conds and unroll 
#  enables a cse across loop iteration (essentially, cse finds loop invariants) which
# *might* result in fewer flops.
cmb_shift, surface_shift = sp.symbols("cmb_shift surface_shift", integer=True)
max_rad, treat_boundary, diagonal, postloop = sp.symbols(
    " max_rad treat_boundary_ diagonal_ postloop", integer=True
)
kernel_code += "\n" + ccode(
    CodeBlock(
        *[
            Variable.deduced(cmb_shift).as_Declaration(
                value=sp.Piecewise(
                    (3, sp.Eq(diagonal, False) & treat_boundary & sp.Eq(r_cell, 0)),
                    (0, True),
                )
            ),
            Variable.deduced(max_rad).as_Declaration(
                value=FunctionCall("radii_.extent", [1]) + - 1,
            ),
            Variable.deduced(surface_shift).as_Declaration(
                value=sp.Piecewise(
                    (
                        3,
                        sp.Eq(diagonal, False)
                        & treat_boundary
                        & sp.Eq(r_cell + 1, max_rad),
                    ),
                    (0, True),
                )
            ),
        ]
    )
)
conditionals = {"trial": [], "test": [], "diag_bc": []}
for i in range(num_nodes_per_wedge):
    cl0 = sp.symbols(f"trial_it{i}_cond", integer=True)
    cl1 = sp.symbols(f"test_it{i}_cond", integer=True)
    cl2 = sp.symbols(f"diag_bc_it{i}_cond", integer=True)

    conditionals["trial"].append(cl0)
    conditionals["test"].append(cl1)
    conditionals["diag_bc"].append(cl2)
    kernel_code += "\n" + ccode(
        CodeBlock(
            *[
                Declaration(Variable.deduced(cl0)),
                Assignment(
                    cl0,
                    Piecewise(
                        (
                            1,
                            sp.Eq(diagonal, False)
                            & (i >= (0 + cmb_shift))
                            & (i < (num_nodes_per_wedge - surface_shift)),
                        ),
                        (0, True),
                    ),
                ),
                Declaration(Variable.deduced(cl1)),
                Assignment(
                    cl1,
                    Piecewise(
                        (
                            1,
                            sp.Eq(diagonal, False)
                            & (i >= (0 + cmb_shift))
                            & (i < (num_nodes_per_wedge - surface_shift)),
                        ),
                        (0, True),
                    ),
                ),
                Declaration(Variable.deduced(cl2)),
                Assignment(
                    cl2,
                    Piecewise(
                        (
                            1,
                            (
                                sp.Eq(diagonal, True)
                                | (
                                    sp.Eq(treat_boundary, True)
                                    & (sp.Eq(r_cell + 1, max_rad) | sp.Eq(r_cell, 0))
                                )
                            )
                            & (
                                (i >= (0 + surface_shift))
                                & (i < (num_nodes_per_wedge - cmb_shift))
                            ),
                        ),
                        (0, True),
                    ),
                ),
            ]
        )
    )

# generate tmp destination array filled during quadrature loop
kernel_code += (
    f"\ndouble dst_array[{num_wedges_per_hex_cell}][{num_nodes_per_wedge}];\n"
)
dst_array_symbol = Pointer("dst_array")
kernel_code += "\n" + ccode(
    CodeBlock(
        *[
            Assignment(dst_array_symbol[w, i], 0.0)
            for w in range(num_wedges_per_hex_cell)
            for i in range(num_nodes_per_wedge)
        ]
    )
)

# construct quadrature loop
quadloop_body = []
quadloop_exprs = []

# assign current quadrature point and weight
# to make the kernel smaller/more readable it would be nice to directly generate array accesses 
# and use them in the jacobian computation etc., but SymPy wont allow multiplication of element 
# accesses with other stuff (symbols, matrices), so we have to generate tmp symbols which are 
# assigned the array accesses and are used in symbolic computations. 
# (those assigns and bridging tmps should be optimized away by backend compiler, so prolly not performance damaging)
qp_symbol = sp.symbols("qp_0 qp_1 qp_2", real=True)
qw_symbol = sp.symbols("qw", real=True)
q_symbol = sp.symbols("q", integer=True)
for i, qp in enumerate(qp_symbol):
    quadloop_body.append(
        Variable.deduced(qp_symbol[i]).as_Declaration(
            value=qp_array_symbol[q_symbol, i]
        )
    )
quadloop_body.append(
    Variable.deduced(qw_symbol).as_Declaration(value=qw_array_symbol[q_symbol])
)

# tmp dest for current wedge + quadpoint
dsts_qw = sp.Matrix.zeros(num_nodes_per_wedge, 1)
# wedge counter
w_symbol = sp.symbols("w", integer=True)

# tmp wedge coords
wedge_tmp_symbols = [
    [symbols(f"wedge_tmp_symbols_{i}_{d}", real=True) for d in range(dim)] for i in range(dim)
]
quadloop_exprs += [
    (wedge_tmp_symbols[i][d], wedge_surf_phy_coords_symbol[w_symbol, i, d])
    for d in range(dim)
    for i in range(dim)
]

# Jacobian
J = jac_from_array(
    wedge_tmp_symbols,
    rads[0],
    rads[1],
    qp_symbol,
)

# CSE on jacobian
J_cse_assignments, J_cse_exprs = flattened_cse(J, f"tmpcse_J_")
quadloop_exprs += J_cse_assignments
#for stmt in J_cse_assignments:

# replace Jacobian entries with tmp symbols to speed up inversion
J_cse = sp.Matrix(3, 3, J_cse_exprs)
J_cse_replaced_assignments, J_cse_replaced = replace_matrix(J_cse, f"J")
quadloop_exprs += J_cse_replaced_assignments

# compute abs determinant + CSE
J_det = J_cse_replaced.det()
J_det_replacements, J_det_reduced_exprs = sp.cse(
    exprs=J_det, symbols=numbered_symbols(prefix=f"tmpcse_det_", real=True)
)
quadloop_exprs += J_det_replacements
J_det_symbol = sp.symbols(f"J_det", real=True)
quadloop_exprs.append((J_det_symbol, J_det_reduced_exprs[0]))
J_abs_det = abs(J_det_symbol)

# invert + transpose jacobian
J_inv = J_cse_replaced.adjugate() / J_det_symbol
J_invT = J_inv.transpose()

# second CSE after inversion
J_invT_replacements, J_invT_reduced_exprs = flattened_cse(J_invT, f"tmpcse_J_invT_")
quadloop_exprs += J_invT_replacements
J_invT_cse = sp.Matrix(3, 3, J_invT_reduced_exprs)
J_invT_cse_assignments, J_invT_cse_replaced = replace_matrix(J_invT_cse, f"J_invT_cse")
quadloop_exprs += J_invT_cse_assignments

# precompute gradients + CSE
grad_is = []
grad_is_symbols = []
for i in range(num_nodes_per_wedge):
    grad_i = J_invT_cse_replaced * grad_shape_vec(i, qp_symbol)
    grad_i_symbol = sp.Matrix(
        sp.symbols(f"grad_i{i}_0 grad_i{i}_1 grad_i{i}_2", real=True)
    )
    grad_is.append(grad_i)
    grad_is_symbols.append(grad_i_symbol)
grad_i_replacements, grad_i_reduced_exprs = sp.cse(
    exprs=grad_is,
    symbols=numbered_symbols(prefix=f"tmpcse_grad_i_", real=True),
)
quadloop_exprs += grad_i_replacements
for grad_i, grad_i_symbol in zip(grad_i_reduced_exprs, grad_is_symbols):
    for gi, gis in zip(grad_i, grad_i_symbol):
        quadloop_exprs.append((gis, gi))

# assemble src/trial gradient (sum over rows in local matrix)
grad_u_symbols = sp.Matrix(sp.symbols(f"grad_u_0 grad_u_1 grad_u_2", real=True))
grad_u = sp.Matrix.zeros(3, 1)
src_tmp_symbols = [symbols(f"src_tmp_symbols_{n}", real=True) for n in range(num_nodes_per_wedge)]
quadloop_exprs += [
    (src_tmp_symbols[n], src_symbol[w_symbol, n]) for n in range(num_nodes_per_wedge)
]
for j in range(num_nodes_per_wedge):
    grad_u += src_tmp_symbols[j] * grad_is_symbols[j] * conditionals["trial"][j]
for gu, gu_symbol in zip(grad_u, grad_u_symbols):
    quadloop_exprs.append((gu_symbol, gu))
for i in range(num_nodes_per_wedge):
    res = (
        J_abs_det
        * qw_symbol
        * grad_is_symbols[i].transpose()
        * grad_u_symbols
        * conditionals["test"][i]
    )
    dsts_qw[i] += res[0]

# post loop for bcs or diagonal kernel
for i in range(num_nodes_per_wedge):
    grad_u_diag = src_tmp_symbols[j] * grad_is_symbols[i] * conditionals["diag_bc"][i]
    res = J_abs_det * qw_symbol * grad_i.transpose() * grad_u_diag
    dsts_qw[i] += res[0]

# final cse
dst_replacements, dst_reduced_exprs = sp.cse(
    exprs=dsts_qw,
    symbols=numbered_symbols(prefix=f"tmpcse_dst_", real=True),
)
quadloop_exprs += dst_replacements

# assemble quadloop body ast
for lhs, rhs in quadloop_exprs:
    quadloop_body.append(Variable.deduced(lhs).as_Declaration(value=rhs))

# add up on dst
for i in range(num_nodes_per_wedge):
    quadloop_body.append(
        aug_assign(dst_array_symbol[w_symbol, i], "+", dst_reduced_exprs[0][i])
    )

# generate code for quadloop and wedgeloop
kernel_code += (
    "\n"
    + ccode(
        CodeBlock(
            *[
                Variable.deduced(w_symbol).as_Declaration(value=0),
                For(
                    w_symbol,
                    range(0, 2),
                    [
                        Variable.deduced(q_symbol).as_Declaration(value=0),
                        For(q_symbol, range(0, num_qps), quadloop_body),
                    ],
                ),
            ]
        )
    )
    + "\n"
)

# atomic adds of local dsts to global dst
kernel_code += make_atomic_add_local_wedge_scalar_coefficients(
    local_subdomain_id, x_cell, y_cell, r_cell, dst_array_symbol
)

print(kernel_code)
with open("Laplace_kernel", "w", encoding="utf-8") as f:
    f.write(kernel_code)

