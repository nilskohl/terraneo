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
    integer,
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
quad, wedge, lat, wedge_assignments = make_wedge_surface_physical_coord_assignments(
    local_subdomain_id, x_cell, y_cell
)
rads, rad_assignments = make_rad_assignments(local_subdomain_id, r_cell)

srcs, src_assignments = make_extract_local_wedge_scalar_assignments(
    local_subdomain_id, x_cell, y_cell, r_cell
)
kernel_code += ccode(wedge_assignments)
kernel_code += "\n" + ccode(rad_assignments)
kernel_code += "\n" + ccode(src_assignments)

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


num_wedges_per_hex_cell = 2
num_nodes_per_wedge_surface = 3
num_nodes_per_wedge = 6
dsts = []


# Produce tenary statements for BCs and diagonal kernels
cmb_shift, surface_shift = sp.symbols("cmb_shift surface_shift", integer=True)
max_rad, treat_boundary, diagonal, postloop = sp.symbols(
    " max_rad treat_boundary_ diagonal_ postloop", integer=True
)
kernel_code += "\n" + ccode(
    CodeBlock(
        *[
            Declaration(Variable.deduced(cmb_shift)),
            Assignment(
                cmb_shift,
                sp.Piecewise(
                    (3, sp.Eq(diagonal, False) & treat_boundary & sp.Eq(r_cell, 0)),
                    (0, True),
                ),
            ),
            Declaration(Variable.deduced(max_rad)),
            Assignment(
                max_rad,
                FunctionCall("radii_.extent", [1]) - 1,
            ),
            Declaration(Variable.deduced(surface_shift)),
            Assignment(
                surface_shift,
                sp.Piecewise(
                    (
                        3,
                        sp.Eq(diagonal, False)
                        & treat_boundary
                        & sp.Eq(r_cell + 1, max_rad),
                    ),
                    (0, True),
                ),
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
                    sp.Piecewise(
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
                    sp.Piecewise(
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
                    sp.Piecewise(
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

for w in range(num_wedges_per_hex_cell):


    # for q in range(num_qps):
    quadloop_body = []

    # assign quadrature point and weight
    qp_symbol = sp.symbols("qp_0 qp_1 qp_2", real=True)
    qw_symbol = sp.symbols("qw", real=True)
    q_symbol = sp.symbols("q", integer=True)
    for i, qp in enumerate(qp_symbol):
        quadloop_body.append(Variable.deduced(qp_symbol[i]).as_Declaration(value=qp_array_symbol[q_symbol, i]))
    quadloop_body.append(Variable.deduced(qw_symbol).as_Declaration(value=qw_array_symbol[q_symbol]))

    quadloop_exprs = []

    dsts_qw = sp.Matrix.zeros(num_nodes_per_wedge, 1)

    ### Jacobian
    J = jac_from_array(wedge[w], rads[0], rads[1], qp_symbol)

    # CSE on jacobian
    J_cse_assignments, J_cse_exprs = flattened_cse(J, f"w{w}_tmpcse_J_")
    for stmt in J_cse_assignments:
        quadloop_exprs += (J_cse_assignments)

    # replace Jacobian entries with tmp symbols to speed up inversion
    J_cse = sp.Matrix(3, 3, J_cse_exprs)
    J_cse_replaced_assignments, J_cse_replaced = replace_matrix(J_cse, f"w{w}_J")
    quadloop_exprs += J_cse_replaced_assignments

    # compute abs determinant + CSE
    J_det = J_cse_replaced.det()
    J_det_replacements, J_det_reduced_exprs = sp.cse(
        exprs=J_det, symbols=numbered_symbols(prefix=f"w{w}_tmpcse_det_", real=True)
    )
    quadloop_exprs += J_det_replacements
    J_det_symbol = sp.symbols(f"w{w}_J_det", real=True)
    quadloop_exprs.append((J_det_symbol, J_det_reduced_exprs[0]))
    J_abs_det = abs(J_det_symbol)

    # invert + transpose jacobian
    J_inv = J_cse_replaced.adjugate() / J_det_symbol
    J_invT = J_inv.transpose()

    # second CSE after inversion
    J_invT_replacements, J_invT_reduced_exprs = flattened_cse(
        J_invT, f"w{w}_tmpcse_J_invT_"
    )
    quadloop_exprs += J_invT_replacements
    J_invT_cse = sp.Matrix(3, 3, J_invT_reduced_exprs)
    J_invT_cse_assignments, J_invT_cse_replaced = replace_matrix(
        J_invT_cse, f"w{w}_J_invT_cse"
    )
    quadloop_exprs += J_invT_cse_assignments

    # precompute gradients + CSE
    grad_is = []
    grad_is_symbols = []
    for i in range(num_nodes_per_wedge):
        grad_i = J_invT_cse_replaced * grad_shape_vec(i, qp_symbol)
        grad_i_symbol = sp.Matrix(
            sp.symbols(f"w{w}_grad_i{i}_0 w{w}_grad_i{i}_1 w{w}_grad_i{i}_2", real=True)
        )
        grad_is.append(grad_i)
        grad_is_symbols.append(grad_i_symbol)

    grad_i_replacements, grad_i_reduced_exprs = sp.cse(
        exprs=grad_is,
        symbols=numbered_symbols(prefix=f"w{w}_tmpcse_grad_i_", real=True),
    )
    quadloop_exprs += grad_i_replacements
    for grad_i, grad_i_symbol in zip(grad_i_reduced_exprs, grad_is_symbols):
        for gi, gis in zip(grad_i, grad_i_symbol):
            quadloop_body.append((gis, gi))

    # assemble src/trial gradient (sum over rows in local matrix)
    srcs_w = sp.Matrix(srcs[w])
    grad_u_symbols = sp.Matrix(sp.symbols(f"w{w}_grad_u_0 w{w}_grad_u_1 w{w}_grad_u_2", real=True))
    grad_u = sp.Matrix.zeros(3, 1)

    for j in range(num_nodes_per_wedge):
        grad_u += srcs_w[j] * grad_is_symbols[j] * conditionals["trial"][j]
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
        grad_u_diag = srcs_w[i] * grad_is_symbols[i] * conditionals["diag_bc"][i]
        res = J_abs_det * qw_symbol * grad_i.transpose() * grad_u_diag
        dsts_qw[i] += res[0]

    # final cse
    dst_replacements, dst_reduced_exprs = sp.cse(
        exprs=dsts_qw,
        symbols=numbered_symbols(prefix=f"w{w}_tmpcse_dst_", real=True),
    )
    quadloop_exprs += dst_replacements

    # assemble quadloop body ast
    quadloop_body = []
    for lhs, rhs in quadloop_exprs:
        quadloop_body.append(Variable.deduced(lhs).as_Declaration(value=rhs))

    # add up on dst
    dsts_w_symbols = [sp.symbols(f"dst_{w}_{i}", real=True) for i in range(num_nodes_per_wedge)]
    kernel_code += ccode(CodeBlock(*[Variable.deduced(dws).as_Declaration(value=0.0) for dws in dsts_w_symbols]))
    for i in range(num_nodes_per_wedge):
        quadloop_body.append(Assignment(dsts_w_symbols[i], dsts_w_symbols[i] + dst_reduced_exprs[0][i]))

    # generate code for quadloop
    kernel_code += "\n" + ccode(CodeBlock(For(q_symbol, range(0, num_qps), quadloop_body))) + "\n"

print(kernel_code)
exit(0)
kernel += make_atomic_add_local_wedge_scalar_coefficients(
    local_subdomain_id, x_cell, y_cell, r_cell, dsts
)

# Finally: print code
cpp_code = "\n// Kernel body:\n"


for stmt in kernel:
    print(stmt)
    if isinstance(stmt, str):
        cpp_code += stmt
    elif isinstance(stmt, For) or isinstance(stmt, Declaration):
        cpp_code += f" {ccode(stmt, contract=False)}; \n"
    else:
        var_name, expr = stmt

        print(var_name)
        print(expr)

        def get_type(sym):
            if var_name.is_integer:
                return "int"
            else:
                return "double"

        if isinstance(expr, str):
            cpp_code += f"{  get_type(var_name) } {var_name} = {expr};\n"
        else:
            cpp_code += (
                f"{  get_type(var_name) } {var_name} = {ccode(expr, contract=False)};\n"
            )

with open("Laplace_kernel", "w", encoding="utf-8") as f:
    f.write(cpp_code)

# print(print_atomic_add_local_wedge_scalar_coefficients())
