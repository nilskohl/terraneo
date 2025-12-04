from sympy import ccode
import os, sys
from integrands import *
from kernel_helpers import *

local_subdomain_id, x_cell, y_cell, r_cell = sp.symbols(
    "local_subdomain_id x_cell y_cell r_cell"
)


def flattened_cse(M, prefix):

    exprs = M.tolist()
    exprs_flat = [item for row in exprs for item in row]

    return sp.cse(exprs=exprs_flat, symbols=numbered_symbols(prefix=prefix))


def replace_matrix(matrix, prefix):
    replace_assignments = []
    replaced_matrix = []
    row, col = matrix.shape
    for i in range(row):
        for j in range(col):
            tmp_ij = sp.symbols(f"{prefix}_{i}_{j}")
            replaced_matrix.append(tmp_ij)
            replace_assignments.append((tmp_ij, matrix[i, j]))
    replaced_matrix = sp.Matrix(row, col, replaced_matrix)
    return replace_assignments, replaced_matrix


kernel = []

### initial decls and loads
quad, wedge, lat, wedge_assignments = make_wedge_surface_physical_coord_assignments(
    local_subdomain_id, x_cell, y_cell
)
rads, rad_assignments = make_rad_assignments(local_subdomain_id, r_cell)

srcs, src_assignments = make_extract_local_wedge_scalar_assignments(
    local_subdomain_id, x_cell, y_cell, r_cell
)
# make_hex_assignments(local_subdomain_id, x_cell, y_cell, r_cell)
kernel += wedge_assignments
kernel += rad_assignments
kernel += src_assignments

qp_data = quad_points_3x2
qw_data = quad_weights_3x2
num_qps = 6

num_wedges_per_hex_cell = 2
num_nodes_per_wedge_surface = 3
num_nodes_per_wedge = 6
dim = 3
dsts = []


# Produce tenary statements for BCs and diagonal kernels
cmb_shift, surface_shift = sp.symbols("cmb_shift surface_shift", integer=True)
max_rad, treat_boundary, diagonal, postloop = sp.symbols(
    " max_rad treat_boundary_ diagonal_ postloop", integer=True
)
kernel.append(
    (
        cmb_shift,
        sp.Piecewise(
            (3, sp.Eq(diagonal, False) & treat_boundary & sp.Eq(r_cell, 0)),
            (0, True),
        ),
    )
)
kernel.append((max_rad, "radii_.extent( 1 ) - 1"))
kernel.append(
    (
        surface_shift,
        sp.Piecewise(
            (
                3,
                sp.Eq(diagonal, False) & treat_boundary & sp.Eq(r_cell + 1, max_rad),
            ),
            (0, True),
        ),
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
    kernel.append(
        (
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
        )
    )
    kernel.append(
        (
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
        )
    )
    kernel.append(
        (
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
        )
    )

for w in range(num_wedges_per_hex_cell):
   
    dsts_w = sp.Matrix.zeros(num_nodes_per_wedge, 1)

    for q in range(num_qps):
        dsts_qw = sp.Matrix.zeros(num_nodes_per_wedge, 1)

        ### Jacobian
        J = jac_from_array(wedge[w], rads[0], rads[1], qp_data[q])

        # CSE on jacobian
        J_cse_assignments, J_cse_exprs = flattened_cse(J, f"w{w}_q{q}_tmpcse_J_")
        kernel += J_cse_assignments

        # replace Jacobian entries with tmp symbols to speed up inversion
        J_cse = sp.Matrix(3, 3, J_cse_exprs)
        J_cse_replaced_assignments, J_cse_replaced = replace_matrix(J_cse, f"w{w}_q{q}_J")
        kernel += J_cse_replaced_assignments

        # compute abs determinant + CSE
        J_det = J_cse_replaced.det()
        J_det_replacements, J_det_reduced_exprs = sp.cse(
            exprs=J_det, symbols=numbered_symbols(prefix=f"w{w}_q{q}_tmpcse_det_")
        )
        kernel += J_det_replacements
        J_det_symbol = sp.symbols(f"w{w}_q{q}_J_det")
        kernel.append((J_det_symbol, J_det_reduced_exprs[0]))
        J_abs_det = abs(J_det_symbol)

        # invert + transpose jacobian
        J_inv = J_cse_replaced.adjugate() / J_det_symbol
        J_invT = J_inv.transpose()

        # second CSE after inversion
        J_invT_replacements, J_invT_reduced_exprs = flattened_cse(
            J_invT, f"w{w}_q{q}_tmpcse_J_invT_"
        )
        kernel += J_invT_replacements
        J_invT_cse = sp.Matrix(3, 3, J_invT_reduced_exprs)
        J_invT_cse_assignments, J_invT_cse_replaced = replace_matrix(
            J_invT_cse, f"w{w}_q{q}_J_invT_cse"
        )
        kernel += J_invT_cse_assignments

       
        # precompute gradients + CSE
        grad_is = []
        grad_is_symbols = []
        for i in range(num_nodes_per_wedge):
            grad_i = J_invT_cse_replaced * grad_shape_vec(i, qp_data[q])
            grad_i_symbol = sp.Matrix(
                sp.symbols(
                    f"w{w}_q{q}_grad_i{i}_0 w{w}_q{q}_grad_i{i}_1 w{w}_q{q}_grad_i{i}_2"
                )
            )
            grad_is.append(grad_i)
            grad_is_symbols.append(grad_i_symbol)
            
        grad_i_replacements, grad_i_reduced_exprs = sp.cse(
            exprs=grad_is,
            symbols=numbered_symbols(prefix=f"w{w}_q{q}_tmpcse_grad_i_"),
        )
        kernel += grad_i_replacements
        for grad_i, grad_i_symbol in zip(grad_i_reduced_exprs, grad_is_symbols):
            for gi, gis in zip(grad_i, grad_i_symbol):
                kernel.append((gis, gi))

        # assemble src/trial gradient (sum over rows in local matrix)
        srcs_w = sp.Matrix(srcs[w])
        grad_u_symbols = sp.Matrix(
            sp.symbols(f"w{w}_q{q}_grad_u_0 w{w}_q{q}_grad_u_1 w{w}_q{q}_grad_u_2")
        )
        grad_u = sp.Matrix.zeros(3, 1)

        for j in range(num_nodes_per_wedge):
            grad_u += srcs_w[j] * grad_is_symbols[j] * conditionals["trial"][j]
        for gu, gu_symbol in zip(grad_u, grad_u_symbols):
            kernel.append((gu_symbol, gu))
            
        for i in range(num_nodes_per_wedge):
            res = J_abs_det * qw_data[q] * grad_is_symbols[i].transpose() * grad_u_symbols  * conditionals["test"][i]
            dsts_qw[i] += res[0] 

        # post loop for bcs or diagonal kernel
        for i in range(num_nodes_per_wedge):
            grad_u_diag = srcs_w[i] * grad_is_symbols[i] * conditionals["diag_bc"][i]
            res = J_abs_det * qw_data[q] * grad_i.transpose() * grad_u_diag
            dsts_qw[i] += res[0] 

        # final cse
        dst_replacements, dst_reduced_exprs = sp.cse(
            exprs=dsts_qw,
            symbols=numbered_symbols(prefix=f"w{w}_q{q}_tmpcse_dst_", ignore=[Pow]),
        )
        kernel += dst_replacements
        for i in range(num_nodes_per_wedge):
            dsts_w[i] += dsts_qw[i]

        
   
    dsts_w_symbols = [
        sp.symbols(f"dst_{w}_{i}") for i in range(num_nodes_per_wedge)
    ]
    for i, dw in enumerate(dsts_w_symbols):
        kernel.append((dw, dsts_w[i]))
    dsts += [dsts_w_symbols]

kernel += make_atomic_add_local_wedge_scalar_coefficients(
    local_subdomain_id, x_cell, y_cell, r_cell, dsts
)

# Finally: print code
cpp_code = "\n// Kernel body:\n"


for stmt in kernel:
    # print(var_name)
    # print(expr)
    if isinstance(stmt, str):
        cpp_code += stmt
    else:
        var_name, expr = stmt

        # print(expr)
        def get_type(sym):
            if var_name.is_integer:
                return "int"
            else:
                return "double"

        if isinstance(expr, str):
            cpp_code += f"{  get_type(var_name) } {var_name} = {expr};\n"
        else:
            cpp_code += f"{  get_type(var_name) } {var_name} = {ccode(expr)};\n"

with open("Laplace_kernel", "w", encoding="utf-8") as f:
    f.write(cpp_code)

# print(print_atomic_add_local_wedge_scalar_coefficients())
