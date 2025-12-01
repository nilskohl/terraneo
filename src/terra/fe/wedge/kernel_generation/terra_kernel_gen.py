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
num_qps, qps, qws, qp_assignments = make_quad_assignments(
    quad_points_1x1, quad_weights_1x1
)
srcs, src_assignments = make_extract_local_wedge_scalar_assignments(
    local_subdomain_id, x_cell, y_cell, r_cell
)
# make_hex_assignments(local_subdomain_id, x_cell, y_cell, r_cell)
kernel += wedge_assignments
kernel += rad_assignments
#kernel += qp_assignments
kernel += src_assignments

qp_data = quad_points_1x1
qw_data = quad_weights_1x1

num_wedges_per_hex_cell = 2
num_nodes_per_wedge_surface = 3
num_nodes_per_wedge = 6
dim = 3
dsts = []
for qi in range(num_qps):
    for w in range(num_wedges_per_hex_cell):
        ### Jacobian
        J = jac_from_array(wedge[w], rads[0], rads[1], qp_data[qi])

        # 1. first CSE
        J_cse_assignments, J_cse_exprs = flattened_cse(J, f"w{w}_tmpcse_J_")
        kernel += J_cse_assignments

        # 2. replace entries with tmp symbols to speed up inversion
        J_cse = sp.Matrix(3, 3, J_cse_exprs)
        J_cse_replaced_assignments, J_cse_replaced = replace_matrix(J_cse, f"w{w}_J")
        kernel += J_cse_replaced_assignments

        # 3. invert
        J_invT = J_cse_replaced.inv().transpose()

        # 4. second CSE after inversion
        J_invT_replacements, J_invT_reduced_exprs = flattened_cse(
            J_invT, f"w{w}_tmpcse_J_invT_"
        )
        kernel += J_invT_replacements
        J_invT_cse = sp.Matrix(3, 3, J_invT_reduced_exprs)
        J_invT_cse_assignments, J_invT_cse_replaced = replace_matrix(
            J_invT_cse, f"w{w}_J_invT_cse"
        )
        kernel += J_invT_cse_assignments

        # 5. compute abs determinant + CSE
        J_abs_det = abs(J_cse_replaced.det())
        J_absdet_replacements, J_absdet_reduced_exprs = sp.cse(
            exprs=J_abs_det, symbols=numbered_symbols(prefix=f"w{w}_tmpcse_absdet_")
        )
        kernel += J_absdet_replacements
        absdet = sp.symbols(f"w{w}_absdet")
        kernel.append((absdet, J_absdet_reduced_exprs[0]))

        # 6. local mat CSE

        local_mat_exprs = []
        for i in range(num_nodes_per_wedge):
            for j in range(num_nodes_per_wedge):

                local_mat_ij = sp.symbols(f"w{w}_local_mat_{i}_{j}")

                # compute upper triangular part
                grad_i = J_invT_cse_replaced * grad_shape_vec(i, qp_data[qi])
                grad_j = J_invT_cse_replaced * grad_shape_vec(j, qp_data[qi])
                tmp = absdet * qw_data[qi] * grad_i.transpose() * grad_j
                local_mat_exprs.append(tmp[0])

        local_mat_replacements, local_mat_reduced_exprs = sp.cse(
            exprs=local_mat_exprs,
            symbols=numbered_symbols(prefix=f"w{w}_tmpcse_local_mat_"),
        )
        kernel += local_mat_replacements
        local_matrix = sp.Matrix(
            num_nodes_per_wedge, num_nodes_per_wedge, local_mat_reduced_exprs
        )

        matrix_name = f"w{w}_local_mat_replaced"

        local_mat_replaced_assignments, local_mat_replaced = replace_matrix(
            local_matrix, matrix_name
        )
        kernel += local_mat_replaced_assignments
        # print(local_mat_replaced)
        # for i in range(num_nodes_per_wedge):
        #    for j in range(0,i):
        #        kernel.append((local_mat_replaced[i, j], local_mat_replaced[j, i]))

        # print(local_mat_replaced)
        kernel.append(make_boundary_handling(matrix_name))
        kernel.append(make_diagonal_handling(matrix_name))

        dst_wedge_rhss = local_mat_replaced * sp.Matrix(srcs[w])
        dsts_wedge = [sp.symbols(f"dst_{w}_{i}") for i in range(num_nodes_per_wedge)]
        for dst, dst_rhs in zip(dsts_wedge, dst_wedge_rhss):
            kernel.append((dst, dst_rhs))
        dsts += [dsts_wedge]

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
        if isinstance(expr, str):
            cpp_code += f"double {var_name} = {expr};\n"
        else:
            cpp_code += f"double {var_name} = {ccode(expr)};\n"

with open("Laplace_kernel", "w", encoding="utf-8") as f:
    f.write(cpp_code)

# print(print_atomic_add_local_wedge_scalar_coefficients())
