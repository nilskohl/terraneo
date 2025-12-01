import sympy as sp
from sympy.codegen.ast import Assignment
from sympy import *
import sympy as sp


def make_float_symbol(name):
    return sp.symbols(name, real=True, finite=True)


def make_wedge_surface_physical_coord_assignments(local_subdomain_id, x_cell, y_cell):
    num_wedges_per_hex_cell = 2
    num_nodes_per_wedge_surface = 3
    dim = 3

   
    # lateral_grid(local_subdomain_id, x_cell+i, y_cell+j, d)
    lateral_grid = [[[[
        f"lateral_grid({local_subdomain_id},{x_cell} + {i},{y_cell} + {j},{d})"
        for d in range(dim)
    ] for j in range(2)] for i in range(2)]]

    # quad_surface_coords
    quad_surface_coords = [[[sp.symbols(f"quad_surface_coords_{i}_{j}_{d}")
                             for d in range(dim)]
                            for j in range(2)]
                           for i in range(2)]

    # wedge_surf_phy_coords
    wedge_surf_phy_coords = [[[sp.symbols(f"wedge_surf_phy_coords_{w}_{n}_{d}")
                               for d in range(dim)]
                              for n in range(num_nodes_per_wedge_surface)]
                             for w in range(num_wedges_per_hex_cell)]

    assignments = []

    # Populate quad surface coordinates
    for i in range(2):
        for j in range(2):
            for d in range(dim):
                assignments.append((
                    quad_surface_coords[i][j][d],
                    lateral_grid[0][i][j][d]
                ))

    # Mapping from C++ wedge logic
    mapping = [
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (0, 2, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 2, 1, 0),
    ]

    for w, n, qi, qj in mapping:
        for d in range(3):
            assignments.append((
                wedge_surf_phy_coords[w][n][d],
                quad_surface_coords[qi][qj][d]
            ))

    return quad_surface_coords, wedge_surf_phy_coords, lateral_grid, assignments


def make_rad_assignments(local_subdomain_id, r_cell):
    assignments = []
    rads = [sp.symbols(f"r_{i}") for i in range(2)]
    rads_array_accesses = [f"radii_(local_subdomain_id, r_cell)" for i in range(2)]
    for i in range(2):
            assignments.append((
                rads[i],
                rads_array_accesses[i]
            ))
    return rads, assignments

def make_quad_assignments(quad_points, quad_weights):
    
    qp_symbols = []
    qw_symbols = []
    assignments = []
    num_qps = 0
    for i, pt in enumerate(quad_points):
        for j, c in enumerate(pt):
            qp = symbols(f"qp_{i}_{j}")
            qp_symbols.append(qp)
            assignments.append((qp, c))
            
    for i, pt in enumerate(quad_weights):
        num_qps += 1
        qw = symbols(f"qw_{i}")
        qw_symbols.append(qw)
        assignments.append((qw, pt))
            
    return num_qps, qp_symbols, qw_symbols, assignments

def make_extract_local_wedge_scalar_assignments(local_subdomain_id, x_cell, y_cell, r_cell):
    # Indices (treated as integer parameters)
   
    # Input loads
    def G(dx, dy, dr):
        return f"src_({local_subdomain_id},{x_cell+dx},{y_cell+dy},{r_cell+dr})"

    # Output stores
    def L(w, i):
        return sp.symbols(
            f"src_{w}_{i}",
            real=True
        )

    assigns = []
    src_symbols = []

    # ---- wedge 0 ----
    assigns += [
        (L(0,0), G(0,0,0)),
        (L(0,1), G(1,0,0)),
        (L(0,2), G(0,1,0)),
        (L(0,3), G(0,0,1)),
        (L(0,4), G(1,0,1)),
        (L(0,5), G(0,1,1)),
    ]
    src_symbols += [[L(0,i) for i in range(6)]]

    # ---- wedge 1 ----
    assigns += [
        (L(1,0), G(1,1,0)),
        (L(1,1), G(0,1,0)),
        (L(1,2), G(1,0,0)),
        (L(1,3), G(1,1,1)),
        (L(1,4), G(0,1,1)),
        (L(1,5), G(1,0,1)),
    ]
    src_symbols += [[L(1,i) for i in range(6)]]

    return src_symbols, assigns

def print_atomic_add_local_wedge_scalar_coefficients(local_subdomain_id, x_cell, y_cell, r_cell, dsts):
    lines = []

    def A(dx, dy, dr):
        return f"dst_({local_subdomain_id}, {x_cell + dx}, {y_cell + dy}, {r_cell + dr})"

    def L(w, i):
        return dsts[w][i]

    # atomic add statements (global index, RHS expression)
    ops = [
        ((0,0,0),  f"{L(0,0)}"),
        ((1,0,0),  f"{L(0,1)} + {L(1,2)}"),
        ((0,1,0),  f"{L(0,2)} + {L(1,1)}"),
        ((0,0,1),  f"{L(0,3)}"),
        ((1,0,1),  f"{L(0,4)} + {L(1,5)}"),
        ((0,1,1),  f"{L(0,5)} + {L(1,4)}"),
        ((1,1,0),  f"{L(1,0)}"),
        ((1,1,1),  f"{L(1,3)}"),
    ]

    for (dx,dy,dr), rhs in ops:
        g = A(dx,dy,dr)
        lines.append(
            f"Kokkos::atomic_add(&{g}, {rhs});\n"
        )

    return lines

# Example: Felippa 3x2 quadrature points
quad_points_3x2= [
    [0.6666666666666666, 0.1666666666666667, -0.5773502691896257],
    [0.1666666666666667, 0.6666666666666666, -0.5773502691896257],
    [0.1666666666666667, 0.1666666666666667, -0.5773502691896257],
    [0.6666666666666666, 0.1666666666666667, 0.5773502691896257],
    [0.1666666666666667, 0.6666666666666666, 0.5773502691896257],
    [0.1666666666666667, 0.1666666666666667, 0.5773502691896257]
]
quad_weights_3x2 = [
    [0.1666666666666667],
    [0.1666666666666667],
    [0.1666666666666667],
    [0.1666666666666667],
    [0.1666666666666667],
    [0.1666666666666667]
]
quad_points_1x1 = [
     [1.0 / 3.0, 1.0 / 3.0, 0.0]
]
quad_weights_1x1 = [
     1.0 
]

from sympy import symbols, IndexedBase
from sympy.tensor.indexed import Idx

def make_hex_assignments( local_subdomain_id, x_cell, y_cell, r_cell, prefix="src_local_hex"):
    """
    Generate unrolled assignments for src_local_hex based on the given offsets.
    
    Returns:
        list of (lhs_symbol, rhs_expression) pairs
    """
    # Define symbolic variables
    src_local_hex_symbols = symbols([f"{prefix}_{i}" for i in range(8)])
    
    # Define offset arrays
    hex_offset_x = [0, 1, 0, 1, 0, 1, 0, 1]
    hex_offset_y = [0, 0, 1, 1, 0, 0, 1, 1]
    hex_offset_r = [0, 0, 0, 0, 1, 1, 1, 1]
    
    assignments = []
    
    for i in range(8):
        lhs = src_local_hex_symbols[i]
        # Symbolically represent the function call
        rhs = f"src_(local_subdomain_id, {x_cell} + {hex_offset_x[i]}, {y_cell} + {hex_offset_y[i]}, {r_cell} + {hex_offset_r[i]})"
        assignments.append((lhs, rhs))
    
    return src_local_hex_symbols, assignments