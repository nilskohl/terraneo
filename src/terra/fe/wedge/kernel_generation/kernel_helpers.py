import sympy as sp
from sympy.codegen.ast import Assignment
from sympy import *
import sympy as sp
from sympy.codegen.ast import (
    Assignment,
    For,
    CodeBlock,
    Variable,
    Declaration,
    Pointer,
    integer,
    float64,
    FunctionCall,
)


def make_float_symbol(name):
    return sp.symbols(name, real=True, finite=True)


def make_wedge_surface_physical_coord_assignments(local_subdomain_id, x_cell, y_cell):
    num_wedges_per_hex_cell = 2
    num_nodes_per_wedge_surface = 3
    dim = 3

    # lateral_grid(local_subdomain_id, x_cell+i, y_cell+j, d)
    lateral_grid = [
        [
            [
                [
                    FunctionCall(
                        "src_", [local_subdomain_id, x_cell + i, y_cell + j, d]
                    )
                    for d in range(dim)
                ]
                for j in range(2)
            ]
            for i in range(2)
        ]
    ]

    # quad_surface_coords
    quad_surface_coords = [
        [
            [
                sp.symbols(
                    f"quad_surface_coords_{i}_{j}_{d}", real=True
                )
                for d in range(dim)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]

    # wedge_surf_phy_coords
    wedge_surf_phy_coords = [
        [
            [
                sp.symbols(
                    f"wedge_surf_phy_coords_{w}_{n}_{d}", real=True
                )
                for d in range(dim)
            ]
            for n in range(num_nodes_per_wedge_surface)
        ]
        for w in range(num_wedges_per_hex_cell)
    ]

    assignments = []

    # Populate quad surface coordinates
    for i in range(2):
        for j in range(2):
            for d in range(dim):
                assignments.append(
                    Variable.deduced(quad_surface_coords[i][j][d]).as_Declaration(value= lateral_grid[0][i][j][d])
                )
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
            assignments.append(
                Variable.deduced(wedge_surf_phy_coords[w][n][d]).as_Declaration(value= quad_surface_coords[qi][qj][d])
            )

    return (
        quad_surface_coords,
        wedge_surf_phy_coords,
        lateral_grid,
        CodeBlock(*assignments),
    )


def make_rad_assignments(local_subdomain_id, r_cell):
    assignments = []
    rads = [sp.symbols(f"r_{i}", real=True) for i in range(2)]

    for i in range(2):
        Variable.deduced(rads[i]).as_Declaration(value= FunctionCall("radii_", [r_cell + i]))
    return rads, CodeBlock(*assignments)


def make_extract_local_wedge_scalar_assignments(
    local_subdomain_id, x_cell, y_cell, r_cell
):
    # Indices (treated as integer parameters)
    rads = [sp.symbols(f"r_{i}", real=True) for i in range(2)]

    # Input loads
    def G(dx, dy, dr):
        return FunctionCall(
            "src_", [local_subdomain_id, x_cell + dx, y_cell + dy, r_cell + dr]
        )

    # Output stores
    def L(w, i):
        return sp.symbols(f"src_{w}_{i}", real=True)

    assigns = []
    src_symbols = []

    # ---- wedge 0 ----
    assigns += [
        Declaration(Variable.deduced(L(0, 0))),
        Assignment(L(0, 0), G(0, 0, 0)),
        Declaration(Variable.deduced(L(0, 1))),
        Assignment(L(0, 1), G(1, 0, 0)),
        Declaration(Variable.deduced(L(0, 2))),
        Assignment(L(0, 2), G(0, 1, 0)),
        Declaration(Variable.deduced(L(0, 3))),
        Assignment(L(0, 3), G(0, 0, 1)),
        Declaration(Variable.deduced(L(0, 4))),
        Assignment(L(0, 4), G(1, 0, 1)),
        Declaration(Variable.deduced(L(0, 5))),
        Assignment(L(0, 5), G(0, 1, 1)),
    ]
    src_symbols += [[L(0, i) for i in range(6)]]

    # ---- wedge 1 ----
    assigns += [
        Declaration(Variable.deduced(L(1, 0))),
        Assignment(L(1, 0), G(1, 1, 0)),
        Declaration(Variable.deduced(L(1, 1))),
        Assignment(L(1, 1), G(0, 1, 0)),
        Declaration(Variable.deduced(L(1, 2))),
        Assignment(L(1, 2), G(1, 0, 0)),
        Declaration(Variable.deduced(L(1, 3))),
        Assignment(L(1, 3), G(1, 1, 1)),
        Declaration(Variable.deduced(L(1, 4))),
        Assignment(L(1, 4), G(0, 1, 1)),
        Declaration(Variable.deduced(L(1, 5))),
        Assignment(L(1, 5), G(1, 0, 1)),
    ]
    src_symbols += [[L(1, i) for i in range(6)]]

    return src_symbols, CodeBlock(*assigns)


def make_atomic_add_local_wedge_scalar_coefficients(
    local_subdomain_id, x_cell, y_cell, r_cell, dsts
):
    lines = []

    def A(dx, dy, dr):
        return (
            f"dst_({local_subdomain_id}, {x_cell + dx}, {y_cell + dy}, {r_cell + dr})"
        )

    def L(w, i):
        return dsts[w][i]

    # atomic add statements (global index, RHS expression)
    ops = [
        ((0, 0, 0), f"{L(0,0)}"),
        ((1, 0, 0), f"{L(0,1)} + {L(1,2)}"),
        ((0, 1, 0), f"{L(0,2)} + {L(1,1)}"),
        ((0, 0, 1), f"{L(0,3)}"),
        ((1, 0, 1), f"{L(0,4)} + {L(1,5)}"),
        ((0, 1, 1), f"{L(0,5)} + {L(1,4)}"),
        ((1, 1, 0), f"{L(1,0)}"),
        ((1, 1, 1), f"{L(1,3)}"),
    ]

    for (dx, dy, dr), rhs in ops:
        g = A(dx, dy, dr)
        lines.append(f"Kokkos::atomic_add(&{g}, {rhs});\n")

    return lines


def make_boundary_handling(matrix_name):
    return f"""
        if ( treat_boundary_ )
        {{
                if ( r_cell == 0 )
                {{
                    
                    // Inner boundary (CMB).
                    {
                        chr(10).join([f"{matrix_name}_{i}_{j} = 0.0;" for i in range(6) for j in range(6) if (i != j and (i < 3 or j < 3 ))])
                    }
                }}

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                {{
                 {
                        chr(10).join([f"{matrix_name}_{i}_{j} = 0.0;" for i in range(6) for j in range(6) if (i != j and (i >= 3 or j >= 3 ))])
                    }
                }}
        }}
        """


def make_diagonal_handling(matrix_name):
    return f"""
        if ( diagonal_ )
        {{
                {{
                 {
                        chr(10).join([f"{matrix_name}_{i}_{j} = 0.0;" for i in range(6) for j in range(6) if (i != j)])
                    }
                }}
        }}
        """


# Example: Felippa 3x2 quadrature points
quad_points_3x2 = [
    [0.6666666666666666, 0.1666666666666667, -0.5773502691896257],
    [0.1666666666666667, 0.6666666666666666, -0.5773502691896257],
    [0.1666666666666667, 0.1666666666666667, -0.5773502691896257],
    [0.6666666666666666, 0.1666666666666667, 0.5773502691896257],
    [0.1666666666666667, 0.6666666666666666, 0.5773502691896257],
    [0.1666666666666667, 0.1666666666666667, 0.5773502691896257],
]
quad_weights_3x2 = [
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
]
quad_points_1x1 = [[1.0 / 3.0, 1.0 / 3.0, 0.0]]
quad_weights_1x1 = [1.0]

from sympy import symbols, IndexedBase
from sympy.tensor.indexed import Idx


def make_hex_assignments(
    local_subdomain_id, x_cell, y_cell, r_cell, prefix="src_local_hex"
):
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
