import sympy
from sympy.printing import ccode


def compute_linear_function_on_triangle(vertices, values):
    """
    Computes a linear function in x and y on a reference triangle given its
    values at the three vertices.

    Args:
        vertices (list of tuples): A list of the (x, y) coordinates of the
                                  three triangle vertices.
        values (list of numbers): A list of the function values at each
                                  corresponding vertex.

    Returns:
        sympy.Expr: The computed linear function as a SymPy expression.
    """
    if len(vertices) != 3 or len(values) != 3:
        raise ValueError("Exactly three vertices and three values must be provided.")

    # Define symbolic variables for the coefficients and coordinates
    a, b, c, xi, eta = sympy.symbols('a b c xi_fine eta_fine')

    # Define the general form of the linear function
    linear_function = a * xi + b * eta + c

    # Create a system of three linear equations using the vertex constraints
    equations = []
    for (vx, vy), value in zip(vertices, values):
        equations.append(sympy.Eq(linear_function.subs({xi: vx, eta: vy}), value))

    # Solve the system of equations for the coefficients a, b, and c. [12]
    solution = sympy.solve(equations, (a, b, c))

    if not solution:
        raise ValueError("Could not find a unique solution for the given constraints.")

    # Substitute the solved coefficients back into the general linear function
    return linear_function.subs(solution)


if __name__ == '__main__':
    # Define the reference triangle vertices
    # You can change these to any three non-collinear points
    triangle_vertices = [(0, 0), (1, 0), (0, 1)]

    # (coarse_node_idx, tri_idx) -> interpolation points
    vv = {
        0: {
            0: [1, 0.5, 0.5],
            1: [0.5, 0, 0],
            2: [0.5, 0, 0],
            3: [0, 0.5, 0.5],
        },
        1: {
            0: [0, 0.5, 0],
            1: [0.5, 1, 0.5],
            2: [0, 0.5, 0],
            3: [0.5, 0, 0.5],
        },
        2: {
            0: [0, 0, 0.5],
            1: [0, 0, 0.5],
            2: [0.5, 0.5, 1],
            3: [0.5, 0.5, 0],
        }
    }

    print(f"switch (coarse_node_idx % 3) {{")
    for coarse_node_idx, values in vv.items():

        print(f"case {coarse_node_idx}:")
        print(f"switch (fine_lateral_wedge_idx) {{")
        for tri_idx, values_at_vertices in values.items():

            # Define the desired values at each vertex
            # These are the constraints for the linear function
            # values_at_vertices = [1, 0, 0]

            # print(f"Computing a linear function on a triangle with vertices at {triangle_vertices}")
            # print(f"and corresponding values {values_at_vertices}\n")

            try:
                # Compute the linear function based on the constraints
                resulting_function = compute_linear_function_on_triangle(
                    triangle_vertices, values_at_vertices
                )

                # print("The computed linear function is:")
                # sympy.pprint(resulting_function)
                print(f"case {tri_idx}:")
                print(f"return {ccode(resulting_function)};")

                # You can now use this function for further calculations
                # For example, find the value at a specific point, e.g., (0.5, 0.5)
                # point_to_evaluate = (0.5, 0.5)
                # xi_sym, eta_sym = sympy.symbols('xi eta')
                # value_at_point = resulting_function.subs(
                #     {xi_sym: point_to_evaluate[0], eta_sym: point_to_evaluate[1]}
                # )
                # print(f"\nThe value of the function at {point_to_evaluate} is: {value_at_point}")

            except (ValueError, TypeError) as e:
                print(f"An error occurred: {e}")

        print(f"default: return 0.0;")
        print(f"}}")
    print(f"default: return 0.0;")
    print(f"}}")

    print()
    print()

    for sym in ["xi_fine", "eta_fine"]:

        print()

        print(f"// derivatives in {sym} direction")
        print(f"switch (coarse_node_idx % 3) {{")
        for coarse_node_idx, values in vv.items():

            print(f"case {coarse_node_idx}:")
            print(f"switch (fine_lateral_wedge_idx) {{")
            for tri_idx, values_at_vertices in values.items():

                # Define the desired values at each vertex
                # These are the constraints for the linear function
                # values_at_vertices = [1, 0, 0]

                # print(f"Computing a linear function on a triangle with vertices at {triangle_vertices}")
                # print(f"and corresponding values {values_at_vertices}\n")

                try:
                    # Compute the linear function based on the constraints
                    resulting_function = compute_linear_function_on_triangle(
                        triangle_vertices, values_at_vertices
                    )

                    # print("The computed linear function is:")
                    # sympy.pprint(resulting_function)
                    print(f"case {tri_idx}:")
                    print(f"return {ccode(resulting_function.diff(sym))};")

                    # You can now use this function for further calculations
                    # For example, find the value at a specific point, e.g., (0.5, 0.5)
                    # point_to_evaluate = (0.5, 0.5)
                    # xi_sym, eta_sym = sympy.symbols('xi eta')
                    # value_at_point = resulting_function.subs(
                    #     {xi_sym: point_to_evaluate[0], eta_sym: point_to_evaluate[1]}
                    # )
                    # print(f"\nThe value of the function at {point_to_evaluate} is: {value_at_point}")

                except (ValueError, TypeError) as e:
                    print(f"An error occurred: {e}")

            print(f"default: return 0.0;")
            print(f"}}")
        print(f"default: return 0.0;")
        print(f"}}")
