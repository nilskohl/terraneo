# Linear algebra {#linear-algebra}

All linear algebra functionality (coefficient vectors, matrices, linear solvers) works on top of a few C++ concepts
defining a simple interface:

* \ref terra::linalg::VectorLike for coefficient vectors
* \ref terra::linalg::OperatorLike for linear operators (sparse matrices)
* \ref terra::linalg::solvers::SolverLike for linear solvers

\note I have not figured out how to display docstrings referring to concept requirements in Doxygen yet. Check the
source code for more details.

### Coefficient vectors

Coefficient vectors that implement the \ref terra::linalg::VectorLike concept are (for hexahedral or wedge meshes)

* \ref terra::linalg::VectorQ1Scalar (coefficient vector for scalar, linear finite element spaces)
* \ref terra::linalg::VectorQ1Vec (coefficient vector for vector-valued, linear finite element spaces)
* \ref terra::linalg::VectorQ1IsoQ2Q1 (coefficient vector the mixed, linear Q1-iso-Q2 / Q1 finite element space - stable
  pairing for Stokes)

Vector-vector operations are found as free functions in the \ref terra::linalg namespace (see \ref vector.hpp
particularly). Use those functions instead of the members of the classes implementing the concept.

Note that the linear algebra functionality may hide implementation details of the coefficient vectors.
For instance, the underlying grid may store duplicate nodes at subdomain boundaries. These duplicated nodes are hidden
from the user through the \ref terra::linalg::VectorLike interface and dealt with internally.

If you want to work directly on the underlying grid data, check out the kernels in \ref grid_operations.hpp.

### Operators

Matrices should implement the \ref terra::linalg::OperatorLike concept. Note that the concept mostly only requires 
an implementation for a matrix-vector product. The concept is meant to primarily target matrix-free implementations
of linear operators. Some concrete implementations are found in the namespace \ref terra::fe::wedge::operators::shell.

### Solvers

Linear solvers and preconditioners should implement the \ref terra::linalg::solvers::SolverLike concept and rely on the
arguments being coefficient vectors implementing the \ref terra::linalg::VectorLike concept and operators implementing 
the \ref terra::linalg::OperatorLike concept.

Preconditioners are just other classes that also implement the \ref terra::linalg::solvers::SolverLike concept. 
They can then usually be passed to a solver as an optional argument.

See the namespace \ref terra::linalg::solvers for concrete implementations of solvers and preconditioners.

Note that many solvers require passing in coefficient vectors that are required as temporary storage internally.
To ensure that no large memory allocations are hidden from the user, the coefficient vectors have to be allocated 
manually in the application code and passed to the solver.

### Example code

Have a look at the tests to see how to use the linear algebra functionality.

