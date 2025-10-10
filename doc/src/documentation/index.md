# Documentation {#main}

## About

🏗️

## Building

### Dependencies

Mandatory:

* MPI (e.g. OpenMPI)

Optional:

* CUDA (for GPU support)

### Compiling on the LMU systems (`cachemiss`, `memoryleak`) for usage with CUDA:

```
$ module load mpi.ompi
$ module load nvidia-hpc

$ mkdir terraneox-build

$ ll
terraneox/               # <== the cloned source code
terraneox-build/

$ cd terraneox-build

$ cmake ../terraneox/ -Kokkos_ENABLE_CUDA=ON

# Build tests
$ cd tests
$ make -j16
```

Note the capitalization: it must be `Kokkos_ENABLE_CUDA=ON`, NOT `KOKKOS_ENABLE_CUDA=ON`.

## Project Structure

```
terraneox/
├── benchmarks/               # Benchmarks ...
│   ├── performance/          # ... run time
│   └── physics/              # ... physics
├── data/                     # Stuff that is not exactly library/framework source code
│   └── scripts/              # Scripts, e.g., for post-processing
│       └── plotting/         # Scripts for visualization, e.g., timing, radial profiles, ... 
├── doc/                      # Documentation
│   ├── documents/            # Documents that are not part of the generated documentation (e.g., Latex documents)
│   └── src/                  # Doxygen documentation
│       └── documentation/    # Doxygen documentation source (this page is likely here)
├── extern/                   # External libraries shipped with this repository
├── src/                      # Source code
└── tests/                    # Tests
```

## Topics

### Model / Partial differential equations

🏗️

-----------

### Finite element discretization

The partial differential equations and their solutions are approximated using the finite element methods.

We are using linear wedge elements for all spaces, unless specified otherwise. This is very similar to the 
implementation in Terra.

\note
See helper functions and documentation in [integrands.hpp](@ref integrands.hpp) or the 
[namespace terra::fe:wedge](@ref terra::fe::wedge) for details, and other, derived
quantities like gradients, Jacobians, determinants, etc.

Linear wedge (or prismatic) elements are formed by extruding a linear triangular element in the radial direction.
The base triangle lies in the lateral plane (parameterized by \f$\xi\f$,\f$\eta\f$), while the extrusion occurs along the radial
coordinate \f$\zeta\f$.

\note
The provided functions for the computation of the gradients, Jacobians, etc. assume that we are working on a spherical
shell and accordingly work with respect to a forward map that maps the reference element onto a wedge with the two
triangular surfaces living on two shell-slices, and the connecting beams being radially extruded from the origin.

#### Geometry

Lateral reference coordinates:
  \f[ \xi, \eta \in [0, 1] \f]

Radial reference coordinates:
  \f[ \zeta \in [-1, 1] \f]

With
  \f[ 0 \leq \xi + \eta \leq 1 \f]

#### Node enumeration

  \code

  r_node_idx = r_cell_idx + 1 (outer):

  5
  |\
  | \
  3--4
  \endcode

  \code

  r_node_idx = r_cell_idx (inner):

  2
  |\
  | \
  0--1
  \endcode

#### Shape functions

Lateral:

  \f[
  \begin{align}
    N^\mathrm{lat}_0 = N^\mathrm{lat}_3 &= 1 - \xi - \eta \\
    N^\mathrm{lat}_1 = N^\mathrm{lat}_4 &= \xi \\
    N^\mathrm{lat}_2 = N^\mathrm{lat}_5 &= \eta
  \end{align}
  \f]

Radial:

  \f[
  \begin{align}
    N^\mathrm{rad}_0 = N^\mathrm{rad}_1 = N^\mathrm{rad}_2 &= \frac{1}{2} ( 1 - \zeta ) \\
    N^\mathrm{rad}_3 = N^\mathrm{rad}_4 = N^\mathrm{rad}_5 &= \frac{1}{2} ( 1 + \zeta ) \\
  \end{align}
  \f]

Full:

  \f[
  N_i = N^\mathrm{lat}_i N^\mathrm{rad}_i
  \f]


#### Physical coordinates

  \code
  r_1, r_2                     radii of bottom and top (r_1 < r_2)
  p1_phy, p2_phy, p3_phy       coords of triangle on unit sphere
  \endcode


#### Spaces:

For the Stokes system we employ the stable (\f$P_1\f$-iso-\f$P_2\f$, \f$P_1\f$) finite element pairing, i.e., both
the velocity and pressure are discretized with linear wedge elements, with the velocity living on a grid with additional
refinement compared to the pressure grid.

-----------

### Linear algebra

🏗️

-----------

### Kokkos

🏗️

-----------

### Grid structure / Thick spherical shell

🏗️

-----------

### Parallelization and communication

🏗️

-----------

### Boundary conditions

#### Strong Dirichlet boundary condition enforcement

\note
See
also: [strong_algebraic_dirichlet_enforcement_poisson_like()](@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like)
and similar functions in that file.

<p></p>

\note
A great explanation is given in Wolfgang Bangerth's
lectures: https://www.math.colostate.edu/~bangerth/videos.676.21.65.html

Consider the linear problem

\f[ Lu = f \f]

with Dirichlet boundary conditions.

We approach the elimination as follows (assuming interpolating FE spaces).

Let \f$ A \f$ be the "Neumann" operator matrix of \f$ L \f$, i.e., we do not treat the boundaries any differently and
just execute the volume integrals.

1. Interpolate Dirichlet boundary conditions into a vector \f$ g \f$.
2. Compute \f$ g_A \gets A g \f$.
3. Compute \f$ g_D \gets \mathrm{diag}(A) g \f$
4. Set the rhs to \f$ b_\text{elim} = b - g_A \f$,
   where \f$ b \f$ is the assembled rhs vector for the homogeneous problem
   (the result of evaluating the linear form into a vector or of the matrix-vector product of a vector
   \f$ f_\text{vec} \f$ where
   the rhs function \f$ f \f$ has been interpolated into, and then \f$ b = M f_\text{vec} \f$ (\f$ M \f$ being the mass
   matrix))
5. Set the rhs \f$ b_elim \f$ at the boundary nodes to \f$ g_D \f$, i.e.
   \f$ b_\text{elim} \gets g_D \f$ on the Dirichlet boundary
6. Solve
   \f$ A_\text{elim} x = b_\text{elim} \f$
   where \f$ A_\text{elim} \f$ is \f$ A \f$, but with all off-diagonal entries in the same row/col as a boundary node
   set to zero.
   This feature has to be supplied by the operator implementation.
   In a matrix-free context, we have to adapt the element matrix \f$ A_\text{local} \f$ accordingly by (symmetrically)
   zeroing
   out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
   We still have \f$ \mathrm{diag}(A) = \mathrm{diag}(A_\text{elim}) \f$ .
7. \f$ x \f$ is the solution of the original problem. No boundary correction should be necessary.

All of this is covered
by [strong_algebraic_dirichlet_enforcement_poisson_like()](@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like)
and similar functions. What needs to be supplied are the operators \f$ A \f$, \f$ \mathrm{diag}(A) \f$, and
\f$ A_\text{elim} \f$. The main "difficulty" is to implement \f$ A_\text{elim} \f$ in a matrix-free kernel.
This can be achieved if one realizes that zeroing out the offdiagonals of the global matrix is equivalently done
by zeroing out the respective entries of the local element matrices.

#### Free-slip

🏗️

#### Plate boundaries

🏗️

-----------

### IO

#### Tablular data

🏗️

#### XDMF

\note Putting the string 'VTK' here in case you are looking for it via full-text search. We are using XDMF instead.

🏗️

#### Radial profiles

🏗️
