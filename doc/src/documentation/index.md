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

## Framework documentation

### Model / Partial differential equations

🏗️

-----------

### Grid structure and subdomains (part I - logical structure)

\note This section is just describing the logical organization of data. For details on actual data layout / allocation
      refer to the Kokkos section. Details on the construction of the thick spherical shell and communication are given
      in dedicated sections below.

All grid operations are performed on a set of hexahedral subdomains. Those are organized as 4- or 5 dimensional arrays.

```
    data( local_subdomain_id, node_x, node_y, node_r )                     // for scalar data
    data( local_subdomain_id, node_x, node_y, node_r, vector_entry )       // for vector-valued data 
```

The first index specifies the subdomain. Then the x, y, r coordinates of the nodes (we are using r instead of z because
we will for our application most of the time think in radial directions). For vector-valued data (think velocity vectors
stored at each node) a fifth index specifies the entry of the vector. In the code, the size of that fifth dimension is
typically a compile-time constant.

This layout forces all subdomains to be of equal size.

As described in the finite element section, we mostly use wedge elements. The nodes that span wedge elements have the
same spatial organization of nodes required for hexahedral elements.
The division of one hexahedron into two wedges is implicitly done in the compute kernels and is not represented by the
grid data structure.

(Since we are using Lagrangian basis functions nodes directly correspond to coefficients. For the linear cases, we can
also use the same grid data structure for linear hexahedral finite elements. Such an extension is straightforward and
just requires respective kernels. One could even mix both - although it is not clear if that is mathematically sound.)

As a convention, the hexahedral elements are split into two wedges diagonally from node (1, 0) to node (0, 1) as 
follows:

```
Example of a 5x4 subdomain, that would be extruded in r-direction.
Each node 'o' either stores a scalar or a vector. 

        o---o---o---o---o
        |\  |\  |\  |\  |
        | \ | \ | \ | \ |
        o---o---o---o---o
        |\  |\  |\  |\  |
        | \ | \ | \ | \ |
        o---o---o---o---o
   ^    |\  |\  |\  |\  |
   |    | \ | \ | \ | \ |
  y|    o---o---o---o---o
   
        -->
        x
```

Some helper functions to work on this grid structure are supplied in `kernel_helpers.hpp`. 

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

#### Coefficient vectors

🏗️

#### Operators

🏗️

#### Solvers

🏗️

#### Multigrid

🏗️

-----------

### Grid structure and subdomains (part II - Kokkos)

🏗️

-----------

### Thick spherical shell

The Earth mantle is approximated via a thick spherical shell \f$\Omega\f$ , i.e., a hollow sphere centered at the origin

\f[ \Omega = \{\mathbf{x} \in \mathbb{R}^3 : r_\mathrm{min} \leq \|\mathbf{x}\| \leq r_\mathrm{max} \} \f]

#### Mesh structure

A corresponding mesh is constructed by splitting the outer surface of \f$\Omega\f$ into 10 spherical diamonds that are 
extruded towards (or equivalently away from) the origin.

The figure below shows the 10 diamonds in a three-dimensional visualization: 

TODO...

Unfolding the surface partitioning, we can visualize the surface of the 10 spherical diamonds as a net that when curved 
and pieced together recovers the spherical shell: 

\image html figures/thick-spherical-shell-diamond-net.jpg

Note that the extrusion in radial direction is not visible from the net.

Each diamond can (optionally) be subdivided in lateral and radial direction. The radial refinement is straightforward.
After (uniform) lateral refinement, each subdomain can be associated with a globally unique identifying tuple

``` 
subdomain_id = (diamond_id, subdomain_x, subdomain_y, subdomain_r)
```

as illustrated in the figure below:

\image html figures/thick-spherical-shell-subdomains.jpg

The `subdomain_id` is implemented in the class \ref terra::grid::shell::SubdomainInfo.

A domain can be set up using the \ref terra::grid::shell::DomainInfo class.
This does not compute any node coordinates. It just stores the refinement information, i.e., how many subdomains
are present for each diamond in either direction.
In lateral direction, refinement currently has to be uniform.
In radial direction, the concrete radii of the layers can be specified.
For more details refer to the documentation of \ref terra::grid::shell::DomainInfo.

#### Local subdomains

Subdomains on the same MPI process are sorted by their global `subdomain_id` (it is sortable and globally unique)
and continuously assigned to an integer `local_subdomain_id` that ranges from 0 to the number of process-local 
subdomains minus 1.
The `local_subdomain_id` is then the first index of the 4D (or 5D) data grids introduced above.
For instance for a scalar data array `data` the expression
```
    data( 3, 55, 20, 4 )
```
accesses the node with
```
    local_subdomain_id =  3
    x_index            = 55
    y_index            = 20
    r_index            =  4
```
The mapping from the `subdomain_id` (`SubdomainInfo`) to the `local_subdomain_id` (`int`) is performed during set up
and stored together with other information in the corresponding \ref terra::grid::shell::DistributedDomain instance.
More details can be found the parallelization section.

#### Node coordinates

The concrete coordinates of the nodes are computed with two functions:
* \ref terra::grid::shell::subdomain_unit_sphere_single_shell_coords - computes the "lateral cartesian coordinates"
  of all nodes, i.e., computes the cartesian coordinates of a single shell of nodes with radius 1 and returns them in
  a 4D array `coords_shell( local_subdomain_id, x_index, y_index, cartesian_coord )`.
* \ref terra::grid::shell::subdomain_shell_radii - computes the radii and stores them in a 2D array
  `coords_radii( local_subdomain_id, r_index )`
The cartesian coordinate of a node `( local_subdomain_id, x_index, y_index, r_index )` can then be computed via
```
    Vec3 cartesian_coords;
    cartesian_coords( 0 ) = coords_shell( local_subdomain_id, x_index, y_index, 0 );
    cartesian_coords( 1 ) = coords_shell( local_subdomain_id, x_index, y_index, 1 );
    cartesian_coords( 2 ) = coords_shell( local_subdomain_id, x_index, y_index, 2 );
    return cartesian_coords * coords_radii( local_subdomain_id, r_index );
```
This is implemented in \ref terra::grid::shell::coords.
The radius is obviously just `coords_radii( local_subdomain_id, r_index )`.

-----------

### Parallelization and communication

🏗️

-----------

### Flag fields and masks

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
A great explanation is given in [Wolfgang Bangerth's
lectures](https://www.math.colostate.edu/~bangerth/videos.676.21.65.html).

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

#### Tabular data

🏗️

#### XDMF

\note Putting the string 'VTK' here in case you are looking for it via full-text search. We are using XDMF instead.

🏗️

#### Radial profiles

🏗️
