
#pragma once

#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "terra/linalg/operator.hpp"

namespace terra::fe {

/// @brief Helper function to enforce Dirichlet conditions strongly by symmetric elimination for Poisson-like systems.
///
/// Consider the linear (Poisson-like) problem
///
/// \f[ Lu = f \f]
///
/// with Dirichlet boundary conditions.
///
/// We approach the elimination as follows (assuming interpolating FE spaces).
///
/// Let \f$ A \f$ be the "Neumann" operator matrix of \f$ L \f$, i.e., we do not treat the boundaries any differently and just execute
/// the volume integrals.
///
/// 1. Interpolate Dirichlet boundary conditions into a vector \f$ g \f$.
/// 2. Compute \f$ g_A \gets A g \f$.
/// 3. Compute \f$ g_D \gets \mathrm{diag}(A) g \f$
/// 4. Set the rhs to \f$ b_\text{elim} = b - g_A \f$,
///    where \f$ b \f$ is the assembled rhs vector for the homogeneous problem
///    (the result of evaluating the linear form into a vector or of the matrix-vector product of a vector \f$ f_\text{vec} \f$ where
///    the rhs function \f$ f \f$ has been interpolated into, and then \f$ b = M f_\text{vec} \f$ (\f$ M \f$ being the mass matrix))
/// 5. Set the rhs \f$ b_\text{elim} \f$ at the boundary nodes to \f$ g_D \f$, i.e.
///    \f$ b_\text{elim} \gets g_D \f$ on the Dirichlet boundary
/// 6. Solve
///         \f$ A_\text{elim} x = b_\text{elim} \f$
///    where \f$ A_\text{elim} \f$ is \f$ A \f$, but with all off-diagonal entries in the same row/col as a boundary node set to zero.
///    This feature has to be supplied by the operator implementation.
///    In a matrix-free context, we have to adapt the element matrix \f$ A_\text{local} \f$ accordingly by (symmetrically) zeroing
///    out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
///    We still have \f$ \mathrm{diag}(A) = \mathrm{diag}(A_\text{elim}) \f$ .
/// 7. \f$ x \f$ is the solution of the original problem. No boundary correction should be necessary.
///
template < typename ScalarType, linalg::OperatorLike OperatorType, typename FlagType >
void strong_algebraic_dirichlet_enforcement_poisson_like(
    OperatorType&                               A_neumann,
    OperatorType&                               A_neumann_diag,
    const linalg::VectorQ1Scalar< ScalarType >& g,
    linalg::VectorQ1Scalar< ScalarType >&       tmp,
    linalg::VectorQ1Scalar< ScalarType >&       b,
    const grid::Grid4DDataScalar< FlagType >&   mask_data,
    const FlagType&                             dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( A_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( A_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), tmp.grid_data(), mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_dirichlet_enforcement_poisson_like() for homogenous boundary conditions (\f$ g = 0 \f$).
///
/// Does not require most of the steps since \f$ g = g_A = g_D = 0 \f$. Still requires solving \f$ A_\text{elim} x = b_elim \f$after this.
template < typename ScalarType, util::FlagLike FlagType >
void strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
    linalg::VectorQ1Scalar< ScalarType >&     b,
    const grid::Grid4DDataScalar< FlagType >& mask_data,
    const FlagType&                           dirichlet_boundary_mask )
{
    // b_elim <- 0 on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), 0.0, mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_dirichlet_enforcement_poisson_like() for Stokes-like systems (with strong enforcement of velocity boundary conditions).
template < typename ScalarType, linalg::OperatorLike OperatorType, util::FlagLike FlagType >
void strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
    OperatorType&                                K_neumann,
    OperatorType&                                K_neumann_diag,
    const linalg::VectorQ1IsoQ2Q1< ScalarType >& g,
    linalg::VectorQ1IsoQ2Q1< ScalarType >&       tmp,
    linalg::VectorQ1IsoQ2Q1< ScalarType >&       b,
    const grid::Grid4DDataScalar< FlagType >&    mask_data,
    const FlagType&                              dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( K_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( K_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old(
        b.block_1().grid_data(), tmp.block_1().grid_data(), mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like() for Stokes-like systems (with strong enforcement of zero velocity boundary conditions).
template < typename ScalarType, util::FlagLike FlagType >
void strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
    linalg::VectorQ1IsoQ2Q1< ScalarType >&    b,
    const grid::Grid4DDataScalar< FlagType >& mask_data,
    const FlagType&                           dirichlet_boundary_mask )
{
    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old(
        b.block_1().grid_data(), ScalarType( 0 ), mask_data, dirichlet_boundary_mask );
}

} // namespace terra::fe