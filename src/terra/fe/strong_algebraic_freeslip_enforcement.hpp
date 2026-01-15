
#pragma once

#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

namespace terra::fe {

/// TODO: Function that
///         1. rotates the passed "Neumann" RHS Stokes vector at the freeslip boundary (in place would be great)
///         2. zeroes all normal components of the velocity part
template < typename ScalarType, typename ScalarTypeGrid, util::FlagLike FlagType >
void strong_algebraic_freeslip_enforcement_in_place(
    linalg::VectorQ1IsoQ2Q1< ScalarType >&          b,
    const grid::Grid3DDataVec< ScalarTypeGrid, 3 >& coords_shell,
    const grid::Grid4DDataScalar< FlagType >&       mask_data,
    const FlagType&                                 freeslip_boundary_mask )
{
    // b <- Rb
    linalg::trafo::cartesian_to_normal_tangential_in_place( b, coords_shell, mask_data, freeslip_boundary_mask );

    // b <- 0 for normal components at FS boundary
    kernels::common::assign_masked_else_keep_old( b.block_1().grid_data(), 0, mask_data, freeslip_boundary_mask, 0 );
}

} // namespace terra::fe