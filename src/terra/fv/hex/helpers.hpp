
#pragma once
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex {

/// @brief Computes cell centers and writes to a vector valued finite volume function.
///
///
/// This function computes for every finite volume cell \f$K\f$ the value
/// \f[
///     u_K = \frac{1}{|K|} \int_K x \ dx
/// \f]
/// with
/// \f[
///     |K| = \int_K 1 \ dx
/// \f]
/// and writes \f$u_K\f$ into the respective finite volume cell.
///
///
/// @param dst [out] finite volume function that is being written to
/// @param coords_shell coords of the unit shell
/// @param coords_radii radii of all shells
template < typename ScalarType, typename GridScalarType >
void compute_cell_centers(
    linalg::VectorFVVec< ScalarType, 3 >&           dst,
    const grid::Grid3DDataVec< GridScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >& coords_radii )
{
    grid::Grid4DDataVec< ScalarType, 3 > fv_grid = dst.grid_data();
    Kokkos::parallel_for(
        "l2_project_into_fv_function",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            ScalarType                  volume   = 0.0;
            dense::Vec< ScalarType, 3 > integral = {};

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, hex_cell_x - 1, hex_cell_y - 1 );

            const auto r_1 = coords_radii( local_subdomain_id, hex_cell_r - 1 );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_quad_points; ++q )
                {
                    const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                    const auto det     = J.det();
                    const auto abs_det = Kokkos::abs( det );
                    const auto Jw      = abs_det * quad_weights[q];
                    volume             = volume + Jw;

                    const auto x = fe::wedge::forward_map(
                        wedge_phy_surf[wedge][0],
                        wedge_phy_surf[wedge][1],
                        wedge_phy_surf[wedge][2],
                        r_1,
                        r_2,
                        quad_points[q]( 0 ),
                        quad_points[q]( 1 ),
                        quad_points[q]( 2 ) );

                    integral = integral + x * Jw;
                }
            }

            for ( int d = 0; d < 3; ++d )
            {
                fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r, d ) = integral( d ) / volume;
            }
        } );
}

} // namespace terra::fv::hex