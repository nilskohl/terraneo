
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::linalg::solvers {

template < typename ScalarT >
class GCAElementsCollector
{
  public:
    using ScalarType = ScalarT;
    using WedgeIndex = std::tuple< int, int, int, int, bool >;

  private:
    grid::shell::DistributedDomain fine_domain_;

    // fine grid coefficient
    grid::Grid4DDataScalar< ScalarType > k_;

    // coarsest grid boolean field for elements on which a GCA hierarchy has to be built
    grid::Grid4DDataScalar< ScalarType >& GCAElements_;

    grid::Grid4DDataScalar< ScalarType >& k_grad_norms_;
    const int                             level_range_;

  public:
    GCAElementsCollector(
        const grid::shell::DistributedDomain&       fine_domain,
        const grid::Grid4DDataScalar< ScalarType >& k,
        grid::Grid4DDataScalar< ScalarType >&       GCAElements,
        grid::Grid4DDataScalar< ScalarType >&       k_grad_norms,
        const int                                   level_range )
    : fine_domain_( fine_domain )
    , k_( k )
    , GCAElements_( GCAElements )
    , k_grad_norms_( k_grad_norms )
    , level_range_( level_range )
    {
        Kokkos::parallel_for(
            "evaluate coefficient gradient", grid::shell::local_domain_md_range_policy_cells( fine_domain_ ), *this );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 6 > k[num_wedges_per_hex_cell];
        terra::fe::wedge::extract_local_wedge_scalar_coefficients( k, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        constexpr auto num_quad_points = fe::wedge::quadrature::quad_felippa_1x1_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        fe::wedge::quadrature::quad_felippa_1x1_quad_points( quad_points );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            const auto qp = quad_points[0];

            dense::Vec< ScalarType, 3 > k_grad_eval = { 0 };
            for ( int j = 0; j < num_nodes_per_wedge; j++ )
            {
                k_grad_eval = k_grad_eval + terra::fe::wedge::grad_shape( j, qp ) * k[wedge]( j );
            }
            auto k_grad_norm = k_grad_eval.norm();
            if ( k_grad_norm > 10 )
            {
                k_grad_norms_( local_subdomain_id, x_cell, y_cell, r_cell ) = k_grad_norm;
                // Todo: map to parent coarsest element
                int x_cell_coarsest = x_cell;
                int y_cell_coarsest = y_cell;
                int r_cell_coarsest = r_cell;
                for ( int l = 0; l < level_range_; ++l )
                {
                    x_cell_coarsest = Kokkos::floor( x_cell_coarsest / 2 );
                    y_cell_coarsest = Kokkos::floor( y_cell_coarsest / 2 );
                    r_cell_coarsest = Kokkos::floor( r_cell_coarsest / 2 );
                }
                GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest, r_cell_coarsest )             = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest + 1, y_cell_coarsest, r_cell_coarsest )         = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest + 1, r_cell_coarsest )         = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest + 1, y_cell_coarsest + 1, r_cell_coarsest )     = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest, r_cell_coarsest + 1 )         = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest + 1, y_cell_coarsest, r_cell_coarsest + 1 )     = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest + 1, r_cell_coarsest + 1 )     = 1;
                GCAElements_( local_subdomain_id, x_cell_coarsest + 1, y_cell_coarsest + 1, r_cell_coarsest + 1 ) = 1;
            }
        }
    }
};
} // namespace terra::linalg::solvers