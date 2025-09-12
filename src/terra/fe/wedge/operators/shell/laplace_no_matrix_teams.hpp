
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class LaplaceNoMatrixTeams
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    using team_policy_t = Kokkos::TeamPolicy<>;
    using member_t      = team_policy_t::member_type;

  public:
    LaplaceNoMatrixTeams(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        // Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        const auto num_cells =
            src_.extent( 0 ) * ( src_.extent( 1 ) - 1 ) * ( src_.extent( 2 ) - 1 ) * ( src_.extent( 3 ) - 1 );
        Kokkos::TeamPolicy<> policy( num_cells, Kokkos::AUTO );
        policy = policy.set_scratch_size( 0, Kokkos::PerTeam( team_scratch_size_in_bytes() ) );
        Kokkos::parallel_for( policy, *this );

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_and_send_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::recv_unpack_and_add_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void map_league_idx_to_indices(
        int  global_cell_idx,
        int  cells_per_subdomain,
        int  num_cells_x,
        int  num_cells_y,
        int  num_cells_r,
        int& local_subdomain_id,
        int& x_cell,
        int& y_cell,
        int& r_cell ) const
    {
        local_subdomain_id = global_cell_idx / cells_per_subdomain;
        int rem            = global_cell_idx % cells_per_subdomain;
        x_cell             = rem / ( num_cells_x * num_cells_y );
        rem                = rem % ( num_cells_x * num_cells_y );
        y_cell             = rem / num_cells_x;
        r_cell             = rem % num_cells_x;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const member_t& team ) const
    {
        const int league_idx = team.league_rank();
        int       local_subdomain_id, x_cell, y_cell, r_cell;
        map_league_idx_to_indices(
            league_idx,
            ( src_.extent( 1 ) - 1 ) * ( src_.extent( 2 ) - 1 ) * ( src_.extent( 3 ) - 1 ),
            src_.extent( 1 ) - 1,
            src_.extent( 2 ) - 1,
            src_.extent( 3 ) - 1,
            local_subdomain_id,
            x_cell,
            y_cell,
            r_cell );

        // calculate scratch sizes:
        // wedge_phy_surf: [num_wedges_per_hex_cell][num_nodes_per_wedge_surface] of Vec<3>
        // dst_local_hex: 8 scalars
        // grad_phy: [num_nodes_per_wedge] of Vec<3>  <-- can be per-thread or reused in scratch
        const int bytes_wedge_surf = sizeof( dense::Vec< ScalarT, 3 > ) * num_nodes_per_wedge_surface;
        const int bytes_dst_local  = sizeof( ScalarT ) * 8;
        // optionally, allocate grad_phy per-thread in registers or small scratch per team
        // Request team scratch size (rounded up)
        char* team_scratch = (char*) team.team_shmem().get_shmem( 2 * bytes_wedge_surf + bytes_dst_local );

        // Pointers into team_scratch:
        auto wedge_phy_surf_0 = reinterpret_cast< dense::Vec< ScalarT, 3 >* >( team_scratch );
        auto wedge_phy_surf_1 = reinterpret_cast< dense::Vec< ScalarT, 3 >* >( team_scratch + bytes_wedge_surf );
        auto dst_local_hex    = reinterpret_cast< ScalarT* >( team_scratch + 2 * bytes_wedge_surf );

        // Init local accumulator to zero (only once per team)
        for ( int i = 0; i < 8; ++i )
            dst_local_hex[i] = ScalarT( 0 );

        // Step 1: single thread loads wedge surface geometry into team scratch
        Kokkos::single( Kokkos::PerTeam( team ), [&]() {
            wedge_0_surface_physical_coords( wedge_phy_surf_0, grid_, local_subdomain_id, x_cell, y_cell );
            wedge_1_surface_physical_coords( wedge_phy_surf_1, grid_, local_subdomain_id, x_cell, y_cell );
        } );

        // Synchronize: ensure surface is visible to other team threads
        team.team_barrier();

        dense::Vec< ScalarT, 3 >* wedge_phy_surf[2] = { wedge_phy_surf_0, wedge_phy_surf_1 };

        // Gather surface points for each wedge.
        // dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        // wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // ScalarType dst_local_hex[8] = { 0 };

        // Gather wedge radii (small, per-team)
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // local arrays for offsets (constexpr)
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // Parallel over wedges and quadrature points using team-level parallelism.
        // We'll parallelize over wedges first (TeamThreadRange), and inside each wedge
        // over quadrature points (ThreadVectorRange) or another TeamThreadRange depending on target.
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, num_wedges_per_hex_cell ), [&]( int wedge ) {
            // Option A: For each wedge, have inner loop over quadrature points sequentially
            // Option B: use ThreadVectorRange to parallelize quadrature (useful on GPUs)
            // We'll use ThreadVectorRange for quad points to use vector lanes.
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, quadrature::quad_felippa_3x2_num_quad_points ), [&]( int q ) {
                    // local copies (prefer registers)
                    const auto quad_point  = quad_points_static[q];
                    const auto quad_weight = quad_weights_static[q];

                    // 1. Jacobian + det + inverse-transposed for this wedge @ quad_point
                    const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
                    const auto det              = J.det();
                    const auto abs_det          = Kokkos::abs( det );
                    const auto J_inv_transposed = J.inv_transposed( det );

                    // 2. Compute physical gradients for all nodes at this quadrature point.
                    dense::Vec< ScalarT, 3 > grad_phy_local[num_nodes_per_wedge];
                    for ( int k = 0; k < num_nodes_per_wedge; ++k )
                    {
                        grad_phy_local[k] = J_inv_transposed * grad_shape( k, quad_point );
                    }

                    // 3. Compute grad_u at this quadrature point.
                    dense::Vec< ScalarT, 3 > grad_u;
                    grad_u.fill( ScalarT( 0 ) );

                    // Accumulate grad_u: unroll small loops where possible to reduce registers
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        const ScalarT val = src_(
                            local_subdomain_id,
                            x_cell + offset_x[wedge][j],
                            y_cell + offset_y[wedge][j],
                            r_cell + offset_r[wedge][j] );
                        // explicit component-wise multiply-add for better FMA generation
                        grad_u( 0 ) += val * grad_phy_local[j]( 0 );
                        grad_u( 1 ) += val * grad_phy_local[j]( 1 );
                        grad_u( 2 ) += val * grad_phy_local[j]( 2 );
                    }

                    // 4. Add contributions to per-team local accumulator (dst_local_hex)
                    // We avoid atomic here because dst_local_hex is in team scratch and unique to team.
                    for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    {
                        const int hx = offset_x[wedge][i];
                        const int hy = offset_y[wedge][i];
                        const int hr = offset_r[wedge][i];
                        // mapping to 0..7 local hex node index: 4*hr + 2*hy + hx
                        const int local_hex_idx = 4 * hr + 2 * hy + hx;

                        // dot product manually for FMAs
                        ScalarT dot = grad_phy_local[i]( 0 ) * grad_u( 0 ) + grad_phy_local[i]( 1 ) * grad_u( 1 ) +
                                      grad_phy_local[i]( 2 ) * grad_u( 2 );

                        // accumulate into team scratch (no atomics)
                        // use atomic on shared memory only if multiple threads write same index concurrently,
                        // but here multiple (wedge,q) combinations can race on same local_hex_idx inside team;
                        // to avoid races we can instead perform a per-thread local accumulator and reduce
                        // For simplicity in this sketch, we perform a team-thread-level atomic on scratch:
                        Kokkos::atomic_add( &dst_local_hex[local_hex_idx], quad_weight * dot * abs_det );
                    } // end for i
                } ); // end ThreadVectorRange over q
        } );         // end TeamThreadRange over wedges

        // Wait for all wedge/quad work to finish
        team.team_barrier();

        // Finally, have a single thread per team write to global dst_.
        // If hex ownership is unique (no other teams will write same global entries), do direct writes.
        // Otherwise, use atomic adds to global dst_. We'll use atomic add here for safety.
        Kokkos::single( Kokkos::PerTeam( team ), [&]() {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            for ( int i = 0; i < 8; ++i )
            {
                const ScalarT val = dst_local_hex[i];
                if ( val != ScalarT( 0 ) )
                {
                    Kokkos::atomic_add(
                        &dst_(
                            local_subdomain_id,
                            x_cell + hex_offset_x[i],
                            y_cell + hex_offset_y[i],
                            r_cell + hex_offset_r[i] ),
                        val );
                }
            }
        } );
    } // operator()

    // Required to tell Kokkos how much team scratch we need (estimate per team)
    static size_t team_scratch_size_in_bytes()
    {
        const size_t bytes_wedge_surf =
            sizeof( dense::Vec< ScalarT, 3 > ) * num_wedges_per_hex_cell * num_nodes_per_wedge_surface;
        const size_t bytes_dst_local = sizeof( ScalarT ) * 8;
        return bytes_wedge_surf + bytes_dst_local;
    }

    // Define static quad arrays (in a .cpp file)
    const std::array< dense::Vec< ScalarT, 3 >, quadrature::quad_felippa_3x2_num_quad_points > quad_points_static =
        []() {
            std::array< dense::Vec< ScalarT, 3 >, quadrature::quad_felippa_3x2_num_quad_points > a;
            quadrature::quad_felippa_3x2_quad_points_ptr( a.data() );
            return a;
        }();

    const std::array< ScalarT, quadrature::quad_felippa_3x2_num_quad_points > quad_weights_static = []() {
        std::array< ScalarT, quadrature::quad_felippa_3x2_num_quad_points > a;
        quadrature::quad_felippa_3x2_quad_weights_ptr( a.data() );
        return a;
    }();

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Compute the local element matrix.

        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        ScalarType dst_local_hex[8] = { 0 };

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto quad_point  = quad_points[q];
                const auto quad_weight = quad_weights[q];

                // 1. Compute Jacobian and inverse at this quadrature point.

                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = J.det();
                const auto abs_det          = Kokkos::abs( det );
                const auto J_inv_transposed = J.inv_transposed( det );

                // 2. Compute physical gradients for all nodes at this quadrature point.
                dense::Vec< ScalarType, 3 > grad_phy[num_nodes_per_wedge];
                for ( int k = 0; k < num_nodes_per_wedge; k++ )
                {
                    grad_phy[k] = J_inv_transposed * grad_shape( k, quad_point );
                }

                // 3. Compute ∇u at this quadrature point.
                dense::Vec< ScalarType, 3 > grad_u;
                grad_u.fill( 0.0 );
                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                {
                    grad_u = grad_u + src_(
                                          local_subdomain_id,
                                          x_cell + offset_x[wedge][j],
                                          y_cell + offset_y[wedge][j],
                                          r_cell + offset_r[wedge][j] ) *
                                          grad_phy[j];
                }

                // 4. Add the test function contributions.
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
#if 0
                    Kokkos::atomic_add(
                        &dst_(
                            local_subdomain_id,
                            x_cell + offset_x[wedge][i],
                            y_cell + offset_y[wedge][i],
                            r_cell + offset_r[wedge][i] ),
                        quad_weight * grad_phy[i].dot( grad_u ) * abs_det );
#endif

                    dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                        quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
                }
            }
        }

        for ( int i = 0; i < 8; i++ )
        {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            Kokkos::atomic_add(
                &dst_(
                    local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] ),
                dst_local_hex[i] );
        }
    }
};

static_assert( linalg::OperatorLike< LaplaceNoMatrix< float > > );
static_assert( linalg::OperatorLike< LaplaceNoMatrix< double > > );

} // namespace terra::fe::wedge::operators::shell