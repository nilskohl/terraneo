
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class LaplaceSimple
{
  public:
    using SrcVectorType           = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType           = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType              = ScalarT;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, 6, 6, 2 >;

    static constexpr int LocalMatrixDim = 6;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application
    Grid4DDataLocalMatrices lmatrices_;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    bool treat_boundary_;
    bool diagonal_;
    bool single_quadpoint_ = true;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    dense::Vec< ScalarT, 3 > quad_points_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    ScalarT                  quad_weights_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    dense::Vec< ScalarT, 3 > quad_points_1x1_[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights_1x1_[quadrature::quad_felippa_1x1_num_quad_points];

  public:
    LaplaceSimple(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        bool                                     treat_boundary,
        bool                                     diagonal,
        bool                                     single_quadpoint    = false,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , single_quadpoint_( single_quadpoint )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        quadrature::quad_felippa_1x1_quad_points( quad_points_1x1_ );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights_1x1_ );
        quadrature::quad_felippa_3x2_quad_points( quad_points_3x2_ );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights_3x2_ );
    }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() const { return grid_; }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief S/Getter for quadpoint member
    void set_single_quadpoint( bool v ) { single_quadpoint_ = v; }

    /// @brief Retrives the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, 6, 6 > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        assert( lmatrices_.data() != nullptr );

        return lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
    }

    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                    local_subdomain_id,
        const int                                                    x_cell,
        const int                                                    y_cell,
        const int                                                    r_cell,
        const int                                                    wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        Kokkos::abort( "Not implemented." );
    }

    /// @brief Setter/Getter for app applyStoredLMatrices_: usage of stored local matrices during apply
    void setApplyStoredLMatrices( bool v ) { applyStoredLMatrices_ = v; }

    /// @brief
    /// allocates memory for the local matrices
    /// calls kernel with storeLMatrices_ = true to assemble and store the local matrices
    /// sets applyStoredLMatrices_, such that future applies use the stored local matrices
    void store_lmatrices()
    {
        storeLMatrices_ = true;
        if ( lmatrices_.data() == nullptr )
        {
            lmatrices_ = Grid4DDataLocalMatrices(
                "LaplaceSimple::lmatrices_",
                domain_.subdomains().size(),
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_radially() - 1 );
            Kokkos::parallel_for(
                "assemble_store_lmatrices", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
            Kokkos::fence();
        }
        storeLMatrices_       = false;
        applyStoredLMatrices_ = true;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( lmatrices_.data() != nullptr );

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

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        /*
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};
        if ( !applyStoredLMatrices_ )
        {
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // Quadrature points.
            int num_quad_points = single_quadpoint_ ? quadrature::quad_felippa_1x1_num_quad_points :
                                                      quadrature::quad_felippa_3x2_num_quad_points;

            // Compute the local element matrix.

            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto w  = single_quadpoint_ ? quad_weights_1x1_[q] : quad_weights_3x2_[q];
                const auto qp = single_quadpoint_ ? quad_points_1x1_[q] : quad_points_3x2_[q];

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det              = Kokkos::abs( J.det() );
                    const auto J_inv_transposed = J.inv().transposed();

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        const auto grad_i = grad_shape( i, qp );

                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const auto grad_j = grad_shape( j, qp );

                            A[wedge]( i, j ) +=
                                w * ( ( J_inv_transposed * grad_i ).dot( J_inv_transposed * grad_j ) * det );
                        }
                    }
                }
            }

            if ( treat_boundary_ )
            {
                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    dense::Mat< ScalarT, 6, 6 > boundary_mask;
                    boundary_mask.fill( 1.0 );
                    if ( r_cell == 0 )
                    {
                        // Inner boundary (CMB).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i != j && ( i < 3 || j < 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }

                    if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                    {
                        // Outer boundary (surface).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i != j && ( i >= 3 || j >= 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }

                    A[wedge].hadamard_product( boundary_mask );
                }
            }
        }
        else
        {
            // load LMatrix for both local wedges
            A[0] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        if ( storeLMatrices_ )
        {
            // write local matrices to mem
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) = A[0];
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) = A[1];
        }
        else
        {
            // apply local matrices to local DoFs
            dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

            dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        }*/

        double quad_surface_coords_0_0_0   = grid_( local_subdomain_id, x_cell + 0, y_cell + 0, 0 );
        double quad_surface_coords_0_0_1   = grid_( local_subdomain_id, x_cell + 0, y_cell + 0, 1 );
        double quad_surface_coords_0_0_2   = grid_( local_subdomain_id, x_cell + 0, y_cell + 0, 2 );
        double quad_surface_coords_0_1_0   = grid_( local_subdomain_id, x_cell + 0, y_cell + 1, 0 );
        double quad_surface_coords_0_1_1   = grid_( local_subdomain_id, x_cell + 0, y_cell + 1, 1 );
        double quad_surface_coords_0_1_2   = grid_( local_subdomain_id, x_cell + 0, y_cell + 1, 2 );
        double quad_surface_coords_1_0_0   = grid_( local_subdomain_id, x_cell + 1, y_cell + 0, 0 );
        double quad_surface_coords_1_0_1   = grid_( local_subdomain_id, x_cell + 1, y_cell + 0, 1 );
        double quad_surface_coords_1_0_2   = grid_( local_subdomain_id, x_cell + 1, y_cell + 0, 2 );
        double quad_surface_coords_1_1_0   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 0 );
        double quad_surface_coords_1_1_1   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 1 );
        double quad_surface_coords_1_1_2   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 2 );
        double wedge_surf_phy_coords_0_0_0 = quad_surface_coords_0_0_0;
        double wedge_surf_phy_coords_0_0_1 = quad_surface_coords_0_0_1;
        double wedge_surf_phy_coords_0_0_2 = quad_surface_coords_0_0_2;
        double wedge_surf_phy_coords_0_1_0 = quad_surface_coords_1_0_0;
        double wedge_surf_phy_coords_0_1_1 = quad_surface_coords_1_0_1;
        double wedge_surf_phy_coords_0_1_2 = quad_surface_coords_1_0_2;
        double wedge_surf_phy_coords_0_2_0 = quad_surface_coords_0_1_0;
        double wedge_surf_phy_coords_0_2_1 = quad_surface_coords_0_1_1;
        double wedge_surf_phy_coords_0_2_2 = quad_surface_coords_0_1_2;
        double wedge_surf_phy_coords_1_0_0 = quad_surface_coords_1_1_0;
        double wedge_surf_phy_coords_1_0_1 = quad_surface_coords_1_1_1;
        double wedge_surf_phy_coords_1_0_2 = quad_surface_coords_1_1_2;
        double wedge_surf_phy_coords_1_1_0 = quad_surface_coords_0_1_0;
        double wedge_surf_phy_coords_1_1_1 = quad_surface_coords_0_1_1;
        double wedge_surf_phy_coords_1_1_2 = quad_surface_coords_0_1_2;
        double wedge_surf_phy_coords_1_2_0 = quad_surface_coords_1_0_0;
        double wedge_surf_phy_coords_1_2_1 = quad_surface_coords_1_0_1;
        double wedge_surf_phy_coords_1_2_2 = quad_surface_coords_1_0_2;
        double r_0                         = radii_( local_subdomain_id, r_cell + 0 );
        double r_1                         = radii_( local_subdomain_id, r_cell + 1 );
        double src_0_0                     = src_( local_subdomain_id, x_cell, y_cell, r_cell );
        double src_0_1                     = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        double src_0_2                     = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        double src_0_3                     = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
        double src_0_4                     = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
        double src_0_5                     = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
        double src_1_0                     = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
        double src_1_1                     = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        double src_1_2                     = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        double src_1_3                     = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
        double src_1_4                     = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
        double src_1_5                     = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
        double w0_tmpcse_J_0               = 0.5 * r_0 + 0.5 * r_1;
        double w0_tmpcse_J_1               = -1.0 / 2.0 * r_0 + ( 1.0 / 2.0 ) * r_1;
        double w0_J_0_0           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_0 + wedge_surf_phy_coords_0_1_0 );
        double w0_J_0_1           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_0 + wedge_surf_phy_coords_0_2_0 );
        double w0_J_0_2           = w0_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_0_0_0 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_1_0 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_2_0 );
        double w0_J_1_0           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_1 + wedge_surf_phy_coords_0_1_1 );
        double w0_J_1_1           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_1 + wedge_surf_phy_coords_0_2_1 );
        double w0_J_1_2           = w0_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_0_0_1 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_1_1 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_2_1 );
        double w0_J_2_0           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_2 + wedge_surf_phy_coords_0_1_2 );
        double w0_J_2_1           = w0_tmpcse_J_0 * ( -wedge_surf_phy_coords_0_0_2 + wedge_surf_phy_coords_0_2_2 );
        double w0_J_2_2           = w0_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_0_0_2 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_1_2 +
                                            0.33333333333333331 * wedge_surf_phy_coords_0_2_2 );
        double w0_tmpcse_J_invT_0 = w0_J_1_1 * w0_J_2_2;
        double w0_tmpcse_J_invT_1 = w0_J_1_2 * w0_J_2_1;
        double w0_tmpcse_J_invT_2 = w0_J_1_0 * w0_J_2_1;
        double w0_tmpcse_J_invT_3 = w0_J_1_0 * w0_J_2_2;
        double w0_tmpcse_J_invT_4 = w0_J_1_1 * w0_J_2_0;
        double w0_tmpcse_J_invT_5 =
            1.0 / ( w0_J_0_0 * w0_tmpcse_J_invT_0 - w0_J_0_0 * w0_tmpcse_J_invT_1 + w0_J_0_1 * w0_J_1_2 * w0_J_2_0 -
                    w0_J_0_1 * w0_tmpcse_J_invT_3 + w0_J_0_2 * w0_tmpcse_J_invT_2 - w0_J_0_2 * w0_tmpcse_J_invT_4 );
        double w0_J_invT_cse_0_0 = w0_tmpcse_J_invT_5 * ( w0_tmpcse_J_invT_0 - w0_tmpcse_J_invT_1 );
        double w0_J_invT_cse_0_1 = w0_tmpcse_J_invT_5 * ( w0_J_1_2 * w0_J_2_0 - w0_tmpcse_J_invT_3 );
        double w0_J_invT_cse_0_2 = w0_tmpcse_J_invT_5 * ( w0_tmpcse_J_invT_2 - w0_tmpcse_J_invT_4 );
        double w0_J_invT_cse_1_0 = w0_tmpcse_J_invT_5 * ( -w0_J_0_1 * w0_J_2_2 + w0_J_0_2 * w0_J_2_1 );
        double w0_J_invT_cse_1_1 = w0_tmpcse_J_invT_5 * ( w0_J_0_0 * w0_J_2_2 - w0_J_0_2 * w0_J_2_0 );
        double w0_J_invT_cse_1_2 = w0_tmpcse_J_invT_5 * ( -w0_J_0_0 * w0_J_2_1 + w0_J_0_1 * w0_J_2_0 );
        double w0_J_invT_cse_2_0 = w0_tmpcse_J_invT_5 * ( w0_J_0_1 * w0_J_1_2 - w0_J_0_2 * w0_J_1_1 );
        double w0_J_invT_cse_2_1 = w0_tmpcse_J_invT_5 * ( -w0_J_0_0 * w0_J_1_2 + w0_J_0_2 * w0_J_1_0 );
        double w0_J_invT_cse_2_2 = w0_tmpcse_J_invT_5 * ( w0_J_0_0 * w0_J_1_1 - w0_J_0_1 * w0_J_1_0 );
        double w0_absdet         = fabs(
            w0_J_0_0 * w0_J_1_1 * w0_J_2_2 - w0_J_0_0 * w0_J_1_2 * w0_J_2_1 - w0_J_0_1 * w0_J_1_0 * w0_J_2_2 +
            w0_J_0_1 * w0_J_1_2 * w0_J_2_0 + w0_J_0_2 * w0_J_1_0 * w0_J_2_1 - w0_J_0_2 * w0_J_1_1 * w0_J_2_0 );
        double w0_tmpcse_local_mat_0  = 0.33333333333333343 * w0_J_invT_cse_0_2;
        double w0_tmpcse_local_mat_1  = w0_J_invT_cse_0_0 + w0_J_invT_cse_0_1;
        double w0_tmpcse_local_mat_2  = 0.33333333333333343 * w0_J_invT_cse_1_2;
        double w0_tmpcse_local_mat_3  = w0_J_invT_cse_1_0 + w0_J_invT_cse_1_1;
        double w0_tmpcse_local_mat_4  = 0.33333333333333343 * w0_J_invT_cse_2_2;
        double w0_tmpcse_local_mat_5  = w0_J_invT_cse_2_0 + w0_J_invT_cse_2_1;
        double w0_tmpcse_local_mat_6  = 0.5 * w0_J_invT_cse_0_0;
        double w0_tmpcse_local_mat_7  = 0.16666666666666666 * w0_J_invT_cse_0_2;
        double w0_tmpcse_local_mat_8  = -w0_tmpcse_local_mat_7;
        double w0_tmpcse_local_mat_9  = w0_tmpcse_local_mat_6 + w0_tmpcse_local_mat_8;
        double w0_tmpcse_local_mat_10 = 0.16666666666666671 * w0_J_invT_cse_0_2;
        double w0_tmpcse_local_mat_11 = 0.5 * w0_J_invT_cse_0_1;
        double w0_tmpcse_local_mat_12 = w0_tmpcse_local_mat_11 + w0_tmpcse_local_mat_6;
        double w0_tmpcse_local_mat_13 = 1.0 * w0_absdet;
        double w0_tmpcse_local_mat_14 = w0_tmpcse_local_mat_13 * ( -w0_tmpcse_local_mat_10 - w0_tmpcse_local_mat_12 );
        double w0_tmpcse_local_mat_15 = 0.5 * w0_J_invT_cse_1_0;
        double w0_tmpcse_local_mat_16 = 0.16666666666666666 * w0_J_invT_cse_1_2;
        double w0_tmpcse_local_mat_17 = -w0_tmpcse_local_mat_16;
        double w0_tmpcse_local_mat_18 = w0_tmpcse_local_mat_15 + w0_tmpcse_local_mat_17;
        double w0_tmpcse_local_mat_19 = 0.16666666666666671 * w0_J_invT_cse_1_2;
        double w0_tmpcse_local_mat_20 = 0.5 * w0_J_invT_cse_1_1;
        double w0_tmpcse_local_mat_21 = w0_tmpcse_local_mat_15 + w0_tmpcse_local_mat_20;
        double w0_tmpcse_local_mat_22 = w0_tmpcse_local_mat_13 * ( -w0_tmpcse_local_mat_19 - w0_tmpcse_local_mat_21 );
        double w0_tmpcse_local_mat_23 = 0.5 * w0_J_invT_cse_2_0;
        double w0_tmpcse_local_mat_24 = 0.16666666666666666 * w0_J_invT_cse_2_2;
        double w0_tmpcse_local_mat_25 = -w0_tmpcse_local_mat_24;
        double w0_tmpcse_local_mat_26 = w0_tmpcse_local_mat_23 + w0_tmpcse_local_mat_25;
        double w0_tmpcse_local_mat_27 = 0.16666666666666671 * w0_J_invT_cse_2_2;
        double w0_tmpcse_local_mat_28 = 0.5 * w0_J_invT_cse_2_1;
        double w0_tmpcse_local_mat_29 = w0_tmpcse_local_mat_23 + w0_tmpcse_local_mat_28;
        double w0_tmpcse_local_mat_30 = w0_tmpcse_local_mat_13 * ( -w0_tmpcse_local_mat_27 - w0_tmpcse_local_mat_29 );
        double w0_tmpcse_local_mat_31 = w0_tmpcse_local_mat_14 * w0_tmpcse_local_mat_9 +
                                        w0_tmpcse_local_mat_18 * w0_tmpcse_local_mat_22 +
                                        w0_tmpcse_local_mat_26 * w0_tmpcse_local_mat_30;
        double w0_tmpcse_local_mat_32 = w0_tmpcse_local_mat_11 + w0_tmpcse_local_mat_8;
        double w0_tmpcse_local_mat_33 = w0_tmpcse_local_mat_17 + w0_tmpcse_local_mat_20;
        double w0_tmpcse_local_mat_34 = w0_tmpcse_local_mat_25 + w0_tmpcse_local_mat_28;
        double w0_tmpcse_local_mat_35 = w0_tmpcse_local_mat_14 * w0_tmpcse_local_mat_32 +
                                        w0_tmpcse_local_mat_22 * w0_tmpcse_local_mat_33 +
                                        w0_tmpcse_local_mat_30 * w0_tmpcse_local_mat_34;
        double w0_tmpcse_local_mat_36 = w0_tmpcse_local_mat_10 - w0_tmpcse_local_mat_12;
        double w0_tmpcse_local_mat_37 = w0_tmpcse_local_mat_19 - w0_tmpcse_local_mat_21;
        double w0_tmpcse_local_mat_38 = w0_tmpcse_local_mat_27 - w0_tmpcse_local_mat_29;
        double w0_tmpcse_local_mat_39 = w0_tmpcse_local_mat_14 * w0_tmpcse_local_mat_36 +
                                        w0_tmpcse_local_mat_22 * w0_tmpcse_local_mat_37 +
                                        w0_tmpcse_local_mat_30 * w0_tmpcse_local_mat_38;
        double w0_tmpcse_local_mat_40 = w0_tmpcse_local_mat_6 + w0_tmpcse_local_mat_7;
        double w0_tmpcse_local_mat_41 = w0_tmpcse_local_mat_15 + w0_tmpcse_local_mat_16;
        double w0_tmpcse_local_mat_42 = w0_tmpcse_local_mat_23 + w0_tmpcse_local_mat_24;
        double w0_tmpcse_local_mat_43 = w0_tmpcse_local_mat_14 * w0_tmpcse_local_mat_40 +
                                        w0_tmpcse_local_mat_22 * w0_tmpcse_local_mat_41 +
                                        w0_tmpcse_local_mat_30 * w0_tmpcse_local_mat_42;
        double w0_tmpcse_local_mat_44 = w0_tmpcse_local_mat_11 + w0_tmpcse_local_mat_7;
        double w0_tmpcse_local_mat_45 = w0_tmpcse_local_mat_16 + w0_tmpcse_local_mat_20;
        double w0_tmpcse_local_mat_46 = w0_tmpcse_local_mat_24 + w0_tmpcse_local_mat_28;
        double w0_tmpcse_local_mat_47 = w0_tmpcse_local_mat_14 * w0_tmpcse_local_mat_44 +
                                        w0_tmpcse_local_mat_22 * w0_tmpcse_local_mat_45 +
                                        w0_tmpcse_local_mat_30 * w0_tmpcse_local_mat_46;
        double w0_tmpcse_local_mat_48 = 0.33333333333333331 * w0_J_invT_cse_0_2;
        double w0_tmpcse_local_mat_49 = -w0_tmpcse_local_mat_48;
        double w0_tmpcse_local_mat_50 = 0.33333333333333331 * w0_J_invT_cse_1_2;
        double w0_tmpcse_local_mat_51 = -w0_tmpcse_local_mat_50;
        double w0_tmpcse_local_mat_52 = 0.33333333333333331 * w0_J_invT_cse_2_2;
        double w0_tmpcse_local_mat_53 = -w0_tmpcse_local_mat_52;
        double w0_tmpcse_local_mat_54 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_9;
        double w0_tmpcse_local_mat_55 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_18;
        double w0_tmpcse_local_mat_56 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_26;
        double w0_tmpcse_local_mat_57 = w0_tmpcse_local_mat_32 * w0_tmpcse_local_mat_54 +
                                        w0_tmpcse_local_mat_33 * w0_tmpcse_local_mat_55 +
                                        w0_tmpcse_local_mat_34 * w0_tmpcse_local_mat_56;
        double w0_tmpcse_local_mat_58 = w0_tmpcse_local_mat_36 * w0_tmpcse_local_mat_54 +
                                        w0_tmpcse_local_mat_37 * w0_tmpcse_local_mat_55 +
                                        w0_tmpcse_local_mat_38 * w0_tmpcse_local_mat_56;
        double w0_tmpcse_local_mat_59 = w0_tmpcse_local_mat_40 * w0_tmpcse_local_mat_54 +
                                        w0_tmpcse_local_mat_41 * w0_tmpcse_local_mat_55 +
                                        w0_tmpcse_local_mat_42 * w0_tmpcse_local_mat_56;
        double w0_tmpcse_local_mat_60 = w0_tmpcse_local_mat_44 * w0_tmpcse_local_mat_54 +
                                        w0_tmpcse_local_mat_45 * w0_tmpcse_local_mat_55 +
                                        w0_tmpcse_local_mat_46 * w0_tmpcse_local_mat_56;
        double w0_tmpcse_local_mat_61 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_32;
        double w0_tmpcse_local_mat_62 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_33;
        double w0_tmpcse_local_mat_63 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_34;
        double w0_tmpcse_local_mat_64 = w0_tmpcse_local_mat_36 * w0_tmpcse_local_mat_61 +
                                        w0_tmpcse_local_mat_37 * w0_tmpcse_local_mat_62 +
                                        w0_tmpcse_local_mat_38 * w0_tmpcse_local_mat_63;
        double w0_tmpcse_local_mat_65 = w0_tmpcse_local_mat_40 * w0_tmpcse_local_mat_61 +
                                        w0_tmpcse_local_mat_41 * w0_tmpcse_local_mat_62 +
                                        w0_tmpcse_local_mat_42 * w0_tmpcse_local_mat_63;
        double w0_tmpcse_local_mat_66 = w0_tmpcse_local_mat_44 * w0_tmpcse_local_mat_61 +
                                        w0_tmpcse_local_mat_45 * w0_tmpcse_local_mat_62 +
                                        w0_tmpcse_local_mat_46 * w0_tmpcse_local_mat_63;
        double w0_tmpcse_local_mat_67 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_36;
        double w0_tmpcse_local_mat_68 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_37;
        double w0_tmpcse_local_mat_69 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_38;
        double w0_tmpcse_local_mat_70 = w0_tmpcse_local_mat_40 * w0_tmpcse_local_mat_67 +
                                        w0_tmpcse_local_mat_41 * w0_tmpcse_local_mat_68 +
                                        w0_tmpcse_local_mat_42 * w0_tmpcse_local_mat_69;
        double w0_tmpcse_local_mat_71 = w0_tmpcse_local_mat_44 * w0_tmpcse_local_mat_67 +
                                        w0_tmpcse_local_mat_45 * w0_tmpcse_local_mat_68 +
                                        w0_tmpcse_local_mat_46 * w0_tmpcse_local_mat_69;
        double w0_tmpcse_local_mat_72 = w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_40 * w0_tmpcse_local_mat_44 +
                                        w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_41 * w0_tmpcse_local_mat_45 +
                                        w0_tmpcse_local_mat_13 * w0_tmpcse_local_mat_42 * w0_tmpcse_local_mat_46;
        double w0_local_mat_replaced_0_0 = 0.25 * w0_absdet * pow( -w0_tmpcse_local_mat_0 - w0_tmpcse_local_mat_1, 2 ) +
                                           0.25 * w0_absdet * pow( -w0_tmpcse_local_mat_2 - w0_tmpcse_local_mat_3, 2 ) +
                                           0.25 * w0_absdet * pow( -w0_tmpcse_local_mat_4 - w0_tmpcse_local_mat_5, 2 );
        double w0_local_mat_replaced_0_1 = w0_tmpcse_local_mat_31;
        double w0_local_mat_replaced_0_2 = w0_tmpcse_local_mat_35;
        double w0_local_mat_replaced_0_3 = w0_tmpcse_local_mat_39;
        double w0_local_mat_replaced_0_4 = w0_tmpcse_local_mat_43;
        double w0_local_mat_replaced_0_5 = w0_tmpcse_local_mat_47;
        double w0_local_mat_replaced_1_0 = w0_tmpcse_local_mat_31;
        double w0_local_mat_replaced_1_1 = 0.25 * w0_absdet * pow( w0_J_invT_cse_0_0 + w0_tmpcse_local_mat_49, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_1_0 + w0_tmpcse_local_mat_51, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_2_0 + w0_tmpcse_local_mat_53, 2 );
        double w0_local_mat_replaced_1_2 = w0_tmpcse_local_mat_57;
        double w0_local_mat_replaced_1_3 = w0_tmpcse_local_mat_58;
        double w0_local_mat_replaced_1_4 = w0_tmpcse_local_mat_59;
        double w0_local_mat_replaced_1_5 = w0_tmpcse_local_mat_60;
        double w0_local_mat_replaced_2_0 = w0_tmpcse_local_mat_35;
        double w0_local_mat_replaced_2_1 = w0_tmpcse_local_mat_57;
        double w0_local_mat_replaced_2_2 = 0.25 * w0_absdet * pow( w0_J_invT_cse_0_1 + w0_tmpcse_local_mat_49, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_1_1 + w0_tmpcse_local_mat_51, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_2_1 + w0_tmpcse_local_mat_53, 2 );
        double w0_local_mat_replaced_2_3 = w0_tmpcse_local_mat_64;
        double w0_local_mat_replaced_2_4 = w0_tmpcse_local_mat_65;
        double w0_local_mat_replaced_2_5 = w0_tmpcse_local_mat_66;
        double w0_local_mat_replaced_3_0 = w0_tmpcse_local_mat_39;
        double w0_local_mat_replaced_3_1 = w0_tmpcse_local_mat_58;
        double w0_local_mat_replaced_3_2 = w0_tmpcse_local_mat_64;
        double w0_local_mat_replaced_3_3 = 0.25 * w0_absdet * pow( w0_tmpcse_local_mat_0 - w0_tmpcse_local_mat_1, 2 ) +
                                           0.25 * w0_absdet * pow( w0_tmpcse_local_mat_2 - w0_tmpcse_local_mat_3, 2 ) +
                                           0.25 * w0_absdet * pow( w0_tmpcse_local_mat_4 - w0_tmpcse_local_mat_5, 2 );
        double w0_local_mat_replaced_3_4 = w0_tmpcse_local_mat_70;
        double w0_local_mat_replaced_3_5 = w0_tmpcse_local_mat_71;
        double w0_local_mat_replaced_4_0 = w0_tmpcse_local_mat_43;
        double w0_local_mat_replaced_4_1 = w0_tmpcse_local_mat_59;
        double w0_local_mat_replaced_4_2 = w0_tmpcse_local_mat_65;
        double w0_local_mat_replaced_4_3 = w0_tmpcse_local_mat_70;
        double w0_local_mat_replaced_4_4 = 0.25 * w0_absdet * pow( w0_J_invT_cse_0_0 + w0_tmpcse_local_mat_48, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_1_0 + w0_tmpcse_local_mat_50, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_2_0 + w0_tmpcse_local_mat_52, 2 );
        double w0_local_mat_replaced_4_5 = w0_tmpcse_local_mat_72;
        double w0_local_mat_replaced_5_0 = w0_tmpcse_local_mat_47;
        double w0_local_mat_replaced_5_1 = w0_tmpcse_local_mat_60;
        double w0_local_mat_replaced_5_2 = w0_tmpcse_local_mat_66;
        double w0_local_mat_replaced_5_3 = w0_tmpcse_local_mat_71;
        double w0_local_mat_replaced_5_4 = w0_tmpcse_local_mat_72;
        double w0_local_mat_replaced_5_5 = 0.25 * w0_absdet * pow( w0_J_invT_cse_0_1 + w0_tmpcse_local_mat_48, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_1_1 + w0_tmpcse_local_mat_50, 2 ) +
                                           0.25 * w0_absdet * pow( w0_J_invT_cse_2_1 + w0_tmpcse_local_mat_52, 2 );

        if ( treat_boundary_ )
        {
            if ( r_cell == 0 )
            {
                // Inner boundary (CMB).
                w0_local_mat_replaced_0_1 = 0.0;
                w0_local_mat_replaced_0_2 = 0.0;
                w0_local_mat_replaced_0_3 = 0.0;
                w0_local_mat_replaced_0_4 = 0.0;
                w0_local_mat_replaced_0_5 = 0.0;
                w0_local_mat_replaced_1_0 = 0.0;
                w0_local_mat_replaced_1_2 = 0.0;
                w0_local_mat_replaced_1_3 = 0.0;
                w0_local_mat_replaced_1_4 = 0.0;
                w0_local_mat_replaced_1_5 = 0.0;
                w0_local_mat_replaced_2_0 = 0.0;
                w0_local_mat_replaced_2_1 = 0.0;
                w0_local_mat_replaced_2_3 = 0.0;
                w0_local_mat_replaced_2_4 = 0.0;
                w0_local_mat_replaced_2_5 = 0.0;
                w0_local_mat_replaced_3_0 = 0.0;
                w0_local_mat_replaced_3_1 = 0.0;
                w0_local_mat_replaced_3_2 = 0.0;
                w0_local_mat_replaced_4_0 = 0.0;
                w0_local_mat_replaced_4_1 = 0.0;
                w0_local_mat_replaced_4_2 = 0.0;
                w0_local_mat_replaced_5_0 = 0.0;
                w0_local_mat_replaced_5_1 = 0.0;
                w0_local_mat_replaced_5_2 = 0.0;
            }

            if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
            {
                w0_local_mat_replaced_0_3 = 0.0;
                w0_local_mat_replaced_0_4 = 0.0;
                w0_local_mat_replaced_0_5 = 0.0;
                w0_local_mat_replaced_1_3 = 0.0;
                w0_local_mat_replaced_1_4 = 0.0;
                w0_local_mat_replaced_1_5 = 0.0;
                w0_local_mat_replaced_2_3 = 0.0;
                w0_local_mat_replaced_2_4 = 0.0;
                w0_local_mat_replaced_2_5 = 0.0;
                w0_local_mat_replaced_3_0 = 0.0;
                w0_local_mat_replaced_3_1 = 0.0;
                w0_local_mat_replaced_3_2 = 0.0;
                w0_local_mat_replaced_3_4 = 0.0;
                w0_local_mat_replaced_3_5 = 0.0;
                w0_local_mat_replaced_4_0 = 0.0;
                w0_local_mat_replaced_4_1 = 0.0;
                w0_local_mat_replaced_4_2 = 0.0;
                w0_local_mat_replaced_4_3 = 0.0;
                w0_local_mat_replaced_4_5 = 0.0;
                w0_local_mat_replaced_5_0 = 0.0;
                w0_local_mat_replaced_5_1 = 0.0;
                w0_local_mat_replaced_5_2 = 0.0;
                w0_local_mat_replaced_5_3 = 0.0;
                w0_local_mat_replaced_5_4 = 0.0;
            }
        }

        if ( diagonal_ )
        {
            {
                w0_local_mat_replaced_0_1 = 0.0;
                w0_local_mat_replaced_0_2 = 0.0;
                w0_local_mat_replaced_0_3 = 0.0;
                w0_local_mat_replaced_0_4 = 0.0;
                w0_local_mat_replaced_0_5 = 0.0;
                w0_local_mat_replaced_1_0 = 0.0;
                w0_local_mat_replaced_1_2 = 0.0;
                w0_local_mat_replaced_1_3 = 0.0;
                w0_local_mat_replaced_1_4 = 0.0;
                w0_local_mat_replaced_1_5 = 0.0;
                w0_local_mat_replaced_2_0 = 0.0;
                w0_local_mat_replaced_2_1 = 0.0;
                w0_local_mat_replaced_2_3 = 0.0;
                w0_local_mat_replaced_2_4 = 0.0;
                w0_local_mat_replaced_2_5 = 0.0;
                w0_local_mat_replaced_3_0 = 0.0;
                w0_local_mat_replaced_3_1 = 0.0;
                w0_local_mat_replaced_3_2 = 0.0;
                w0_local_mat_replaced_3_4 = 0.0;
                w0_local_mat_replaced_3_5 = 0.0;
                w0_local_mat_replaced_4_0 = 0.0;
                w0_local_mat_replaced_4_1 = 0.0;
                w0_local_mat_replaced_4_2 = 0.0;
                w0_local_mat_replaced_4_3 = 0.0;
                w0_local_mat_replaced_4_5 = 0.0;
                w0_local_mat_replaced_5_0 = 0.0;
                w0_local_mat_replaced_5_1 = 0.0;
                w0_local_mat_replaced_5_2 = 0.0;
                w0_local_mat_replaced_5_3 = 0.0;
                w0_local_mat_replaced_5_4 = 0.0;
            }
        }
        double dst_0_0 = src_0_0 * w0_local_mat_replaced_0_0 + src_0_1 * w0_local_mat_replaced_0_1 +
                         src_0_2 * w0_local_mat_replaced_0_2 + src_0_3 * w0_local_mat_replaced_0_3 +
                         src_0_4 * w0_local_mat_replaced_0_4 + src_0_5 * w0_local_mat_replaced_0_5;
        double dst_0_1 = src_0_0 * w0_local_mat_replaced_1_0 + src_0_1 * w0_local_mat_replaced_1_1 +
                         src_0_2 * w0_local_mat_replaced_1_2 + src_0_3 * w0_local_mat_replaced_1_3 +
                         src_0_4 * w0_local_mat_replaced_1_4 + src_0_5 * w0_local_mat_replaced_1_5;
        double dst_0_2 = src_0_0 * w0_local_mat_replaced_2_0 + src_0_1 * w0_local_mat_replaced_2_1 +
                         src_0_2 * w0_local_mat_replaced_2_2 + src_0_3 * w0_local_mat_replaced_2_3 +
                         src_0_4 * w0_local_mat_replaced_2_4 + src_0_5 * w0_local_mat_replaced_2_5;
        double dst_0_3 = src_0_0 * w0_local_mat_replaced_3_0 + src_0_1 * w0_local_mat_replaced_3_1 +
                         src_0_2 * w0_local_mat_replaced_3_2 + src_0_3 * w0_local_mat_replaced_3_3 +
                         src_0_4 * w0_local_mat_replaced_3_4 + src_0_5 * w0_local_mat_replaced_3_5;
        double dst_0_4 = src_0_0 * w0_local_mat_replaced_4_0 + src_0_1 * w0_local_mat_replaced_4_1 +
                         src_0_2 * w0_local_mat_replaced_4_2 + src_0_3 * w0_local_mat_replaced_4_3 +
                         src_0_4 * w0_local_mat_replaced_4_4 + src_0_5 * w0_local_mat_replaced_4_5;
        double dst_0_5 = src_0_0 * w0_local_mat_replaced_5_0 + src_0_1 * w0_local_mat_replaced_5_1 +
                         src_0_2 * w0_local_mat_replaced_5_2 + src_0_3 * w0_local_mat_replaced_5_3 +
                         src_0_4 * w0_local_mat_replaced_5_4 + src_0_5 * w0_local_mat_replaced_5_5;
        double w1_tmpcse_J_0      = 0.5 * r_0 + 0.5 * r_1;
        double w1_tmpcse_J_1      = -1.0 / 2.0 * r_0 + ( 1.0 / 2.0 ) * r_1;
        double w1_J_0_0           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_0 + wedge_surf_phy_coords_1_1_0 );
        double w1_J_0_1           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_0 + wedge_surf_phy_coords_1_2_0 );
        double w1_J_0_2           = w1_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_1_0_0 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_1_0 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_2_0 );
        double w1_J_1_0           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_1 + wedge_surf_phy_coords_1_1_1 );
        double w1_J_1_1           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_1 + wedge_surf_phy_coords_1_2_1 );
        double w1_J_1_2           = w1_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_1_0_1 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_1_1 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_2_1 );
        double w1_J_2_0           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_2 + wedge_surf_phy_coords_1_1_2 );
        double w1_J_2_1           = w1_tmpcse_J_0 * ( -wedge_surf_phy_coords_1_0_2 + wedge_surf_phy_coords_1_2_2 );
        double w1_J_2_2           = w1_tmpcse_J_1 * ( 0.33333333333333343 * wedge_surf_phy_coords_1_0_2 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_1_2 +
                                            0.33333333333333331 * wedge_surf_phy_coords_1_2_2 );
        double w1_tmpcse_J_invT_0 = w1_J_1_1 * w1_J_2_2;
        double w1_tmpcse_J_invT_1 = w1_J_1_2 * w1_J_2_1;
        double w1_tmpcse_J_invT_2 = w1_J_1_0 * w1_J_2_1;
        double w1_tmpcse_J_invT_3 = w1_J_1_0 * w1_J_2_2;
        double w1_tmpcse_J_invT_4 = w1_J_1_1 * w1_J_2_0;
        double w1_tmpcse_J_invT_5 =
            1.0 / ( w1_J_0_0 * w1_tmpcse_J_invT_0 - w1_J_0_0 * w1_tmpcse_J_invT_1 + w1_J_0_1 * w1_J_1_2 * w1_J_2_0 -
                    w1_J_0_1 * w1_tmpcse_J_invT_3 + w1_J_0_2 * w1_tmpcse_J_invT_2 - w1_J_0_2 * w1_tmpcse_J_invT_4 );
        double w1_J_invT_cse_0_0 = w1_tmpcse_J_invT_5 * ( w1_tmpcse_J_invT_0 - w1_tmpcse_J_invT_1 );
        double w1_J_invT_cse_0_1 = w1_tmpcse_J_invT_5 * ( w1_J_1_2 * w1_J_2_0 - w1_tmpcse_J_invT_3 );
        double w1_J_invT_cse_0_2 = w1_tmpcse_J_invT_5 * ( w1_tmpcse_J_invT_2 - w1_tmpcse_J_invT_4 );
        double w1_J_invT_cse_1_0 = w1_tmpcse_J_invT_5 * ( -w1_J_0_1 * w1_J_2_2 + w1_J_0_2 * w1_J_2_1 );
        double w1_J_invT_cse_1_1 = w1_tmpcse_J_invT_5 * ( w1_J_0_0 * w1_J_2_2 - w1_J_0_2 * w1_J_2_0 );
        double w1_J_invT_cse_1_2 = w1_tmpcse_J_invT_5 * ( -w1_J_0_0 * w1_J_2_1 + w1_J_0_1 * w1_J_2_0 );
        double w1_J_invT_cse_2_0 = w1_tmpcse_J_invT_5 * ( w1_J_0_1 * w1_J_1_2 - w1_J_0_2 * w1_J_1_1 );
        double w1_J_invT_cse_2_1 = w1_tmpcse_J_invT_5 * ( -w1_J_0_0 * w1_J_1_2 + w1_J_0_2 * w1_J_1_0 );
        double w1_J_invT_cse_2_2 = w1_tmpcse_J_invT_5 * ( w1_J_0_0 * w1_J_1_1 - w1_J_0_1 * w1_J_1_0 );
        double w1_absdet         = fabs(
            w1_J_0_0 * w1_J_1_1 * w1_J_2_2 - w1_J_0_0 * w1_J_1_2 * w1_J_2_1 - w1_J_0_1 * w1_J_1_0 * w1_J_2_2 +
            w1_J_0_1 * w1_J_1_2 * w1_J_2_0 + w1_J_0_2 * w1_J_1_0 * w1_J_2_1 - w1_J_0_2 * w1_J_1_1 * w1_J_2_0 );
        double w1_tmpcse_local_mat_0  = 0.33333333333333343 * w1_J_invT_cse_0_2;
        double w1_tmpcse_local_mat_1  = w1_J_invT_cse_0_0 + w1_J_invT_cse_0_1;
        double w1_tmpcse_local_mat_2  = 0.33333333333333343 * w1_J_invT_cse_1_2;
        double w1_tmpcse_local_mat_3  = w1_J_invT_cse_1_0 + w1_J_invT_cse_1_1;
        double w1_tmpcse_local_mat_4  = 0.33333333333333343 * w1_J_invT_cse_2_2;
        double w1_tmpcse_local_mat_5  = w1_J_invT_cse_2_0 + w1_J_invT_cse_2_1;
        double w1_tmpcse_local_mat_6  = 0.5 * w1_J_invT_cse_0_0;
        double w1_tmpcse_local_mat_7  = 0.16666666666666666 * w1_J_invT_cse_0_2;
        double w1_tmpcse_local_mat_8  = -w1_tmpcse_local_mat_7;
        double w1_tmpcse_local_mat_9  = w1_tmpcse_local_mat_6 + w1_tmpcse_local_mat_8;
        double w1_tmpcse_local_mat_10 = 0.16666666666666671 * w1_J_invT_cse_0_2;
        double w1_tmpcse_local_mat_11 = 0.5 * w1_J_invT_cse_0_1;
        double w1_tmpcse_local_mat_12 = w1_tmpcse_local_mat_11 + w1_tmpcse_local_mat_6;
        double w1_tmpcse_local_mat_13 = 1.0 * w1_absdet;
        double w1_tmpcse_local_mat_14 = w1_tmpcse_local_mat_13 * ( -w1_tmpcse_local_mat_10 - w1_tmpcse_local_mat_12 );
        double w1_tmpcse_local_mat_15 = 0.5 * w1_J_invT_cse_1_0;
        double w1_tmpcse_local_mat_16 = 0.16666666666666666 * w1_J_invT_cse_1_2;
        double w1_tmpcse_local_mat_17 = -w1_tmpcse_local_mat_16;
        double w1_tmpcse_local_mat_18 = w1_tmpcse_local_mat_15 + w1_tmpcse_local_mat_17;
        double w1_tmpcse_local_mat_19 = 0.16666666666666671 * w1_J_invT_cse_1_2;
        double w1_tmpcse_local_mat_20 = 0.5 * w1_J_invT_cse_1_1;
        double w1_tmpcse_local_mat_21 = w1_tmpcse_local_mat_15 + w1_tmpcse_local_mat_20;
        double w1_tmpcse_local_mat_22 = w1_tmpcse_local_mat_13 * ( -w1_tmpcse_local_mat_19 - w1_tmpcse_local_mat_21 );
        double w1_tmpcse_local_mat_23 = 0.5 * w1_J_invT_cse_2_0;
        double w1_tmpcse_local_mat_24 = 0.16666666666666666 * w1_J_invT_cse_2_2;
        double w1_tmpcse_local_mat_25 = -w1_tmpcse_local_mat_24;
        double w1_tmpcse_local_mat_26 = w1_tmpcse_local_mat_23 + w1_tmpcse_local_mat_25;
        double w1_tmpcse_local_mat_27 = 0.16666666666666671 * w1_J_invT_cse_2_2;
        double w1_tmpcse_local_mat_28 = 0.5 * w1_J_invT_cse_2_1;
        double w1_tmpcse_local_mat_29 = w1_tmpcse_local_mat_23 + w1_tmpcse_local_mat_28;
        double w1_tmpcse_local_mat_30 = w1_tmpcse_local_mat_13 * ( -w1_tmpcse_local_mat_27 - w1_tmpcse_local_mat_29 );
        double w1_tmpcse_local_mat_31 = w1_tmpcse_local_mat_14 * w1_tmpcse_local_mat_9 +
                                        w1_tmpcse_local_mat_18 * w1_tmpcse_local_mat_22 +
                                        w1_tmpcse_local_mat_26 * w1_tmpcse_local_mat_30;
        double w1_tmpcse_local_mat_32 = w1_tmpcse_local_mat_11 + w1_tmpcse_local_mat_8;
        double w1_tmpcse_local_mat_33 = w1_tmpcse_local_mat_17 + w1_tmpcse_local_mat_20;
        double w1_tmpcse_local_mat_34 = w1_tmpcse_local_mat_25 + w1_tmpcse_local_mat_28;
        double w1_tmpcse_local_mat_35 = w1_tmpcse_local_mat_14 * w1_tmpcse_local_mat_32 +
                                        w1_tmpcse_local_mat_22 * w1_tmpcse_local_mat_33 +
                                        w1_tmpcse_local_mat_30 * w1_tmpcse_local_mat_34;
        double w1_tmpcse_local_mat_36 = w1_tmpcse_local_mat_10 - w1_tmpcse_local_mat_12;
        double w1_tmpcse_local_mat_37 = w1_tmpcse_local_mat_19 - w1_tmpcse_local_mat_21;
        double w1_tmpcse_local_mat_38 = w1_tmpcse_local_mat_27 - w1_tmpcse_local_mat_29;
        double w1_tmpcse_local_mat_39 = w1_tmpcse_local_mat_14 * w1_tmpcse_local_mat_36 +
                                        w1_tmpcse_local_mat_22 * w1_tmpcse_local_mat_37 +
                                        w1_tmpcse_local_mat_30 * w1_tmpcse_local_mat_38;
        double w1_tmpcse_local_mat_40 = w1_tmpcse_local_mat_6 + w1_tmpcse_local_mat_7;
        double w1_tmpcse_local_mat_41 = w1_tmpcse_local_mat_15 + w1_tmpcse_local_mat_16;
        double w1_tmpcse_local_mat_42 = w1_tmpcse_local_mat_23 + w1_tmpcse_local_mat_24;
        double w1_tmpcse_local_mat_43 = w1_tmpcse_local_mat_14 * w1_tmpcse_local_mat_40 +
                                        w1_tmpcse_local_mat_22 * w1_tmpcse_local_mat_41 +
                                        w1_tmpcse_local_mat_30 * w1_tmpcse_local_mat_42;
        double w1_tmpcse_local_mat_44 = w1_tmpcse_local_mat_11 + w1_tmpcse_local_mat_7;
        double w1_tmpcse_local_mat_45 = w1_tmpcse_local_mat_16 + w1_tmpcse_local_mat_20;
        double w1_tmpcse_local_mat_46 = w1_tmpcse_local_mat_24 + w1_tmpcse_local_mat_28;
        double w1_tmpcse_local_mat_47 = w1_tmpcse_local_mat_14 * w1_tmpcse_local_mat_44 +
                                        w1_tmpcse_local_mat_22 * w1_tmpcse_local_mat_45 +
                                        w1_tmpcse_local_mat_30 * w1_tmpcse_local_mat_46;
        double w1_tmpcse_local_mat_48 = 0.33333333333333331 * w1_J_invT_cse_0_2;
        double w1_tmpcse_local_mat_49 = -w1_tmpcse_local_mat_48;
        double w1_tmpcse_local_mat_50 = 0.33333333333333331 * w1_J_invT_cse_1_2;
        double w1_tmpcse_local_mat_51 = -w1_tmpcse_local_mat_50;
        double w1_tmpcse_local_mat_52 = 0.33333333333333331 * w1_J_invT_cse_2_2;
        double w1_tmpcse_local_mat_53 = -w1_tmpcse_local_mat_52;
        double w1_tmpcse_local_mat_54 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_9;
        double w1_tmpcse_local_mat_55 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_18;
        double w1_tmpcse_local_mat_56 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_26;
        double w1_tmpcse_local_mat_57 = w1_tmpcse_local_mat_32 * w1_tmpcse_local_mat_54 +
                                        w1_tmpcse_local_mat_33 * w1_tmpcse_local_mat_55 +
                                        w1_tmpcse_local_mat_34 * w1_tmpcse_local_mat_56;
        double w1_tmpcse_local_mat_58 = w1_tmpcse_local_mat_36 * w1_tmpcse_local_mat_54 +
                                        w1_tmpcse_local_mat_37 * w1_tmpcse_local_mat_55 +
                                        w1_tmpcse_local_mat_38 * w1_tmpcse_local_mat_56;
        double w1_tmpcse_local_mat_59 = w1_tmpcse_local_mat_40 * w1_tmpcse_local_mat_54 +
                                        w1_tmpcse_local_mat_41 * w1_tmpcse_local_mat_55 +
                                        w1_tmpcse_local_mat_42 * w1_tmpcse_local_mat_56;
        double w1_tmpcse_local_mat_60 = w1_tmpcse_local_mat_44 * w1_tmpcse_local_mat_54 +
                                        w1_tmpcse_local_mat_45 * w1_tmpcse_local_mat_55 +
                                        w1_tmpcse_local_mat_46 * w1_tmpcse_local_mat_56;
        double w1_tmpcse_local_mat_61 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_32;
        double w1_tmpcse_local_mat_62 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_33;
        double w1_tmpcse_local_mat_63 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_34;
        double w1_tmpcse_local_mat_64 = w1_tmpcse_local_mat_36 * w1_tmpcse_local_mat_61 +
                                        w1_tmpcse_local_mat_37 * w1_tmpcse_local_mat_62 +
                                        w1_tmpcse_local_mat_38 * w1_tmpcse_local_mat_63;
        double w1_tmpcse_local_mat_65 = w1_tmpcse_local_mat_40 * w1_tmpcse_local_mat_61 +
                                        w1_tmpcse_local_mat_41 * w1_tmpcse_local_mat_62 +
                                        w1_tmpcse_local_mat_42 * w1_tmpcse_local_mat_63;
        double w1_tmpcse_local_mat_66 = w1_tmpcse_local_mat_44 * w1_tmpcse_local_mat_61 +
                                        w1_tmpcse_local_mat_45 * w1_tmpcse_local_mat_62 +
                                        w1_tmpcse_local_mat_46 * w1_tmpcse_local_mat_63;
        double w1_tmpcse_local_mat_67 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_36;
        double w1_tmpcse_local_mat_68 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_37;
        double w1_tmpcse_local_mat_69 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_38;
        double w1_tmpcse_local_mat_70 = w1_tmpcse_local_mat_40 * w1_tmpcse_local_mat_67 +
                                        w1_tmpcse_local_mat_41 * w1_tmpcse_local_mat_68 +
                                        w1_tmpcse_local_mat_42 * w1_tmpcse_local_mat_69;
        double w1_tmpcse_local_mat_71 = w1_tmpcse_local_mat_44 * w1_tmpcse_local_mat_67 +
                                        w1_tmpcse_local_mat_45 * w1_tmpcse_local_mat_68 +
                                        w1_tmpcse_local_mat_46 * w1_tmpcse_local_mat_69;
        double w1_tmpcse_local_mat_72 = w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_40 * w1_tmpcse_local_mat_44 +
                                        w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_41 * w1_tmpcse_local_mat_45 +
                                        w1_tmpcse_local_mat_13 * w1_tmpcse_local_mat_42 * w1_tmpcse_local_mat_46;
        double w1_local_mat_replaced_0_0 = 0.25 * w1_absdet * pow( -w1_tmpcse_local_mat_0 - w1_tmpcse_local_mat_1, 2 ) +
                                           0.25 * w1_absdet * pow( -w1_tmpcse_local_mat_2 - w1_tmpcse_local_mat_3, 2 ) +
                                           0.25 * w1_absdet * pow( -w1_tmpcse_local_mat_4 - w1_tmpcse_local_mat_5, 2 );
        double w1_local_mat_replaced_0_1 = w1_tmpcse_local_mat_31;
        double w1_local_mat_replaced_0_2 = w1_tmpcse_local_mat_35;
        double w1_local_mat_replaced_0_3 = w1_tmpcse_local_mat_39;
        double w1_local_mat_replaced_0_4 = w1_tmpcse_local_mat_43;
        double w1_local_mat_replaced_0_5 = w1_tmpcse_local_mat_47;
        double w1_local_mat_replaced_1_0 = w1_tmpcse_local_mat_31;
        double w1_local_mat_replaced_1_1 = 0.25 * w1_absdet * pow( w1_J_invT_cse_0_0 + w1_tmpcse_local_mat_49, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_1_0 + w1_tmpcse_local_mat_51, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_2_0 + w1_tmpcse_local_mat_53, 2 );
        double w1_local_mat_replaced_1_2 = w1_tmpcse_local_mat_57;
        double w1_local_mat_replaced_1_3 = w1_tmpcse_local_mat_58;
        double w1_local_mat_replaced_1_4 = w1_tmpcse_local_mat_59;
        double w1_local_mat_replaced_1_5 = w1_tmpcse_local_mat_60;
        double w1_local_mat_replaced_2_0 = w1_tmpcse_local_mat_35;
        double w1_local_mat_replaced_2_1 = w1_tmpcse_local_mat_57;
        double w1_local_mat_replaced_2_2 = 0.25 * w1_absdet * pow( w1_J_invT_cse_0_1 + w1_tmpcse_local_mat_49, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_1_1 + w1_tmpcse_local_mat_51, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_2_1 + w1_tmpcse_local_mat_53, 2 );
        double w1_local_mat_replaced_2_3 = w1_tmpcse_local_mat_64;
        double w1_local_mat_replaced_2_4 = w1_tmpcse_local_mat_65;
        double w1_local_mat_replaced_2_5 = w1_tmpcse_local_mat_66;
        double w1_local_mat_replaced_3_0 = w1_tmpcse_local_mat_39;
        double w1_local_mat_replaced_3_1 = w1_tmpcse_local_mat_58;
        double w1_local_mat_replaced_3_2 = w1_tmpcse_local_mat_64;
        double w1_local_mat_replaced_3_3 = 0.25 * w1_absdet * pow( w1_tmpcse_local_mat_0 - w1_tmpcse_local_mat_1, 2 ) +
                                           0.25 * w1_absdet * pow( w1_tmpcse_local_mat_2 - w1_tmpcse_local_mat_3, 2 ) +
                                           0.25 * w1_absdet * pow( w1_tmpcse_local_mat_4 - w1_tmpcse_local_mat_5, 2 );
        double w1_local_mat_replaced_3_4 = w1_tmpcse_local_mat_70;
        double w1_local_mat_replaced_3_5 = w1_tmpcse_local_mat_71;
        double w1_local_mat_replaced_4_0 = w1_tmpcse_local_mat_43;
        double w1_local_mat_replaced_4_1 = w1_tmpcse_local_mat_59;
        double w1_local_mat_replaced_4_2 = w1_tmpcse_local_mat_65;
        double w1_local_mat_replaced_4_3 = w1_tmpcse_local_mat_70;
        double w1_local_mat_replaced_4_4 = 0.25 * w1_absdet * pow( w1_J_invT_cse_0_0 + w1_tmpcse_local_mat_48, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_1_0 + w1_tmpcse_local_mat_50, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_2_0 + w1_tmpcse_local_mat_52, 2 );
        double w1_local_mat_replaced_4_5 = w1_tmpcse_local_mat_72;
        double w1_local_mat_replaced_5_0 = w1_tmpcse_local_mat_47;
        double w1_local_mat_replaced_5_1 = w1_tmpcse_local_mat_60;
        double w1_local_mat_replaced_5_2 = w1_tmpcse_local_mat_66;
        double w1_local_mat_replaced_5_3 = w1_tmpcse_local_mat_71;
        double w1_local_mat_replaced_5_4 = w1_tmpcse_local_mat_72;
        double w1_local_mat_replaced_5_5 = 0.25 * w1_absdet * pow( w1_J_invT_cse_0_1 + w1_tmpcse_local_mat_48, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_1_1 + w1_tmpcse_local_mat_50, 2 ) +
                                           0.25 * w1_absdet * pow( w1_J_invT_cse_2_1 + w1_tmpcse_local_mat_52, 2 );

        if ( treat_boundary_ )
        {
            if ( r_cell == 0 )
            {
                // Inner boundary (CMB).
                w1_local_mat_replaced_0_1 = 0.0;
                w1_local_mat_replaced_0_2 = 0.0;
                w1_local_mat_replaced_0_3 = 0.0;
                w1_local_mat_replaced_0_4 = 0.0;
                w1_local_mat_replaced_0_5 = 0.0;
                w1_local_mat_replaced_1_0 = 0.0;
                w1_local_mat_replaced_1_2 = 0.0;
                w1_local_mat_replaced_1_3 = 0.0;
                w1_local_mat_replaced_1_4 = 0.0;
                w1_local_mat_replaced_1_5 = 0.0;
                w1_local_mat_replaced_2_0 = 0.0;
                w1_local_mat_replaced_2_1 = 0.0;
                w1_local_mat_replaced_2_3 = 0.0;
                w1_local_mat_replaced_2_4 = 0.0;
                w1_local_mat_replaced_2_5 = 0.0;
                w1_local_mat_replaced_3_0 = 0.0;
                w1_local_mat_replaced_3_1 = 0.0;
                w1_local_mat_replaced_3_2 = 0.0;
                w1_local_mat_replaced_4_0 = 0.0;
                w1_local_mat_replaced_4_1 = 0.0;
                w1_local_mat_replaced_4_2 = 0.0;
                w1_local_mat_replaced_5_0 = 0.0;
                w1_local_mat_replaced_5_1 = 0.0;
                w1_local_mat_replaced_5_2 = 0.0;
            }

            if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
            {
                w1_local_mat_replaced_0_3 = 0.0;
                w1_local_mat_replaced_0_4 = 0.0;
                w1_local_mat_replaced_0_5 = 0.0;
                w1_local_mat_replaced_1_3 = 0.0;
                w1_local_mat_replaced_1_4 = 0.0;
                w1_local_mat_replaced_1_5 = 0.0;
                w1_local_mat_replaced_2_3 = 0.0;
                w1_local_mat_replaced_2_4 = 0.0;
                w1_local_mat_replaced_2_5 = 0.0;
                w1_local_mat_replaced_3_0 = 0.0;
                w1_local_mat_replaced_3_1 = 0.0;
                w1_local_mat_replaced_3_2 = 0.0;
                w1_local_mat_replaced_3_4 = 0.0;
                w1_local_mat_replaced_3_5 = 0.0;
                w1_local_mat_replaced_4_0 = 0.0;
                w1_local_mat_replaced_4_1 = 0.0;
                w1_local_mat_replaced_4_2 = 0.0;
                w1_local_mat_replaced_4_3 = 0.0;
                w1_local_mat_replaced_4_5 = 0.0;
                w1_local_mat_replaced_5_0 = 0.0;
                w1_local_mat_replaced_5_1 = 0.0;
                w1_local_mat_replaced_5_2 = 0.0;
                w1_local_mat_replaced_5_3 = 0.0;
                w1_local_mat_replaced_5_4 = 0.0;
            }
        }

        if ( diagonal_ )
        {
            {
                w1_local_mat_replaced_0_1 = 0.0;
                w1_local_mat_replaced_0_2 = 0.0;
                w1_local_mat_replaced_0_3 = 0.0;
                w1_local_mat_replaced_0_4 = 0.0;
                w1_local_mat_replaced_0_5 = 0.0;
                w1_local_mat_replaced_1_0 = 0.0;
                w1_local_mat_replaced_1_2 = 0.0;
                w1_local_mat_replaced_1_3 = 0.0;
                w1_local_mat_replaced_1_4 = 0.0;
                w1_local_mat_replaced_1_5 = 0.0;
                w1_local_mat_replaced_2_0 = 0.0;
                w1_local_mat_replaced_2_1 = 0.0;
                w1_local_mat_replaced_2_3 = 0.0;
                w1_local_mat_replaced_2_4 = 0.0;
                w1_local_mat_replaced_2_5 = 0.0;
                w1_local_mat_replaced_3_0 = 0.0;
                w1_local_mat_replaced_3_1 = 0.0;
                w1_local_mat_replaced_3_2 = 0.0;
                w1_local_mat_replaced_3_4 = 0.0;
                w1_local_mat_replaced_3_5 = 0.0;
                w1_local_mat_replaced_4_0 = 0.0;
                w1_local_mat_replaced_4_1 = 0.0;
                w1_local_mat_replaced_4_2 = 0.0;
                w1_local_mat_replaced_4_3 = 0.0;
                w1_local_mat_replaced_4_5 = 0.0;
                w1_local_mat_replaced_5_0 = 0.0;
                w1_local_mat_replaced_5_1 = 0.0;
                w1_local_mat_replaced_5_2 = 0.0;
                w1_local_mat_replaced_5_3 = 0.0;
                w1_local_mat_replaced_5_4 = 0.0;
            }
        }
        double dst_1_0 = src_1_0 * w1_local_mat_replaced_0_0 + src_1_1 * w1_local_mat_replaced_0_1 +
                         src_1_2 * w1_local_mat_replaced_0_2 + src_1_3 * w1_local_mat_replaced_0_3 +
                         src_1_4 * w1_local_mat_replaced_0_4 + src_1_5 * w1_local_mat_replaced_0_5;
        double dst_1_1 = src_1_0 * w1_local_mat_replaced_1_0 + src_1_1 * w1_local_mat_replaced_1_1 +
                         src_1_2 * w1_local_mat_replaced_1_2 + src_1_3 * w1_local_mat_replaced_1_3 +
                         src_1_4 * w1_local_mat_replaced_1_4 + src_1_5 * w1_local_mat_replaced_1_5;
        double dst_1_2 = src_1_0 * w1_local_mat_replaced_2_0 + src_1_1 * w1_local_mat_replaced_2_1 +
                         src_1_2 * w1_local_mat_replaced_2_2 + src_1_3 * w1_local_mat_replaced_2_3 +
                         src_1_4 * w1_local_mat_replaced_2_4 + src_1_5 * w1_local_mat_replaced_2_5;
        double dst_1_3 = src_1_0 * w1_local_mat_replaced_3_0 + src_1_1 * w1_local_mat_replaced_3_1 +
                         src_1_2 * w1_local_mat_replaced_3_2 + src_1_3 * w1_local_mat_replaced_3_3 +
                         src_1_4 * w1_local_mat_replaced_3_4 + src_1_5 * w1_local_mat_replaced_3_5;
        double dst_1_4 = src_1_0 * w1_local_mat_replaced_4_0 + src_1_1 * w1_local_mat_replaced_4_1 +
                         src_1_2 * w1_local_mat_replaced_4_2 + src_1_3 * w1_local_mat_replaced_4_3 +
                         src_1_4 * w1_local_mat_replaced_4_4 + src_1_5 * w1_local_mat_replaced_4_5;
        double dst_1_5 = src_1_0 * w1_local_mat_replaced_5_0 + src_1_1 * w1_local_mat_replaced_5_1 +
                         src_1_2 * w1_local_mat_replaced_5_2 + src_1_3 * w1_local_mat_replaced_5_3 +
                         src_1_4 * w1_local_mat_replaced_5_4 + src_1_5 * w1_local_mat_replaced_5_5;
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell ), dst_0_0 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst_0_1 + dst_1_2 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst_0_2 + dst_1_1 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), dst_0_3 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst_0_4 + dst_1_5 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst_0_5 + dst_1_4 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), dst_1_0 );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), dst_1_3 );
    }

    // Kernel body:
};

static_assert( linalg::OperatorLike< LaplaceSimple< float > > );
static_assert( linalg::OperatorLike< LaplaceSimple< double > > );
static_assert( linalg::GCACapable< LaplaceSimple< float > > );

} // namespace terra::fe::wedge::operators::shell
