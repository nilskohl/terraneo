
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
class LaplaceKerngen
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
    LaplaceKerngen(
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

double quad_surface_coords_0_0_0 = grid_(local_subdomain_id,x_cell + 0,y_cell + 0,0);
double quad_surface_coords_0_0_1 = grid_(local_subdomain_id,x_cell + 0,y_cell + 0,1);
double quad_surface_coords_0_0_2 = grid_(local_subdomain_id,x_cell + 0,y_cell + 0,2);
double quad_surface_coords_0_1_0 = grid_(local_subdomain_id,x_cell + 0,y_cell + 1,0);
double quad_surface_coords_0_1_1 = grid_(local_subdomain_id,x_cell + 0,y_cell + 1,1);
double quad_surface_coords_0_1_2 = grid_(local_subdomain_id,x_cell + 0,y_cell + 1,2);
double quad_surface_coords_1_0_0 = grid_(local_subdomain_id,x_cell + 1,y_cell + 0,0);
double quad_surface_coords_1_0_1 = grid_(local_subdomain_id,x_cell + 1,y_cell + 0,1);
double quad_surface_coords_1_0_2 = grid_(local_subdomain_id,x_cell + 1,y_cell + 0,2);
double quad_surface_coords_1_1_0 = grid_(local_subdomain_id,x_cell + 1,y_cell + 1,0);
double quad_surface_coords_1_1_1 = grid_(local_subdomain_id,x_cell + 1,y_cell + 1,1);
double quad_surface_coords_1_1_2 = grid_(local_subdomain_id,x_cell + 1,y_cell + 1,2);
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
double r_0 = radii_(local_subdomain_id, r_cell + 0);
double r_1 = radii_(local_subdomain_id, r_cell + 1);
double src_0_0 = src_(local_subdomain_id,x_cell,y_cell,r_cell);
double src_0_1 = src_(local_subdomain_id,x_cell + 1,y_cell,r_cell);
double src_0_2 = src_(local_subdomain_id,x_cell,y_cell + 1,r_cell);
double src_0_3 = src_(local_subdomain_id,x_cell,y_cell,r_cell + 1);
double src_0_4 = src_(local_subdomain_id,x_cell + 1,y_cell,r_cell + 1);
double src_0_5 = src_(local_subdomain_id,x_cell,y_cell + 1,r_cell + 1);
double src_1_0 = src_(local_subdomain_id,x_cell + 1,y_cell + 1,r_cell);
double src_1_1 = src_(local_subdomain_id,x_cell,y_cell + 1,r_cell);
double src_1_2 = src_(local_subdomain_id,x_cell + 1,y_cell,r_cell);
double src_1_3 = src_(local_subdomain_id,x_cell + 1,y_cell + 1,r_cell + 1);
double src_1_4 = src_(local_subdomain_id,x_cell,y_cell + 1,r_cell + 1);
double src_1_5 = src_(local_subdomain_id,x_cell + 1,y_cell,r_cell + 1);
int cmb_shift = ((treat_boundary_ && diagonal_ == false && r_cell == 0) ? (
   3
)
: (
   0
));
int max_rad = radii_.extent( 1 ) - 1;
int surface_shift = ((treat_boundary_ && diagonal_ == false && max_rad == r_cell + 1) ? (
   3
)
: (
   0
));
int trial_it0_cond = ((diagonal_ == false && cmb_shift <= 0 && surface_shift < 6) ? (
   1
)
: (
   0
));
int test_it0_cond = ((diagonal_ == false && cmb_shift <= 0 && surface_shift < 6) ? (
   1
)
: (
   0
));
int diag_bc_it0_cond = ((surface_shift <= 0 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 6) ? (
   1
)
: (
   0
));
int trial_it1_cond = ((diagonal_ == false && cmb_shift <= 1 && surface_shift < 5) ? (
   1
)
: (
   0
));
int test_it1_cond = ((diagonal_ == false && cmb_shift <= 1 && surface_shift < 5) ? (
   1
)
: (
   0
));
int diag_bc_it1_cond = ((surface_shift <= 1 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 5) ? (
   1
)
: (
   0
));
int trial_it2_cond = ((diagonal_ == false && cmb_shift <= 2 && surface_shift < 4) ? (
   1
)
: (
   0
));
int test_it2_cond = ((diagonal_ == false && cmb_shift <= 2 && surface_shift < 4) ? (
   1
)
: (
   0
));
int diag_bc_it2_cond = ((surface_shift <= 2 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 4) ? (
   1
)
: (
   0
));
int trial_it3_cond = ((diagonal_ == false && cmb_shift <= 3 && surface_shift < 3) ? (
   1
)
: (
   0
));
int test_it3_cond = ((diagonal_ == false && cmb_shift <= 3 && surface_shift < 3) ? (
   1
)
: (
   0
));
int diag_bc_it3_cond = ((surface_shift <= 3 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 3) ? (
   1
)
: (
   0
));
int trial_it4_cond = ((diagonal_ == false && cmb_shift <= 4 && surface_shift < 2) ? (
   1
)
: (
   0
));
int test_it4_cond = ((diagonal_ == false && cmb_shift <= 4 && surface_shift < 2) ? (
   1
)
: (
   0
));
int diag_bc_it4_cond = ((surface_shift <= 4 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 2) ? (
   1
)
: (
   0
));
int trial_it5_cond = ((diagonal_ == false && cmb_shift <= 5 && surface_shift < 1) ? (
   1
)
: (
   0
));
int test_it5_cond = ((diagonal_ == false && cmb_shift <= 5 && surface_shift < 1) ? (
   1
)
: (
   0
));
int diag_bc_it5_cond = ((surface_shift <= 5 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 1) ? (
   1
)
: (
   0
));
double w0_tmpcse_J_0 = 0.5*r_0 + 0.5*r_1;
double w0_tmpcse_J_1 = -1.0/2.0*r_0 + (1.0/2.0)*r_1;
double w0_J_0_0 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_0 + wedge_surf_phy_coords_0_1_0);
double w0_J_0_1 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_0 + wedge_surf_phy_coords_0_2_0);
double w0_J_0_2 = w0_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_0_0_0 + 0.33333333333333331*wedge_surf_phy_coords_0_1_0 + 0.33333333333333331*wedge_surf_phy_coords_0_2_0);
double w0_J_1_0 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_1 + wedge_surf_phy_coords_0_1_1);
double w0_J_1_1 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_1 + wedge_surf_phy_coords_0_2_1);
double w0_J_1_2 = w0_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_0_0_1 + 0.33333333333333331*wedge_surf_phy_coords_0_1_1 + 0.33333333333333331*wedge_surf_phy_coords_0_2_1);
double w0_J_2_0 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_2 + wedge_surf_phy_coords_0_1_2);
double w0_J_2_1 = w0_tmpcse_J_0*(-wedge_surf_phy_coords_0_0_2 + wedge_surf_phy_coords_0_2_2);
double w0_J_2_2 = w0_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_0_0_2 + 0.33333333333333331*wedge_surf_phy_coords_0_1_2 + 0.33333333333333331*wedge_surf_phy_coords_0_2_2);
double w0_J_det = w0_J_0_0*w0_J_1_1*w0_J_2_2 - w0_J_0_0*w0_J_1_2*w0_J_2_1 - w0_J_0_1*w0_J_1_0*w0_J_2_2 + w0_J_0_1*w0_J_1_2*w0_J_2_0 + w0_J_0_2*w0_J_1_0*w0_J_2_1 - w0_J_0_2*w0_J_1_1*w0_J_2_0;
double w0_tmpcse_J_invT_0 = 1.0/w0_J_det;
double w0_J_invT_cse_0_0 = w0_tmpcse_J_invT_0*(w0_J_1_1*w0_J_2_2 - w0_J_1_2*w0_J_2_1);
double w0_J_invT_cse_0_1 = w0_tmpcse_J_invT_0*(-w0_J_1_0*w0_J_2_2 + w0_J_1_2*w0_J_2_0);
double w0_J_invT_cse_0_2 = w0_tmpcse_J_invT_0*(w0_J_1_0*w0_J_2_1 - w0_J_1_1*w0_J_2_0);
double w0_J_invT_cse_1_0 = w0_tmpcse_J_invT_0*(-w0_J_0_1*w0_J_2_2 + w0_J_0_2*w0_J_2_1);
double w0_J_invT_cse_1_1 = w0_tmpcse_J_invT_0*(w0_J_0_0*w0_J_2_2 - w0_J_0_2*w0_J_2_0);
double w0_J_invT_cse_1_2 = w0_tmpcse_J_invT_0*(-w0_J_0_0*w0_J_2_1 + w0_J_0_1*w0_J_2_0);
double w0_J_invT_cse_2_0 = w0_tmpcse_J_invT_0*(w0_J_0_1*w0_J_1_2 - w0_J_0_2*w0_J_1_1);
double w0_J_invT_cse_2_1 = w0_tmpcse_J_invT_0*(-w0_J_0_0*w0_J_1_2 + w0_J_0_2*w0_J_1_0);
double w0_J_invT_cse_2_2 = w0_tmpcse_J_invT_0*(w0_J_0_0*w0_J_1_1 - w0_J_0_1*w0_J_1_0);
double w0_tmpcse_grad_i_0 = 0.16666666666666671*w0_J_invT_cse_0_2;
double w0_tmpcse_grad_i_1 = 0.5*w0_J_invT_cse_0_0;
double w0_tmpcse_grad_i_2 = 0.5*w0_J_invT_cse_0_1;
double w0_tmpcse_grad_i_3 = w0_tmpcse_grad_i_1 + w0_tmpcse_grad_i_2;
double w0_tmpcse_grad_i_4 = 0.16666666666666671*w0_J_invT_cse_1_2;
double w0_tmpcse_grad_i_5 = 0.5*w0_J_invT_cse_1_0;
double w0_tmpcse_grad_i_6 = 0.5*w0_J_invT_cse_1_1;
double w0_tmpcse_grad_i_7 = w0_tmpcse_grad_i_5 + w0_tmpcse_grad_i_6;
double w0_tmpcse_grad_i_8 = 0.16666666666666671*w0_J_invT_cse_2_2;
double w0_tmpcse_grad_i_9 = 0.5*w0_J_invT_cse_2_0;
double w0_tmpcse_grad_i_10 = 0.5*w0_J_invT_cse_2_1;
double w0_tmpcse_grad_i_11 = w0_tmpcse_grad_i_10 + w0_tmpcse_grad_i_9;
double w0_tmpcse_grad_i_12 = 0.16666666666666666*w0_J_invT_cse_0_2;
double w0_tmpcse_grad_i_13 = -w0_tmpcse_grad_i_12;
double w0_tmpcse_grad_i_14 = 0.16666666666666666*w0_J_invT_cse_1_2;
double w0_tmpcse_grad_i_15 = -w0_tmpcse_grad_i_14;
double w0_tmpcse_grad_i_16 = 0.16666666666666666*w0_J_invT_cse_2_2;
double w0_tmpcse_grad_i_17 = -w0_tmpcse_grad_i_16;
double w0_grad_i0_0 = -w0_tmpcse_grad_i_0 - w0_tmpcse_grad_i_3;
double w0_grad_i0_1 = -w0_tmpcse_grad_i_4 - w0_tmpcse_grad_i_7;
double w0_grad_i0_2 = -w0_tmpcse_grad_i_11 - w0_tmpcse_grad_i_8;
double w0_grad_i1_0 = w0_tmpcse_grad_i_1 + w0_tmpcse_grad_i_13;
double w0_grad_i1_1 = w0_tmpcse_grad_i_15 + w0_tmpcse_grad_i_5;
double w0_grad_i1_2 = w0_tmpcse_grad_i_17 + w0_tmpcse_grad_i_9;
double w0_grad_i2_0 = w0_tmpcse_grad_i_13 + w0_tmpcse_grad_i_2;
double w0_grad_i2_1 = w0_tmpcse_grad_i_15 + w0_tmpcse_grad_i_6;
double w0_grad_i2_2 = w0_tmpcse_grad_i_10 + w0_tmpcse_grad_i_17;
double w0_grad_i3_0 = w0_tmpcse_grad_i_0 - w0_tmpcse_grad_i_3;
double w0_grad_i3_1 = w0_tmpcse_grad_i_4 - w0_tmpcse_grad_i_7;
double w0_grad_i3_2 = -w0_tmpcse_grad_i_11 + w0_tmpcse_grad_i_8;
double w0_grad_i4_0 = w0_tmpcse_grad_i_1 + w0_tmpcse_grad_i_12;
double w0_grad_i4_1 = w0_tmpcse_grad_i_14 + w0_tmpcse_grad_i_5;
double w0_grad_i4_2 = w0_tmpcse_grad_i_16 + w0_tmpcse_grad_i_9;
double w0_grad_i5_0 = w0_tmpcse_grad_i_12 + w0_tmpcse_grad_i_2;
double w0_grad_i5_1 = w0_tmpcse_grad_i_14 + w0_tmpcse_grad_i_6;
double w0_grad_i5_2 = w0_tmpcse_grad_i_10 + w0_tmpcse_grad_i_16;
double w0_grad_u_0 = src_0_0*trial_it0_cond*w0_grad_i0_0 + src_0_1*trial_it1_cond*w0_grad_i1_0 + src_0_2*trial_it2_cond*w0_grad_i2_0 + src_0_3*trial_it3_cond*w0_grad_i3_0 + src_0_4*trial_it4_cond*w0_grad_i4_0 + src_0_5*trial_it5_cond*w0_grad_i5_0;
double w0_grad_u_1 = src_0_0*trial_it0_cond*w0_grad_i0_1 + src_0_1*trial_it1_cond*w0_grad_i1_1 + src_0_2*trial_it2_cond*w0_grad_i2_1 + src_0_3*trial_it3_cond*w0_grad_i3_1 + src_0_4*trial_it4_cond*w0_grad_i4_1 + src_0_5*trial_it5_cond*w0_grad_i5_1;
double w0_grad_u_2 = src_0_0*trial_it0_cond*w0_grad_i0_2 + src_0_1*trial_it1_cond*w0_grad_i1_2 + src_0_2*trial_it2_cond*w0_grad_i2_2 + src_0_3*trial_it3_cond*w0_grad_i3_2 + src_0_4*trial_it4_cond*w0_grad_i4_2 + src_0_5*trial_it5_cond*w0_grad_i5_2;
double w0_tmpcse_dst_0 = 1.0*fabs(w0_J_det);
double w0_tmpcse_dst_1 = w0_grad_u_0*w0_tmpcse_dst_0;
double w0_tmpcse_dst_2 = w0_grad_u_1*w0_tmpcse_dst_0;
double w0_tmpcse_dst_3 = w0_grad_u_2*w0_tmpcse_dst_0;
double w0_tmpcse_dst_4 = w0_tmpcse_grad_i_12 + w0_tmpcse_grad_i_2;
double w0_tmpcse_dst_5 = src_0_0*w0_tmpcse_dst_0;
double w0_tmpcse_dst_6 = w0_tmpcse_grad_i_14 + w0_tmpcse_grad_i_6;
double w0_tmpcse_dst_7 = w0_tmpcse_grad_i_10 + w0_tmpcse_grad_i_16;
double w0_tmpcse_dst_8 = src_0_1*w0_tmpcse_dst_0;
double w0_tmpcse_dst_9 = src_0_2*w0_tmpcse_dst_0;
double w0_tmpcse_dst_10 = src_0_3*w0_tmpcse_dst_0;
double w0_tmpcse_dst_11 = src_0_4*w0_tmpcse_dst_0;
double w0_tmpcse_dst_12 = src_0_5*w0_tmpcse_dst_0;
double dst_0_0 = diag_bc_it0_cond*(w0_grad_i0_0*w0_tmpcse_dst_4*w0_tmpcse_dst_5 + w0_grad_i0_1*w0_tmpcse_dst_5*w0_tmpcse_dst_6 + w0_grad_i0_2*w0_tmpcse_dst_5*w0_tmpcse_dst_7) + test_it0_cond*(w0_grad_i0_0*w0_tmpcse_dst_1 + w0_grad_i0_1*w0_tmpcse_dst_2 + w0_grad_i0_2*w0_tmpcse_dst_3);
double dst_0_1 = diag_bc_it1_cond*(w0_grad_i1_0*w0_tmpcse_dst_4*w0_tmpcse_dst_8 + w0_grad_i1_1*w0_tmpcse_dst_6*w0_tmpcse_dst_8 + w0_grad_i1_2*w0_tmpcse_dst_7*w0_tmpcse_dst_8) + test_it1_cond*(w0_grad_i1_0*w0_tmpcse_dst_1 + w0_grad_i1_1*w0_tmpcse_dst_2 + w0_grad_i1_2*w0_tmpcse_dst_3);
double dst_0_2 = diag_bc_it2_cond*(w0_grad_i2_0*w0_tmpcse_dst_4*w0_tmpcse_dst_9 + w0_grad_i2_1*w0_tmpcse_dst_6*w0_tmpcse_dst_9 + w0_grad_i2_2*w0_tmpcse_dst_7*w0_tmpcse_dst_9) + test_it2_cond*(w0_grad_i2_0*w0_tmpcse_dst_1 + w0_grad_i2_1*w0_tmpcse_dst_2 + w0_grad_i2_2*w0_tmpcse_dst_3);
double dst_0_3 = diag_bc_it3_cond*(w0_grad_i3_0*w0_tmpcse_dst_10*w0_tmpcse_dst_4 + w0_grad_i3_1*w0_tmpcse_dst_10*w0_tmpcse_dst_6 + w0_grad_i3_2*w0_tmpcse_dst_10*w0_tmpcse_dst_7) + test_it3_cond*(w0_grad_i3_0*w0_tmpcse_dst_1 + w0_grad_i3_1*w0_tmpcse_dst_2 + w0_grad_i3_2*w0_tmpcse_dst_3);
double dst_0_4 = diag_bc_it4_cond*(w0_grad_i4_0*w0_tmpcse_dst_11*w0_tmpcse_dst_4 + w0_grad_i4_1*w0_tmpcse_dst_11*w0_tmpcse_dst_6 + w0_grad_i4_2*w0_tmpcse_dst_11*w0_tmpcse_dst_7) + test_it4_cond*(w0_grad_i4_0*w0_tmpcse_dst_1 + w0_grad_i4_1*w0_tmpcse_dst_2 + w0_grad_i4_2*w0_tmpcse_dst_3);
double dst_0_5 = diag_bc_it5_cond*(w0_grad_i5_0*w0_tmpcse_dst_12*w0_tmpcse_dst_4 + w0_grad_i5_1*w0_tmpcse_dst_12*w0_tmpcse_dst_6 + w0_grad_i5_2*w0_tmpcse_dst_12*w0_tmpcse_dst_7) + test_it5_cond*(w0_grad_i5_0*w0_tmpcse_dst_1 + w0_grad_i5_1*w0_tmpcse_dst_2 + w0_grad_i5_2*w0_tmpcse_dst_3);
double w1_tmpcse_J_0 = 0.5*r_0 + 0.5*r_1;
double w1_tmpcse_J_1 = -1.0/2.0*r_0 + (1.0/2.0)*r_1;
double w1_J_0_0 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_0 + wedge_surf_phy_coords_1_1_0);
double w1_J_0_1 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_0 + wedge_surf_phy_coords_1_2_0);
double w1_J_0_2 = w1_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_1_0_0 + 0.33333333333333331*wedge_surf_phy_coords_1_1_0 + 0.33333333333333331*wedge_surf_phy_coords_1_2_0);
double w1_J_1_0 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_1 + wedge_surf_phy_coords_1_1_1);
double w1_J_1_1 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_1 + wedge_surf_phy_coords_1_2_1);
double w1_J_1_2 = w1_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_1_0_1 + 0.33333333333333331*wedge_surf_phy_coords_1_1_1 + 0.33333333333333331*wedge_surf_phy_coords_1_2_1);
double w1_J_2_0 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_2 + wedge_surf_phy_coords_1_1_2);
double w1_J_2_1 = w1_tmpcse_J_0*(-wedge_surf_phy_coords_1_0_2 + wedge_surf_phy_coords_1_2_2);
double w1_J_2_2 = w1_tmpcse_J_1*(0.33333333333333343*wedge_surf_phy_coords_1_0_2 + 0.33333333333333331*wedge_surf_phy_coords_1_1_2 + 0.33333333333333331*wedge_surf_phy_coords_1_2_2);
double w1_J_det = w1_J_0_0*w1_J_1_1*w1_J_2_2 - w1_J_0_0*w1_J_1_2*w1_J_2_1 - w1_J_0_1*w1_J_1_0*w1_J_2_2 + w1_J_0_1*w1_J_1_2*w1_J_2_0 + w1_J_0_2*w1_J_1_0*w1_J_2_1 - w1_J_0_2*w1_J_1_1*w1_J_2_0;
double w1_tmpcse_J_invT_0 = 1.0/w1_J_det;
double w1_J_invT_cse_0_0 = w1_tmpcse_J_invT_0*(w1_J_1_1*w1_J_2_2 - w1_J_1_2*w1_J_2_1);
double w1_J_invT_cse_0_1 = w1_tmpcse_J_invT_0*(-w1_J_1_0*w1_J_2_2 + w1_J_1_2*w1_J_2_0);
double w1_J_invT_cse_0_2 = w1_tmpcse_J_invT_0*(w1_J_1_0*w1_J_2_1 - w1_J_1_1*w1_J_2_0);
double w1_J_invT_cse_1_0 = w1_tmpcse_J_invT_0*(-w1_J_0_1*w1_J_2_2 + w1_J_0_2*w1_J_2_1);
double w1_J_invT_cse_1_1 = w1_tmpcse_J_invT_0*(w1_J_0_0*w1_J_2_2 - w1_J_0_2*w1_J_2_0);
double w1_J_invT_cse_1_2 = w1_tmpcse_J_invT_0*(-w1_J_0_0*w1_J_2_1 + w1_J_0_1*w1_J_2_0);
double w1_J_invT_cse_2_0 = w1_tmpcse_J_invT_0*(w1_J_0_1*w1_J_1_2 - w1_J_0_2*w1_J_1_1);
double w1_J_invT_cse_2_1 = w1_tmpcse_J_invT_0*(-w1_J_0_0*w1_J_1_2 + w1_J_0_2*w1_J_1_0);
double w1_J_invT_cse_2_2 = w1_tmpcse_J_invT_0*(w1_J_0_0*w1_J_1_1 - w1_J_0_1*w1_J_1_0);
double w1_tmpcse_grad_i_0 = 0.16666666666666671*w1_J_invT_cse_0_2;
double w1_tmpcse_grad_i_1 = 0.5*w1_J_invT_cse_0_0;
double w1_tmpcse_grad_i_2 = 0.5*w1_J_invT_cse_0_1;
double w1_tmpcse_grad_i_3 = w1_tmpcse_grad_i_1 + w1_tmpcse_grad_i_2;
double w1_tmpcse_grad_i_4 = 0.16666666666666671*w1_J_invT_cse_1_2;
double w1_tmpcse_grad_i_5 = 0.5*w1_J_invT_cse_1_0;
double w1_tmpcse_grad_i_6 = 0.5*w1_J_invT_cse_1_1;
double w1_tmpcse_grad_i_7 = w1_tmpcse_grad_i_5 + w1_tmpcse_grad_i_6;
double w1_tmpcse_grad_i_8 = 0.16666666666666671*w1_J_invT_cse_2_2;
double w1_tmpcse_grad_i_9 = 0.5*w1_J_invT_cse_2_0;
double w1_tmpcse_grad_i_10 = 0.5*w1_J_invT_cse_2_1;
double w1_tmpcse_grad_i_11 = w1_tmpcse_grad_i_10 + w1_tmpcse_grad_i_9;
double w1_tmpcse_grad_i_12 = 0.16666666666666666*w1_J_invT_cse_0_2;
double w1_tmpcse_grad_i_13 = -w1_tmpcse_grad_i_12;
double w1_tmpcse_grad_i_14 = 0.16666666666666666*w1_J_invT_cse_1_2;
double w1_tmpcse_grad_i_15 = -w1_tmpcse_grad_i_14;
double w1_tmpcse_grad_i_16 = 0.16666666666666666*w1_J_invT_cse_2_2;
double w1_tmpcse_grad_i_17 = -w1_tmpcse_grad_i_16;
double w1_grad_i0_0 = -w1_tmpcse_grad_i_0 - w1_tmpcse_grad_i_3;
double w1_grad_i0_1 = -w1_tmpcse_grad_i_4 - w1_tmpcse_grad_i_7;
double w1_grad_i0_2 = -w1_tmpcse_grad_i_11 - w1_tmpcse_grad_i_8;
double w1_grad_i1_0 = w1_tmpcse_grad_i_1 + w1_tmpcse_grad_i_13;
double w1_grad_i1_1 = w1_tmpcse_grad_i_15 + w1_tmpcse_grad_i_5;
double w1_grad_i1_2 = w1_tmpcse_grad_i_17 + w1_tmpcse_grad_i_9;
double w1_grad_i2_0 = w1_tmpcse_grad_i_13 + w1_tmpcse_grad_i_2;
double w1_grad_i2_1 = w1_tmpcse_grad_i_15 + w1_tmpcse_grad_i_6;
double w1_grad_i2_2 = w1_tmpcse_grad_i_10 + w1_tmpcse_grad_i_17;
double w1_grad_i3_0 = w1_tmpcse_grad_i_0 - w1_tmpcse_grad_i_3;
double w1_grad_i3_1 = w1_tmpcse_grad_i_4 - w1_tmpcse_grad_i_7;
double w1_grad_i3_2 = -w1_tmpcse_grad_i_11 + w1_tmpcse_grad_i_8;
double w1_grad_i4_0 = w1_tmpcse_grad_i_1 + w1_tmpcse_grad_i_12;
double w1_grad_i4_1 = w1_tmpcse_grad_i_14 + w1_tmpcse_grad_i_5;
double w1_grad_i4_2 = w1_tmpcse_grad_i_16 + w1_tmpcse_grad_i_9;
double w1_grad_i5_0 = w1_tmpcse_grad_i_12 + w1_tmpcse_grad_i_2;
double w1_grad_i5_1 = w1_tmpcse_grad_i_14 + w1_tmpcse_grad_i_6;
double w1_grad_i5_2 = w1_tmpcse_grad_i_10 + w1_tmpcse_grad_i_16;
double w1_grad_u_0 = src_1_0*trial_it0_cond*w1_grad_i0_0 + src_1_1*trial_it1_cond*w1_grad_i1_0 + src_1_2*trial_it2_cond*w1_grad_i2_0 + src_1_3*trial_it3_cond*w1_grad_i3_0 + src_1_4*trial_it4_cond*w1_grad_i4_0 + src_1_5*trial_it5_cond*w1_grad_i5_0;
double w1_grad_u_1 = src_1_0*trial_it0_cond*w1_grad_i0_1 + src_1_1*trial_it1_cond*w1_grad_i1_1 + src_1_2*trial_it2_cond*w1_grad_i2_1 + src_1_3*trial_it3_cond*w1_grad_i3_1 + src_1_4*trial_it4_cond*w1_grad_i4_1 + src_1_5*trial_it5_cond*w1_grad_i5_1;
double w1_grad_u_2 = src_1_0*trial_it0_cond*w1_grad_i0_2 + src_1_1*trial_it1_cond*w1_grad_i1_2 + src_1_2*trial_it2_cond*w1_grad_i2_2 + src_1_3*trial_it3_cond*w1_grad_i3_2 + src_1_4*trial_it4_cond*w1_grad_i4_2 + src_1_5*trial_it5_cond*w1_grad_i5_2;
double w1_tmpcse_dst_0 = 1.0*fabs(w1_J_det);
double w1_tmpcse_dst_1 = w1_grad_u_0*w1_tmpcse_dst_0;
double w1_tmpcse_dst_2 = w1_grad_u_1*w1_tmpcse_dst_0;
double w1_tmpcse_dst_3 = w1_grad_u_2*w1_tmpcse_dst_0;
double w1_tmpcse_dst_4 = w1_tmpcse_grad_i_12 + w1_tmpcse_grad_i_2;
double w1_tmpcse_dst_5 = src_1_0*w1_tmpcse_dst_0;
double w1_tmpcse_dst_6 = w1_tmpcse_grad_i_14 + w1_tmpcse_grad_i_6;
double w1_tmpcse_dst_7 = w1_tmpcse_grad_i_10 + w1_tmpcse_grad_i_16;
double w1_tmpcse_dst_8 = src_1_1*w1_tmpcse_dst_0;
double w1_tmpcse_dst_9 = src_1_2*w1_tmpcse_dst_0;
double w1_tmpcse_dst_10 = src_1_3*w1_tmpcse_dst_0;
double w1_tmpcse_dst_11 = src_1_4*w1_tmpcse_dst_0;
double w1_tmpcse_dst_12 = src_1_5*w1_tmpcse_dst_0;
double dst_1_0 = diag_bc_it0_cond*(w1_grad_i0_0*w1_tmpcse_dst_4*w1_tmpcse_dst_5 + w1_grad_i0_1*w1_tmpcse_dst_5*w1_tmpcse_dst_6 + w1_grad_i0_2*w1_tmpcse_dst_5*w1_tmpcse_dst_7) + test_it0_cond*(w1_grad_i0_0*w1_tmpcse_dst_1 + w1_grad_i0_1*w1_tmpcse_dst_2 + w1_grad_i0_2*w1_tmpcse_dst_3);
double dst_1_1 = diag_bc_it1_cond*(w1_grad_i1_0*w1_tmpcse_dst_4*w1_tmpcse_dst_8 + w1_grad_i1_1*w1_tmpcse_dst_6*w1_tmpcse_dst_8 + w1_grad_i1_2*w1_tmpcse_dst_7*w1_tmpcse_dst_8) + test_it1_cond*(w1_grad_i1_0*w1_tmpcse_dst_1 + w1_grad_i1_1*w1_tmpcse_dst_2 + w1_grad_i1_2*w1_tmpcse_dst_3);
double dst_1_2 = diag_bc_it2_cond*(w1_grad_i2_0*w1_tmpcse_dst_4*w1_tmpcse_dst_9 + w1_grad_i2_1*w1_tmpcse_dst_6*w1_tmpcse_dst_9 + w1_grad_i2_2*w1_tmpcse_dst_7*w1_tmpcse_dst_9) + test_it2_cond*(w1_grad_i2_0*w1_tmpcse_dst_1 + w1_grad_i2_1*w1_tmpcse_dst_2 + w1_grad_i2_2*w1_tmpcse_dst_3);
double dst_1_3 = diag_bc_it3_cond*(w1_grad_i3_0*w1_tmpcse_dst_10*w1_tmpcse_dst_4 + w1_grad_i3_1*w1_tmpcse_dst_10*w1_tmpcse_dst_6 + w1_grad_i3_2*w1_tmpcse_dst_10*w1_tmpcse_dst_7) + test_it3_cond*(w1_grad_i3_0*w1_tmpcse_dst_1 + w1_grad_i3_1*w1_tmpcse_dst_2 + w1_grad_i3_2*w1_tmpcse_dst_3);
double dst_1_4 = diag_bc_it4_cond*(w1_grad_i4_0*w1_tmpcse_dst_11*w1_tmpcse_dst_4 + w1_grad_i4_1*w1_tmpcse_dst_11*w1_tmpcse_dst_6 + w1_grad_i4_2*w1_tmpcse_dst_11*w1_tmpcse_dst_7) + test_it4_cond*(w1_grad_i4_0*w1_tmpcse_dst_1 + w1_grad_i4_1*w1_tmpcse_dst_2 + w1_grad_i4_2*w1_tmpcse_dst_3);
double dst_1_5 = diag_bc_it5_cond*(w1_grad_i5_0*w1_tmpcse_dst_12*w1_tmpcse_dst_4 + w1_grad_i5_1*w1_tmpcse_dst_12*w1_tmpcse_dst_6 + w1_grad_i5_2*w1_tmpcse_dst_12*w1_tmpcse_dst_7) + test_it5_cond*(w1_grad_i5_0*w1_tmpcse_dst_1 + w1_grad_i5_1*w1_tmpcse_dst_2 + w1_grad_i5_2*w1_tmpcse_dst_3);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell), dst_0_0);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell), dst_0_1 + dst_1_2);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell), dst_0_2 + dst_1_1);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell + 1), dst_0_3);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1), dst_0_4 + dst_1_5);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1), dst_0_5 + dst_1_4);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell), dst_1_0);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1), dst_1_3);

      }
    // Kernel body:
};

static_assert( linalg::OperatorLike< LaplaceKerngen< float > > );
static_assert( linalg::OperatorLike< LaplaceKerngen< double > > );
static_assert( linalg::GCACapable< LaplaceKerngen< float > > );

} // namespace terra::fe::wedge::operators::shell
