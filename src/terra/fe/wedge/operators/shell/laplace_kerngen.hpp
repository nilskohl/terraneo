
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
        /*
        const int N1 = ( src_.extent(1) - 1 );
        const int N2 = ( src_.extent(2) - 1 );
        const int N3 = ( src_.extent(3) - 1 );
        // Convert flat index back to 4D:
        int local_subdomain_id = idx / ( N1 * N2 * N3 );
        int r                  = idx % ( N1 * N2 * N3 );

        int x_cell = r / ( N2 * N3 );
        r          = r % ( N2 * N3 );

        int y_cell = r / N3;
        int r_cell = r % N3;*/

        // Kernel body:
        // Kernel body:

double wedge_surf_phy_coords[2][3][3];
double quad_surface_coords[2][2][3];
quad_surface_coords[0][0][0] = grid_(local_subdomain_id, x_cell, y_cell, 0);
quad_surface_coords[0][0][1] = grid_(local_subdomain_id, x_cell, y_cell, 1);
quad_surface_coords[0][0][2] = grid_(local_subdomain_id, x_cell, y_cell, 2);
quad_surface_coords[0][1][0] = grid_(local_subdomain_id, x_cell, y_cell + 1, 0);
quad_surface_coords[0][1][1] = grid_(local_subdomain_id, x_cell, y_cell + 1, 1);
quad_surface_coords[0][1][2] = grid_(local_subdomain_id, x_cell, y_cell + 1, 2);
quad_surface_coords[1][0][0] = grid_(local_subdomain_id, x_cell + 1, y_cell, 0);
quad_surface_coords[1][0][1] = grid_(local_subdomain_id, x_cell + 1, y_cell, 1);
quad_surface_coords[1][0][2] = grid_(local_subdomain_id, x_cell + 1, y_cell, 2);
quad_surface_coords[1][1][0] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 0);
quad_surface_coords[1][1][1] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 1);
quad_surface_coords[1][1][2] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 2);
wedge_surf_phy_coords[0][0][0] = quad_surface_coords[0][0][0];
wedge_surf_phy_coords[0][0][1] = quad_surface_coords[0][0][1];
wedge_surf_phy_coords[0][0][2] = quad_surface_coords[0][0][2];
wedge_surf_phy_coords[0][1][0] = quad_surface_coords[1][0][0];
wedge_surf_phy_coords[0][1][1] = quad_surface_coords[1][0][1];
wedge_surf_phy_coords[0][1][2] = quad_surface_coords[1][0][2];
wedge_surf_phy_coords[0][2][0] = quad_surface_coords[0][1][0];
wedge_surf_phy_coords[0][2][1] = quad_surface_coords[0][1][1];
wedge_surf_phy_coords[0][2][2] = quad_surface_coords[0][1][2];
wedge_surf_phy_coords[1][0][0] = quad_surface_coords[1][1][0];
wedge_surf_phy_coords[1][0][1] = quad_surface_coords[1][1][1];
wedge_surf_phy_coords[1][0][2] = quad_surface_coords[1][1][2];
wedge_surf_phy_coords[1][1][0] = quad_surface_coords[0][1][0];
wedge_surf_phy_coords[1][1][1] = quad_surface_coords[0][1][1];
wedge_surf_phy_coords[1][1][2] = quad_surface_coords[0][1][2];
wedge_surf_phy_coords[1][2][0] = quad_surface_coords[1][0][0];
wedge_surf_phy_coords[1][2][1] = quad_surface_coords[1][0][1];
wedge_surf_phy_coords[1][2][2] = quad_surface_coords[1][0][2];
double r_0 = radii_(local_subdomain_id, r_cell);
double r_1 = radii_(local_subdomain_id, r_cell + 1);
double src_local_hex[2][6];
src_local_hex[0][0] = src_(local_subdomain_id, x_cell, y_cell, r_cell);
src_local_hex[0][1] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell);
src_local_hex[0][2] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell);
src_local_hex[0][3] = src_(local_subdomain_id, x_cell, y_cell, r_cell + 1);
src_local_hex[0][4] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1);
src_local_hex[0][5] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1);
src_local_hex[1][0] = src_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell);
src_local_hex[1][1] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell);
src_local_hex[1][2] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell);
src_local_hex[1][3] = src_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1);
src_local_hex[1][4] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1);
src_local_hex[1][5] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1);
double qp_array[6][3];
double qw_array[6];
qp_array[0][0] = 0.66666666666666663;
qp_array[1][0] = 0.16666666666666671;
qp_array[2][0] = 0.16666666666666671;
qp_array[3][0] = 0.66666666666666663;
qp_array[4][0] = 0.16666666666666671;
qp_array[5][0] = 0.16666666666666671;
qp_array[0][1] = 0.16666666666666671;
qp_array[1][1] = 0.66666666666666663;
qp_array[2][1] = 0.16666666666666671;
qp_array[3][1] = 0.16666666666666671;
qp_array[4][1] = 0.66666666666666663;
qp_array[5][1] = 0.16666666666666671;
qp_array[0][2] = -0.57735026918962573;
qp_array[1][2] = -0.57735026918962573;
qp_array[2][2] = -0.57735026918962573;
qp_array[3][2] = 0.57735026918962573;
qp_array[4][2] = 0.57735026918962573;
qp_array[5][2] = 0.57735026918962573;
qw_array[0] = 0.16666666666666671;
qw_array[1] = 0.16666666666666671;
qw_array[2] = 0.16666666666666671;
qw_array[3] = 0.16666666666666671;
qw_array[4] = 0.16666666666666671;
qw_array[5] = 0.16666666666666671;
int cmb_shift = ((treat_boundary_ && diagonal_ == false && r_cell == 0) ? (
   3
)
: (
   0
));
int max_rad = -1 + radii_.extent(1);
int surface_shift = ((treat_boundary_ && diagonal_ == false && max_rad == r_cell + 1) ? (
   3
)
: (
   0
));
int trial_it0_cond;
if (diagonal_ == false && cmb_shift <= 0 && surface_shift < 6) {
   trial_it0_cond = 1;
}
else {
   trial_it0_cond = 0;
};
int test_it0_cond;
if (diagonal_ == false && cmb_shift <= 0 && surface_shift < 6) {
   test_it0_cond = 1;
}
else {
   test_it0_cond = 0;
};
int diag_bc_it0_cond;
if (surface_shift <= 0 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 6) {
   diag_bc_it0_cond = 1;
}
else {
   diag_bc_it0_cond = 0;
};
int trial_it1_cond;
if (diagonal_ == false && cmb_shift <= 1 && surface_shift < 5) {
   trial_it1_cond = 1;
}
else {
   trial_it1_cond = 0;
};
int test_it1_cond;
if (diagonal_ == false && cmb_shift <= 1 && surface_shift < 5) {
   test_it1_cond = 1;
}
else {
   test_it1_cond = 0;
};
int diag_bc_it1_cond;
if (surface_shift <= 1 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 5) {
   diag_bc_it1_cond = 1;
}
else {
   diag_bc_it1_cond = 0;
};
int trial_it2_cond;
if (diagonal_ == false && cmb_shift <= 2 && surface_shift < 4) {
   trial_it2_cond = 1;
}
else {
   trial_it2_cond = 0;
};
int test_it2_cond;
if (diagonal_ == false && cmb_shift <= 2 && surface_shift < 4) {
   test_it2_cond = 1;
}
else {
   test_it2_cond = 0;
};
int diag_bc_it2_cond;
if (surface_shift <= 2 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 4) {
   diag_bc_it2_cond = 1;
}
else {
   diag_bc_it2_cond = 0;
};
int trial_it3_cond;
if (diagonal_ == false && cmb_shift <= 3 && surface_shift < 3) {
   trial_it3_cond = 1;
}
else {
   trial_it3_cond = 0;
};
int test_it3_cond;
if (diagonal_ == false && cmb_shift <= 3 && surface_shift < 3) {
   test_it3_cond = 1;
}
else {
   test_it3_cond = 0;
};
int diag_bc_it3_cond;
if (surface_shift <= 3 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 3) {
   diag_bc_it3_cond = 1;
}
else {
   diag_bc_it3_cond = 0;
};
int trial_it4_cond;
if (diagonal_ == false && cmb_shift <= 4 && surface_shift < 2) {
   trial_it4_cond = 1;
}
else {
   trial_it4_cond = 0;
};
int test_it4_cond;
if (diagonal_ == false && cmb_shift <= 4 && surface_shift < 2) {
   test_it4_cond = 1;
}
else {
   test_it4_cond = 0;
};
int diag_bc_it4_cond;
if (surface_shift <= 4 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 2) {
   diag_bc_it4_cond = 1;
}
else {
   diag_bc_it4_cond = 0;
};
int trial_it5_cond;
if (diagonal_ == false && cmb_shift <= 5 && surface_shift < 1) {
   trial_it5_cond = 1;
}
else {
   trial_it5_cond = 0;
};
int test_it5_cond;
if (diagonal_ == false && cmb_shift <= 5 && surface_shift < 1) {
   test_it5_cond = 1;
}
else {
   test_it5_cond = 0;
};
int diag_bc_it5_cond;
if (surface_shift <= 5 && (treat_boundary_ == true && (max_rad == r_cell + 1 || r_cell == 0) || diagonal_ == true) && cmb_shift < 1) {
   diag_bc_it5_cond = 1;
}
else {
   diag_bc_it5_cond = 0;
};
double dst_array[2][6];

dst_array[0][0] = 0.0;
dst_array[0][1] = 0.0;
dst_array[0][2] = 0.0;
dst_array[0][3] = 0.0;
dst_array[0][4] = 0.0;
dst_array[0][5] = 0.0;
dst_array[1][0] = 0.0;
dst_array[1][1] = 0.0;
dst_array[1][2] = 0.0;
dst_array[1][3] = 0.0;
dst_array[1][4] = 0.0;
dst_array[1][5] = 0.0;
int w = 0;
for (w = 0; w < 2; w += 1) {
   int q = 0;
   for (q = 0; q < 6; q += 1) {
      double qp_0 = qp_array[q][0];
      double qp_1 = qp_array[q][1];
      double qp_2 = qp_array[q][2];
      double qw = qw_array[q];
      double wedge_tmp_symbols_0_0 = wedge_surf_phy_coords[w][0][0];
      double wedge_tmp_symbols_1_0 = wedge_surf_phy_coords[w][1][0];
      double wedge_tmp_symbols_2_0 = wedge_surf_phy_coords[w][2][0];
      double wedge_tmp_symbols_0_1 = wedge_surf_phy_coords[w][0][1];
      double wedge_tmp_symbols_1_1 = wedge_surf_phy_coords[w][1][1];
      double wedge_tmp_symbols_2_1 = wedge_surf_phy_coords[w][2][1];
      double wedge_tmp_symbols_0_2 = wedge_surf_phy_coords[w][0][2];
      double wedge_tmp_symbols_1_2 = wedge_surf_phy_coords[w][1][2];
      double wedge_tmp_symbols_2_2 = wedge_surf_phy_coords[w][2][2];
      double tmpcse_J_0 = -1.0/2.0*r_0 + (1.0/2.0)*r_1;
      double tmpcse_J_1 = r_0 + tmpcse_J_0*(qp_2 + 1);
      double tmpcse_J_2 = -qp_0 - qp_1 + 1;
      double J_0_0 = tmpcse_J_1*(-wedge_tmp_symbols_0_0 + wedge_tmp_symbols_1_0);
      double J_0_1 = tmpcse_J_1*(-wedge_tmp_symbols_0_0 + wedge_tmp_symbols_2_0);
      double J_0_2 = tmpcse_J_0*(qp_0*wedge_tmp_symbols_1_0 + qp_1*wedge_tmp_symbols_2_0 + tmpcse_J_2*wedge_tmp_symbols_0_0);
      double J_1_0 = tmpcse_J_1*(-wedge_tmp_symbols_0_1 + wedge_tmp_symbols_1_1);
      double J_1_1 = tmpcse_J_1*(-wedge_tmp_symbols_0_1 + wedge_tmp_symbols_2_1);
      double J_1_2 = tmpcse_J_0*(qp_0*wedge_tmp_symbols_1_1 + qp_1*wedge_tmp_symbols_2_1 + tmpcse_J_2*wedge_tmp_symbols_0_1);
      double J_2_0 = tmpcse_J_1*(-wedge_tmp_symbols_0_2 + wedge_tmp_symbols_1_2);
      double J_2_1 = tmpcse_J_1*(-wedge_tmp_symbols_0_2 + wedge_tmp_symbols_2_2);
      double J_2_2 = tmpcse_J_0*(qp_0*wedge_tmp_symbols_1_2 + qp_1*wedge_tmp_symbols_2_2 + tmpcse_J_2*wedge_tmp_symbols_0_2);
      double J_det = J_0_0*J_1_1*J_2_2 - J_0_0*J_1_2*J_2_1 - J_0_1*J_1_0*J_2_2 + J_0_1*J_1_2*J_2_0 + J_0_2*J_1_0*J_2_1 - J_0_2*J_1_1*J_2_0;
      double tmpcse_J_invT_0 = 1.0/J_det;
      double J_invT_cse_0_0 = tmpcse_J_invT_0*(J_1_1*J_2_2 - J_1_2*J_2_1);
      double J_invT_cse_0_1 = tmpcse_J_invT_0*(-J_1_0*J_2_2 + J_1_2*J_2_0);
      double J_invT_cse_0_2 = tmpcse_J_invT_0*(J_1_0*J_2_1 - J_1_1*J_2_0);
      double J_invT_cse_1_0 = tmpcse_J_invT_0*(-J_0_1*J_2_2 + J_0_2*J_2_1);
      double J_invT_cse_1_1 = tmpcse_J_invT_0*(J_0_0*J_2_2 - J_0_2*J_2_0);
      double J_invT_cse_1_2 = tmpcse_J_invT_0*(-J_0_0*J_2_1 + J_0_1*J_2_0);
      double J_invT_cse_2_0 = tmpcse_J_invT_0*(J_0_1*J_1_2 - J_0_2*J_1_1);
      double J_invT_cse_2_1 = tmpcse_J_invT_0*(-J_0_0*J_1_2 + J_0_2*J_1_0);
      double J_invT_cse_2_2 = tmpcse_J_invT_0*(J_0_0*J_1_1 - J_0_1*J_1_0);
      double tmpcse_grad_i_0 = (1.0/2.0)*qp_2;
      double tmpcse_grad_i_1 = tmpcse_grad_i_0 - 1.0/2.0;
      double tmpcse_grad_i_2 = (1.0/2.0)*qp_0;
      double tmpcse_grad_i_3 = (1.0/2.0)*qp_1;
      double tmpcse_grad_i_4 = tmpcse_grad_i_2 + tmpcse_grad_i_3 - 1.0/2.0;
      double tmpcse_grad_i_5 = J_invT_cse_0_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_6 = -tmpcse_grad_i_1;
      double tmpcse_grad_i_7 = J_invT_cse_1_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_8 = J_invT_cse_2_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_9 = J_invT_cse_0_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_10 = J_invT_cse_1_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_11 = J_invT_cse_2_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_12 = tmpcse_grad_i_0 + 1.0/2.0;
      double tmpcse_grad_i_13 = -tmpcse_grad_i_12;
      double tmpcse_grad_i_14 = -tmpcse_grad_i_4;
      double grad_i0_0 = J_invT_cse_0_0*tmpcse_grad_i_1 + J_invT_cse_0_1*tmpcse_grad_i_1 + J_invT_cse_0_2*tmpcse_grad_i_4;
      double grad_i0_1 = J_invT_cse_1_0*tmpcse_grad_i_1 + J_invT_cse_1_1*tmpcse_grad_i_1 + J_invT_cse_1_2*tmpcse_grad_i_4;
      double grad_i0_2 = J_invT_cse_2_0*tmpcse_grad_i_1 + J_invT_cse_2_1*tmpcse_grad_i_1 + J_invT_cse_2_2*tmpcse_grad_i_4;
      double grad_i1_0 = J_invT_cse_0_0*tmpcse_grad_i_6 - tmpcse_grad_i_5;
      double grad_i1_1 = J_invT_cse_1_0*tmpcse_grad_i_6 - tmpcse_grad_i_7;
      double grad_i1_2 = J_invT_cse_2_0*tmpcse_grad_i_6 - tmpcse_grad_i_8;
      double grad_i2_0 = J_invT_cse_0_1*tmpcse_grad_i_6 - tmpcse_grad_i_9;
      double grad_i2_1 = J_invT_cse_1_1*tmpcse_grad_i_6 - tmpcse_grad_i_10;
      double grad_i2_2 = J_invT_cse_2_1*tmpcse_grad_i_6 - tmpcse_grad_i_11;
      double grad_i3_0 = J_invT_cse_0_0*tmpcse_grad_i_13 + J_invT_cse_0_1*tmpcse_grad_i_13 + J_invT_cse_0_2*tmpcse_grad_i_14;
      double grad_i3_1 = J_invT_cse_1_0*tmpcse_grad_i_13 + J_invT_cse_1_1*tmpcse_grad_i_13 + J_invT_cse_1_2*tmpcse_grad_i_14;
      double grad_i3_2 = J_invT_cse_2_0*tmpcse_grad_i_13 + J_invT_cse_2_1*tmpcse_grad_i_13 + J_invT_cse_2_2*tmpcse_grad_i_14;
      double grad_i4_0 = J_invT_cse_0_0*tmpcse_grad_i_12 + tmpcse_grad_i_5;
      double grad_i4_1 = J_invT_cse_1_0*tmpcse_grad_i_12 + tmpcse_grad_i_7;
      double grad_i4_2 = J_invT_cse_2_0*tmpcse_grad_i_12 + tmpcse_grad_i_8;
      double grad_i5_0 = J_invT_cse_0_1*tmpcse_grad_i_12 + tmpcse_grad_i_9;
      double grad_i5_1 = J_invT_cse_1_1*tmpcse_grad_i_12 + tmpcse_grad_i_10;
      double grad_i5_2 = J_invT_cse_2_1*tmpcse_grad_i_12 + tmpcse_grad_i_11;
      double src_tmp_symbols_0 = src_local_hex[w][0];
      double src_tmp_symbols_1 = src_local_hex[w][1];
      double src_tmp_symbols_2 = src_local_hex[w][2];
      double src_tmp_symbols_3 = src_local_hex[w][3];
      double src_tmp_symbols_4 = src_local_hex[w][4];
      double src_tmp_symbols_5 = src_local_hex[w][5];
      double grad_u_0 = grad_i0_0*src_tmp_symbols_0 + grad_i1_0*src_tmp_symbols_1 + grad_i2_0*src_tmp_symbols_2 + grad_i3_0*src_tmp_symbols_3 + grad_i4_0*src_tmp_symbols_4 + grad_i5_0*src_tmp_symbols_5;
      double grad_u_1 = grad_i0_1*src_tmp_symbols_0 + grad_i1_1*src_tmp_symbols_1 + grad_i2_1*src_tmp_symbols_2 + grad_i3_1*src_tmp_symbols_3 + grad_i4_1*src_tmp_symbols_4 + grad_i5_1*src_tmp_symbols_5;
      double grad_u_2 = grad_i0_2*src_tmp_symbols_0 + grad_i1_2*src_tmp_symbols_1 + grad_i2_2*src_tmp_symbols_2 + grad_i3_2*src_tmp_symbols_3 + grad_i4_2*src_tmp_symbols_4 + grad_i5_2*src_tmp_symbols_5;
      double tmpcse_dst_0 = qw*fabs(J_det);
      double tmpcse_dst_1 = grad_u_0*tmpcse_dst_0;
      double tmpcse_dst_2 = grad_u_1*tmpcse_dst_0;
      double tmpcse_dst_3 = grad_u_2*tmpcse_dst_0;
      double tmpcse_dst_4 = src_tmp_symbols_5*tmpcse_dst_0;
      double tmpcse_dst_5 = tmpcse_dst_4*(J_invT_cse_0_1*tmpcse_grad_i_12 + tmpcse_grad_i_9);
      double tmpcse_dst_6 = tmpcse_dst_4*(J_invT_cse_1_1*tmpcse_grad_i_12 + tmpcse_grad_i_10);
      double tmpcse_dst_7 = tmpcse_dst_4*(J_invT_cse_2_1*tmpcse_grad_i_12 + tmpcse_grad_i_11);
      dst_array[w][0] += grad_i0_0*tmpcse_dst_1 + grad_i0_0*tmpcse_dst_5 + grad_i0_1*tmpcse_dst_2 + grad_i0_1*tmpcse_dst_6 + grad_i0_2*tmpcse_dst_3 + grad_i0_2*tmpcse_dst_7;
      dst_array[w][1] += grad_i1_0*tmpcse_dst_1 + grad_i1_0*tmpcse_dst_5 + grad_i1_1*tmpcse_dst_2 + grad_i1_1*tmpcse_dst_6 + grad_i1_2*tmpcse_dst_3 + grad_i1_2*tmpcse_dst_7;
      dst_array[w][2] += grad_i2_0*tmpcse_dst_1 + grad_i2_0*tmpcse_dst_5 + grad_i2_1*tmpcse_dst_2 + grad_i2_1*tmpcse_dst_6 + grad_i2_2*tmpcse_dst_3 + grad_i2_2*tmpcse_dst_7;
      dst_array[w][3] += grad_i3_0*tmpcse_dst_1 + grad_i3_0*tmpcse_dst_5 + grad_i3_1*tmpcse_dst_2 + grad_i3_1*tmpcse_dst_6 + grad_i3_2*tmpcse_dst_3 + grad_i3_2*tmpcse_dst_7;
      dst_array[w][4] += grad_i4_0*tmpcse_dst_1 + grad_i4_0*tmpcse_dst_5 + grad_i4_1*tmpcse_dst_2 + grad_i4_1*tmpcse_dst_6 + grad_i4_2*tmpcse_dst_3 + grad_i4_2*tmpcse_dst_7;
      dst_array[w][5] += grad_i5_0*tmpcse_dst_1 + grad_i5_0*tmpcse_dst_5 + grad_i5_1*tmpcse_dst_2 + grad_i5_1*tmpcse_dst_6 + grad_i5_2*tmpcse_dst_3 + grad_i5_2*tmpcse_dst_7;
   };
};

Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell), dst_array[0][0]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell), dst_array[0][1] + dst_array[1][2]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell), dst_array[0][2] + dst_array[1][1]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell + 1), dst_array[0][3]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1), dst_array[0][4] + dst_array[1][5]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1), dst_array[0][5] + dst_array[1][4]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell), dst_array[1][0]);
Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1), dst_array[1][3]);

    }
    // Kernel body:
};

static_assert( linalg::OperatorLike< LaplaceKerngen< float > > );
static_assert( linalg::OperatorLike< LaplaceKerngen< double > > );
static_assert( linalg::GCACapable< LaplaceKerngen< float > > );

} // namespace terra::fe::wedge::operators::shell
