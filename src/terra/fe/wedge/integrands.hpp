
#pragma once

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"

/// Relevant integrands for wedge elements.
///
/// Geometry:
///
///   xi, eta ∈ [0, 1] (lateral reference coords)
///   zeta ∈ [-1, 1]   (radial reference coords)
///
///   0 <= xi + eta <= 1
///
/// Node enumeration:
///
///   r_node_idx = r_cell_idx + 1 (outer):
///
///   5
///   |\
///   | \
///   3--4
///
///   r_node_idx = r_cell_idx (inner):
///
///   2
///   |\
///   | \
///   0--1
///
/// Enumeration of shape functions:
///
///   N_lat:
///     N_lat_0 = N_lat_3 = 1 - xi - eta
///     N_lat_1 = N_lat_4 = xi
///     N_lat_2 = N_lat_5 = eta
///
///   N_rad:
///     N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
///     N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
///
///   N_i = N_lat_i * N_rad_i
///
/// Physical coordinates:
///
///   r_1, r_2                     radii of bottom and top (r_1 < r_2)
///   p1_phy, p2_phy, p3_phy       coords of triangle on unit sphere
namespace terra::fe::wedge {

/// @brief Radial shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad( const int node_idx, const double zeta )
{
    const double N_rad[2] = { 0.5 * ( 1 - zeta ), 0.5 * ( 1 + zeta ) };
    return N_rad[node_idx / 3];
}

/// @brief Radial shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_rad( node_idx, xi_eta_zeta( 2 ) );
}

/// @brief Lateral shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat( const int node_idx, const double xi, const double eta )
{
    const double N_lat[3] = { 1.0 - xi - eta, xi, eta };
    return N_lat[node_idx % 3];
}

/// @brief Lateral shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta( 0 ), xi_eta_zeta( 1 ) );
}

/// @brief (Tensor-product) Shape function N_j = N_lat_j * N_rad_j.
KOKKOS_INLINE_FUNCTION
constexpr double shape( const int node_idx, const double xi, const double eta, const double zeta )
{
    return shape_lat( node_idx, xi, eta ) * shape_rad( node_idx, zeta );
}

/// @brief (Tensor-product) Shape function N_j = N_lat_j * N_rad_j.
KOKKOS_INLINE_FUNCTION
constexpr double shape( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta ) * shape_rad( node_idx, xi_eta_zeta );
}

/// @brief Gradient of the radial part of the shape function, in the radial direction:
///
///     ∂/∂zeta N_rad_j
///
/// This is different from the radial part of the gradient of the full shape function!
///
/// That would be
///
///     ∂/∂zeta N_j = N_lat_j * ∂/∂zeta N_rad_j
///
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_rad( const int node_idx )
{
    constexpr double grad_N_rad[2] = { -0.5, 0.5 };
    return grad_N_rad[node_idx / 3];
}

/// @brief Gradient of the lateral part of the shape function, in xi direction:
///
///     ∂/∂xi N_lat_j
///
/// This is different from the xi part (first entry) of the gradient of the full shape function!
///
/// That would be
///
///     ∂/∂xi N_j = N_rad_j * ∂/∂xi N_lat_j
///
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_xi( const int node_idx )
{
    constexpr double grad_N_lat_xi[3] = { -1.0, 1.0, 0.0 };
    return grad_N_lat_xi[node_idx % 3];
}

/// @brief Gradient of the lateral part of the shape function, in eta direction:
///
///     ∂/∂eta N_lat_j
///
/// This is different from the eta part (first entry) of the gradient of the full shape function!
///
/// That would be
///
///     ∂/∂eta N_j = N_rad_j * ∂/∂eta N_lat_j
///
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_eta( const int node_idx )
{
    constexpr double grad_N_lat_eta[3] = { -1.0, 0.0, 1.0 };
    return grad_N_lat_eta[node_idx % 3];
}

/// @brief Gradient of the full shape function:
///
///     ∇ N_j =
///     [
///         ∂/∂xi N_j
///         ∂/∂eta N_j
///         ∂/∂zeta N_j
///     ]
///     =
///     [
///         N_rad_j * ∂/∂xi N_lat_j
///         N_rad_j * ∂/∂eta N_lat_j,
///         N_lat_j * ∂/∂zeta N_rad_j
///     ]
///
KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_shape( const int node_idx, const double xi, const double eta, const double zeta )
{
    dense::Vec< double, 3 > grad_N;
    grad_N( 0 ) = grad_shape_lat_xi( node_idx ) * shape_rad( node_idx, zeta );
    grad_N( 1 ) = grad_shape_lat_eta( node_idx ) * shape_rad( node_idx, zeta );
    grad_N( 2 ) = shape_lat( node_idx, xi, eta ) * grad_shape_rad( node_idx );
    return grad_N;
}

/// @brief Overload of grad_shape( int, double, double, double ).
KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_shape( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return grad_shape( node_idx, xi_eta_zeta( 0 ), xi_eta_zeta( 1 ), xi_eta_zeta( 2 ) );
}

/// @brief Returns the coarse grid radial shape function evaluated at a point of the reference fine grid wedge.
///
/// @param coarse_node_idx       wedge node index of the coarse grid shape function ("which coarse grid shape function to
///                              evaluate") (in {0, ..., 5})
/// @param fine_radial_wedge_idx 0 for inner fine wedge, 1 for outer fine wedge
/// @param zeta_fine             coordinate in the reference fine-wedge (in [-1, 1])
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad_coarse( const int coarse_node_idx, const int fine_radial_wedge_idx, const double zeta_fine )
{
    switch ( coarse_node_idx / 3 )
    {
    case 0:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at bottom (in {0, 1, 2}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 1 at -1,
            return 0.25 * ( 3 - zeta_fine );
        case 1:
            // coarse node at bottom (in {0, 1, 2}), evaluated in top fine wedge
            // 0 at 1
            // 0.5 at -1,
            return 0.25 * ( 1 - zeta_fine );
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at top (in {3, 4, 5}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 0 at -1,
            return 0.25 * ( 1 + zeta_fine );
        case 1:
            // coarse node at top (in {3, 4, 5}), evaluated in top fine wedge
            // 1 at 1
            // 0.5 at -1
            return 0.25 * ( 3 + zeta_fine );
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

/// @brief Returns the coarse grid lateral shape function evaluated at a point of the reference fine grid wedge.
///
/// @param coarse_node_idx        wedge node index of the coarse grid shape function ("which coarse grid shape function
///                               to evaluate") (in {0, ..., 5})
/// @param fine_lateral_wedge_idx index of the lateral wedge index in a (once) refined coarse mesh, in {0, 1, 2, 3}
///                               0: bottom left triangle (orientation up)
///                               1: bottom right triangle (orientation: up)
///                               2: top triangle (orientation: up)
///                               3: center triangle (orientation: down)
/// @param xi_fine                xi-coordinate in the reference fine-wedge (in [0, 1])
/// @param eta_fine               eta-coordinate in the reference fine-wedge (in [0, 1])
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat_coarse(
    const int    coarse_node_idx,
    const int    fine_lateral_wedge_idx,
    const double xi_fine,
    const double eta_fine )
{
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5 * eta_fine - 0.5 * xi_fine + 1.0;
        case 1:
            return -0.5 * eta_fine - 0.5 * xi_fine + 0.5;
        case 2:
            return -0.5 * eta_fine - 0.5 * xi_fine + 0.5;
        case 3:
            return 0.5 * eta_fine + 0.5 * xi_fine;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5 * xi_fine;
        case 1:
            return 0.5 * xi_fine + 0.5;
        case 2:
            return 0.5 * xi_fine;
        case 3:
            return 0.5 - 0.5 * xi_fine;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5 * eta_fine;
        case 1:
            return 0.5 * eta_fine;
        case 2:
            return 0.5 * eta_fine + 0.5;
        case 3:
            return 0.5 - 0.5 * eta_fine;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
constexpr double shape_coarse(
    const int    coarse_node_idx,
    const int    fine_radial_wedge_idx,
    const int    fine_lateral_wedge_idx,
    const double xi_fine,
    const double eta_fine,
    const double zeta_fine )
{
    return shape_lat_coarse( coarse_node_idx, fine_lateral_wedge_idx, xi_fine, eta_fine ) *
           shape_rad_coarse( coarse_node_idx, fine_radial_wedge_idx, zeta_fine );
}

KOKKOS_INLINE_FUNCTION
constexpr double shape_coarse(
    const int                      coarse_node_idx,
    const int                      fine_radial_wedge_idx,
    const int                      fine_lateral_wedge_idx,
    const dense::Vec< double, 3 >& xi_eta_zeta_fine )
{
    return shape_lat_coarse( coarse_node_idx, fine_lateral_wedge_idx, xi_eta_zeta_fine( 0 ), xi_eta_zeta_fine( 1 ) ) *
           shape_rad_coarse( coarse_node_idx, fine_radial_wedge_idx, xi_eta_zeta_fine( 2 ) );
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_rad_coarse( const int coarse_node_idx, const int fine_radial_wedge_idx )
{
    switch ( coarse_node_idx / 3 )
    {
    case 0:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at bottom (in {0, 1, 2}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 1 at -1,
            return -0.25;
        case 1:
            // coarse node at bottom (in {0, 1, 2}), evaluated in top fine wedge
            // 0 at 1
            // 0.5 at -1,
            return -0.25;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at top (in {3, 4, 5}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 0 at -1,
            return 0.25;
        case 1:
            // coarse node at top (in {3, 4, 5}), evaluated in top fine wedge
            // 1 at 1
            // 0.5 at -1
            return 0.25;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_coarse_xi( const int coarse_node_idx, const int fine_lateral_wedge_idx )
{
    // derivatives in xi_fine direction
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5;
        case 1:
            return -0.5;
        case 2:
            return -0.5;
        case 3:
            return 0.5;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5;
        case 1:
            return 0.5;
        case 2:
            return 0.5;
        case 3:
            return -0.5;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0;
        case 1:
            return 0;
        case 2:
            return 0;
        case 3:
            return 0;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_coarse_eta( const int coarse_node_idx, const int fine_lateral_wedge_idx )
{
    // derivatives in eta_fine direction
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5;
        case 1:
            return -0.5;
        case 2:
            return -0.5;
        case 3:
            return 0.5;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0;
        case 1:
            return 0;
        case 2:
            return 0;
        case 3:
            return 0;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5;
        case 1:
            return 0.5;
        case 2:
            return 0.5;
        case 3:
            return -0.5;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_shape_coarse(
    const int    node_idx,
    const int    fine_radial_wedge_idx,
    const int    fine_lateral_wedge_idx,
    const double xi,
    const double eta,
    const double zeta )
{
    dense::Vec< double, 3 > grad_N;
    grad_N( 0 ) = grad_shape_lat_coarse_xi( node_idx, fine_lateral_wedge_idx ) *
                  shape_rad_coarse( node_idx, fine_radial_wedge_idx, zeta );
    grad_N( 1 ) = grad_shape_lat_coarse_eta( node_idx, fine_lateral_wedge_idx ) *
                  shape_rad_coarse( node_idx, fine_radial_wedge_idx, zeta );
    grad_N( 2 ) = shape_lat_coarse( node_idx, fine_lateral_wedge_idx, xi, eta ) *
                  grad_shape_rad_coarse( node_idx, fine_radial_wedge_idx );
    return grad_N;
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_shape_coarse(
    const int                      node_idx,
    const int                      fine_radial_wedge_idx,
    const int                      fine_lateral_wedge_idx,
    const dense::Vec< double, 3 >& xi_eta_zeta_fine )
{
    return grad_shape_coarse(
        node_idx,
        fine_radial_wedge_idx,
        fine_lateral_wedge_idx,
        xi_eta_zeta_fine( 0 ),
        xi_eta_zeta_fine( 1 ),
        xi_eta_zeta_fine( 2 ) );
}

KOKKOS_INLINE_FUNCTION constexpr double forward_map_rad( const double r_1, const double r_2, const double zeta )
{
    return r_1 + 0.5 * ( r_2 - r_1 ) * ( 1 + zeta );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > forward_map_lat(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& p3_phy,
    const double                   xi,
    const double                   eta )
{
    return ( 1 - xi - eta ) * p1_phy + xi * p2_phy + eta * p3_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_forward_map_rad( const double r_1, const double r_2 )
{
    return 0.5 * ( r_2 - r_1 );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_forward_map_lat_xi(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& /* p3_phy */ )
{
    return p2_phy - p1_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_forward_map_lat_eta(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& /* p2_phy */,
    const dense::Vec< double, 3 >& p3_phy )
{
    return p3_phy - p1_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Mat< double, 3, 3 > jac_lat(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& p3_phy,
    const double                   xi,
    const double                   eta )
{
    const auto col_0 = grad_forward_map_lat_xi( p1_phy, p2_phy, p3_phy );
    const auto col_1 = grad_forward_map_lat_eta( p1_phy, p2_phy, p3_phy );
    const auto col_2 = forward_map_lat( p1_phy, p2_phy, p3_phy, xi, eta );
    return dense::Mat< double, 3, 3 >::from_col_vecs( col_0, col_1, col_2 );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > jac_rad( const double r_1, const double r_2, const double zeta )
{
    const auto r      = forward_map_rad( r_1, r_2, zeta );
    const auto grad_r = grad_forward_map_rad( r_1, r_2 );
    return dense::Vec< double, 3 >{ r, r, grad_r };
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Mat< double, 3, 3 >
    jac( const dense::Vec< double, 3 >& p1_phy,
         const dense::Vec< double, 3 >& p2_phy,
         const dense::Vec< double, 3 >& p3_phy,
         const double                   r_1,
         const double                   r_2,
         const double                   xi,
         const double                   eta,
         const double                   zeta )
{
    return jac_lat( p1_phy, p2_phy, p3_phy, xi, eta ) *
           dense::Mat< double, 3, 3 >::diagonal_from_vec( jac_rad( r_1, r_2, zeta ) );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Mat< double, 3, 3 >
    jac( const dense::Vec< double, 3 >  p_phy[3],
         const double                   r_1,
         const double                   r_2,
         const dense::Vec< double, 3 >& xi_eta_zeta_fine )
{
    return jac(
        p_phy[0], p_phy[1], p_phy[2], r_1, r_2, xi_eta_zeta_fine( 0 ), xi_eta_zeta_fine( 1 ), xi_eta_zeta_fine( 2 ) );
}

} // namespace terra::fe::wedge