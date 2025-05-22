#pragma once

#include <cmath>
#include <stdexcept>

#include "../terra/kokkos/kokkos_wrapper.hpp"
#include "dense/vec.hpp"
#include "grid_types.hpp"

namespace terra::grid {

Grid2DDataVec< double, 3 > unit_sphere_single_shell_subdomain_coords(
    int diamond_id,
    int lateral_refinement_level,
    int num_subdomains_per_side,
    int subdomain_x,
    int subdomain_y );

inline Grid1DDataScalar< double > shell_radii_from_vector( const std::vector< double >& radii )
{
    const int                  num_shells = static_cast< int >( radii.size() );
    const int                  num_layers = num_shells - 1;
    Grid1DDataScalar< double > radii_device( "shell_radii", num_shells );
    auto                       radii_host = Kokkos::create_mirror_view( radii_device );
    for ( int i = 0; i < num_shells; ++i )
    {
        radii_host( i ) = 0.5 + ( 0.5 / num_layers ) * i;
    }
    Kokkos::deep_copy( radii_device, radii_host );
    return radii_device;
}

class ThickSphericalShellSubdomainGrid
{
  public:
    ThickSphericalShellSubdomainGrid(
        int                   lateral_refinement_level,
        int                   diamond_id,
        int                   num_subdomains_per_diamond_side,
        int                   subdomain_x,
        int                   subdomain_y,
        std::vector< double > shell_radii )
    : lateral_refinement_level_( lateral_refinement_level )
    , diamond_id_( diamond_id )
    , num_subdomains_per_diamond_side_( num_subdomains_per_diamond_side )
    , subdomain_x_( subdomain_x )
    , subdomain_y_( subdomain_y )
    {
        unit_sphere_coords_ = unit_sphere_single_shell_subdomain_coords(
            diamond_id, lateral_refinement_level, num_subdomains_per_diamond_side, subdomain_x, subdomain_y );
        shell_radii_ = shell_radii_from_vector( shell_radii );
    }

    KOKKOS_INLINE_FUNCTION Grid2DDataVec< double, 3 > unit_sphere_coords() const { return unit_sphere_coords_; }

    KOKKOS_INLINE_FUNCTION Grid1DDataScalar< double > shell_radii() const { return shell_radii_; }

    KOKKOS_INLINE_FUNCTION dense::Vec< double, 3 > coords( const int x, const int y, const int r ) const
    {
        dense::Vec< double, 3 > coords;
        coords( 0 ) = unit_sphere_coords_( x, y, 0 );
        coords( 1 ) = unit_sphere_coords_( x, y, 1 );
        coords( 2 ) = unit_sphere_coords_( x, y, 2 );
        return coords * shell_radii_( r );
    }

    KOKKOS_INLINE_FUNCTION int size_x() const { return unit_sphere_coords_.extent( 0 ); }
    KOKKOS_INLINE_FUNCTION int size_y() const { return unit_sphere_coords_.extent( 1 ); }
    KOKKOS_INLINE_FUNCTION int size_r() const { return shell_radii_.extent( 0 ); }

  private:
    int lateral_refinement_level_;
    int diamond_id_;
    int num_subdomains_per_diamond_side_;
    int subdomain_x_;
    int subdomain_y_;

    Grid2DDataVec< double, 3 > unit_sphere_coords_;
    Grid1DDataScalar< double > shell_radii_;
};

} // namespace terra::grid
