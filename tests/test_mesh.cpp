#include <fstream> // For VTK output example
#include <iomanip>
#include <iostream>
#include <optional>

#include "terra/point_3d.hpp"
#include "terra/spherical_shell_mesh.hpp"
#include "terra/vtk.hpp"
#include "vtk.hpp"

int main( int argc, char** argv )
{
    Kokkos::initialize( argc, argv );
    {
        const int global_refinements    = 2;
        const int n_subdomains_per_side = 2;
        const int diamond_id            = 0;

        const int num_shells = 4;
        const int num_layers = num_shells - 1;

        auto mesh_view_full =
            terra::unit_sphere_single_shell_subdomain_coords( diamond_id, global_refinements, 1, 0, 0 );
        auto mesh_view_partial = terra::unit_sphere_single_shell_subdomain_coords(
            diamond_id, global_refinements, n_subdomains_per_side, 0, 0 );
        auto mesh_view_partial_2 = terra::unit_sphere_single_shell_subdomain_coords(
            diamond_id, global_refinements, n_subdomains_per_side, 1, 0 );

        terra::write_rectilinear_to_triangular_vtu(
            mesh_view_full, "mesh_full.vtu", terra::DiagonalSplitType::BACKWARD_SLASH );
        terra::write_rectilinear_to_triangular_vtu(
            mesh_view_partial, "mesh_view_partial.vtu", terra::DiagonalSplitType::BACKWARD_SLASH );
        terra::write_rectilinear_to_triangular_vtu(
            mesh_view_partial_2, "mesh_view_partial_2.vtu", terra::DiagonalSplitType::BACKWARD_SLASH );

        terra::GridDataScalar1D< double > radii( "radii", num_shells );
        auto                              radii_host = Kokkos::create_mirror_view( radii );
        for ( int i = 0; i < num_shells; ++i )
        {
            radii_host( i ) = 0.5 + ( 0.5 / num_layers ) * i;
        }
        Kokkos::deep_copy( radii_host, radii );

        terra::write_surface_radial_extruded_to_wedge_vtu(
            mesh_view_full,
            radii,
            std::optional< terra::GridData3D< double, 3 > >(),
            "data_name",
            "shell.vtu",
            terra::DiagonalSplitType::BACKWARD_SLASH );
    }
    Kokkos::finalize();

    return 0;
}