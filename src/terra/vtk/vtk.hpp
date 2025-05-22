#pragma once

#include <fstream>
#include <iomanip> // For std::fixed, std::setprecision
#include <iostream>
#include <stdexcept> // For error handling (std::runtime_error)
#include <string>    // For filenames (std::string)
#include <vector>    // Can be useful for intermediate storage if needed

#include "../grid/grid_types.hpp"

namespace terra::vtk {

// Define VTK cell type IDs for clarity
constexpr int VTK_QUAD           = 9;
constexpr int VTK_QUADRATIC_QUAD = 23;

// Enum to specify the desired element type
enum class VtkElementType
{
    LINEAR_QUAD,
    QUADRATIC_QUAD
};

/**
 * @brief Writes a 2D grid of vertices stored in a Kokkos View to a VTK XML
 *        Unstructured Grid file (.vtu) representing a quadrilateral mesh
 *        (linear or quadratic).
 *
 * @param filename The path to the output VTK file (.vtu).
 * @param vertices A Kokkos View containing the vertex coordinates.
 *                 Assumed dimensions: (Nx, Ny, 3).
 *                 vertices(i, j, 0) = X coordinate of point (i, j)
 *                 vertices(i, j, 1) = Y coordinate of point (i, j)
 *                 vertices(i, j, 2) = Z coordinate of point (i, j)
 *                 Nx = vertices.extent(0), Ny = vertices.extent(1)
 *                 For QUADRATIC_QUAD, Nx and Ny must be odd and >= 3.
 * @param elementType Specifies whether to write linear or quadratic elements.
 */
void write_vtk_xml_quad_mesh(
    const std::string&                  filename,
    const Kokkos::View< double** [3] >& vertices,
    VtkElementType                      elementType = VtkElementType::LINEAR_QUAD )
{
    // 1. Get Dimensions and Validate
    const size_t nx         = vertices.extent( 0 );
    const size_t ny         = vertices.extent( 1 );
    const size_t num_points = nx * ny;

    size_t  num_elements       = 0;
    size_t  points_per_element = 0;
    uint8_t vtk_cell_type_id   = 0; // Use uint8_t for VTK types array

    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        if ( nx < 2 || ny < 2 )
        {
            throw std::runtime_error( "XML: Cannot create linear quads from a mesh smaller than 2x2 points." );
        }
        num_elements       = ( nx - 1 ) * ( ny - 1 );
        points_per_element = 4;
        vtk_cell_type_id   = static_cast< uint8_t >( VTK_QUAD );
    }
    else if ( elementType == VtkElementType::QUADRATIC_QUAD )
    {
        if ( nx < 3 || ny < 3 )
        {
            throw std::runtime_error( "XML: Cannot create quadratic quads from a mesh smaller than 3x3 points." );
        }
        if ( nx % 2 == 0 || ny % 2 == 0 )
        {
            throw std::runtime_error(
                "XML: For QUADRATIC_QUAD elements using the 'every second node' scheme, Nx and Ny must be odd." );
        }
        size_t num_quad_elems_x = ( nx - 1 ) / 2;
        size_t num_quad_elems_y = ( ny - 1 ) / 2;
        num_elements            = num_quad_elems_x * num_quad_elems_y;
        points_per_element      = 8;
        vtk_cell_type_id        = static_cast< uint8_t >( VTK_QUADRATIC_QUAD );
    }
    else
    {
        throw std::runtime_error( "XML: Unsupported VtkElementType." );
    }

    if ( num_elements == 0 && num_points > 0 )
    {
        // Handle cases like 2x1 or 3x1 grids where no elements can be formed
        // Write points but zero cells
        std::cout << "Warning: Input dimensions result in zero elements. Writing points only." << std::endl;
    }
    else if ( num_elements == 0 && num_points == 0 )
    {
        throw std::runtime_error( "XML: Input dimensions result in zero points and zero elements." );
    }

    // 2. Ensure data is accessible on the Host
    auto h_vertices = Kokkos::create_mirror_view( vertices );
    Kokkos::deep_copy( h_vertices, vertices );
    Kokkos::fence();

    // 3. Open the output file stream
    std::ofstream ofs( filename );
    if ( !ofs.is_open() )
    {
        throw std::runtime_error( "XML: Could not open file for writing: " + filename );
    }
    ofs << std::fixed << std::setprecision( 8 ); // Precision for coordinates

    // --- 4. Write VTK XML Header ---
    ofs << "<?xml version=\"1.0\"?>\n";
    // Use Float64 for coordinates (double), Int64 for connectivity/offsets (safer for large meshes)
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    // Piece contains the main mesh data. NumberOfPoints/Cells must be correct.
    ofs << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << num_elements << "\">\n";

    // --- 5. Write Points ---
    ofs << "      <Points>\n";
    // DataArray for coordinates: Float64, 3 components (XYZ)
    ofs << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for ( size_t i = 0; i < nx; ++i )
    {
        for ( size_t j = 0; j < ny; ++j )
        {
            ofs << "          " << h_vertices( i, j, 0 ) << " " << h_vertices( i, j, 1 ) << " " << h_vertices( i, j, 2 )
                << "\n";
        }
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // --- 6. Write Cells (Connectivity, Offsets, Types) ---
    ofs << "      <Cells>\n";

    // 6.a. Connectivity Array (flat list of point indices for all cells)
    // Use Int64 for indices to be safe with large meshes (size_t can exceed Int32)
    std::vector< int64_t > connectivity;                       // Use std::vector temporarily or write directly
    connectivity.reserve( num_elements * points_per_element ); // Pre-allocate roughly

    // 6.b. Offsets Array (index in connectivity where each cell ENDS)
    std::vector< int64_t > offsets;
    offsets.reserve( num_elements );
    int64_t current_offset = 0; // VTK offsets are cumulative

    // 6.c. Types Array (VTK cell type ID for each cell)
    std::vector< uint8_t > types;
    types.reserve( num_elements );

    // --- Populate Connectivity, Offsets, and Types ---
    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        for ( size_t i = 0; i < nx - 1; ++i )
        {
            for ( size_t j = 0; j < ny - 1; ++j )
            {
                // Calculate the 0-based indices
                int64_t p0_idx = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t p2_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 1 ) );
                int64_t p3_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }
    else
    { // QUADRATIC_QUAD
        for ( size_t i = 0; i < nx - 1; i += 2 )
        {
            for ( size_t j = 0; j < ny - 1; j += 2 )
            {
                // Calculate indices (casting to int64_t for the array)
                int64_t p0_idx  = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx  = static_cast< int64_t >( ( i + 2 ) * ny + j );
                int64_t p2_idx  = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 2 ) );
                int64_t p3_idx  = static_cast< int64_t >( i * ny + ( j + 2 ) );
                int64_t m01_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t m12_idx = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 1 ) );
                int64_t m23_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 2 ) );
                int64_t m30_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity (VTK order: corners then midsides)
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );
                connectivity.push_back( m01_idx );
                connectivity.push_back( m12_idx );
                connectivity.push_back( m23_idx );
                connectivity.push_back( m30_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }

    // --- Write the populated arrays to the file ---
    // Connectivity
    ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    ofs << "          "; // Indentation
    for ( size_t i = 0; i < connectivity.size(); ++i )
    {
        ofs << connectivity[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " ); // Newline every 12 values
    }
    ofs << "\n        </DataArray>\n"; // Add newline before closing tag if needed

    // Offsets
    ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < offsets.size(); ++i )
    {
        ofs << offsets[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    // Types
    ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < types.size(); ++i )
    {
        // Need to cast uint8_t to int for printing as number, not char
        ofs << static_cast< int >( types[i] ) << ( ( i + 1 ) % 20 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    ofs << "      </Cells>\n";

    // --- 7. Write Empty PointData and CellData (Good Practice) ---
    ofs << "      <PointData>\n";
    // Add <DataArray> tags here if you have data associated with points
    ofs << "      </PointData>\n";
    ofs << "      <CellData>\n";
    // Add <DataArray> tags here if you have data associated with cells
    ofs << "      </CellData>\n";

    // --- 8. Write VTK XML Footer ---
    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";

    // 9. Close the file
    ofs.close();
    if ( !ofs )
    {
        throw std::runtime_error( "XML: Error occurred during writing or closing file: " + filename );
    }
}

// Enum to choose the diagonal for splitting quads
enum class DiagonalSplitType
{
    FORWARD_SLASH, // Connects (i,j) with (i+1,j+1)
    BACKWARD_SLASH // Connects (i+1,j) with (i,j+1)
};

// Helper to get VTK type string from C++ type
template < typename T >
std::string get_vtk_type_string()
{
    if ( std::is_same_v< T, float > )
        return "Float32";
    if ( std::is_same_v< T, double > )
        return "Float64";
    if ( std::is_same_v< T, int > )
        return "Int32";
    if ( std::is_same_v< T, long long > )
        return "Int64";
    if ( std::is_same_v< T, int8_t > )
        return "Int8";
    if ( std::is_same_v< T, uint8_t > )
        return "UInt8";
    // Add more types as needed
    throw std::runtime_error( "Unsupported data type for VTK output" );
}

template < typename ScalarType >
void write_rectilinear_to_triangular_vtu(
    Kokkos::View< ScalarType** [3] > points_device_view,
    const std::string&               filename,
    DiagonalSplitType                split_type )
{
    auto points_host_mirror = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, points_device_view );

    const int Nx = points_host_mirror.extent( 0 ); // Number of points in 1st dim (e.g., x)
    const int Ny = points_host_mirror.extent( 1 ); // Number of points in 2nd dim (e.g., y)

    if ( Nx < 2 || Ny < 2 )
    {
        throw std::runtime_error( "Grid dimensions are too small to form cells (Nx, Ny must be >= 2)." );
    }

    const long long num_total_points   = static_cast< long long >( Nx ) * Ny;
    const long long num_quads_in_plane = static_cast< long long >( Nx - 1 ) * ( Ny - 1 );
    const long long num_cells          = num_quads_in_plane * 2; // Each quad becomes 2 triangles

    std::ofstream vtk_file( filename );
    if ( !vtk_file.is_open() )
    {
        throw std::runtime_error( "Failed to open file: " + filename );
    }

    vtk_file
        << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << num_total_points << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // --- Points ---
    // Points are written by iterating through the first index (Nx), then the second index (Ny).
    // The global point ID for point_host_mirror(i,j,*) is (i * Ny + j).
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< ScalarType >()
             << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtk_file << std::fixed << std::setprecision( 10 );
    for ( int i = 0; i < Nx; ++i )
    { // Iterate 1st dim
        for ( int j = 0; j < Ny; ++j )
        { // Iterate 2nd dim
            vtk_file << "          " << points_host_mirror( i, j, 0 ) << " " << points_host_mirror( i, j, 1 ) << " "
                     << points_host_mirror( i, j, 2 ) << "\n";
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // --- Cells (Connectivity, Offsets, Types) ---
    vtk_file << "      <Cells>\n";

    // Connectivity
    std::vector< long long > connectivity_data;
    connectivity_data.reserve( num_cells * 3 ); // 3 (indices) per triangle

    for ( int i = 0; i < Nx - 1; ++i )
    { // Iterate over quads in 1st dim
        for ( int j = 0; j < Ny - 1; ++j )
        { // Iterate over quads in 2nd dim
            // Global 0-based indices of the quad's corners
            // Point (i,j) has global ID: i * Ny + j
            long long p00 = static_cast< long long >( i ) * Ny + j;             // (i, j)
            long long p10 = static_cast< long long >( i + 1 ) * Ny + j;         // (i+1, j)
            long long p01 = static_cast< long long >( i ) * Ny + ( j + 1 );     // (i, j+1)
            long long p11 = static_cast< long long >( i + 1 ) * Ny + ( j + 1 ); // (i+1, j+1)

            if ( split_type == DiagonalSplitType::FORWARD_SLASH )
            {
                // Diagonal from (i,j) to (i+1,j+1)
                // Triangle 1: (i,j), (i+1,j), (i+1,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p11 );
                // Triangle 2: (i,j), (i+1,j+1), (i,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p11 );
                connectivity_data.push_back( p01 );
            }
            else
            { // BACKWARD_SLASH
                // Diagonal from (i+1,j) to (i,j+1)
                // Triangle 1: (i,j), (i+1,j), (i,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p01 );
                // Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p11 );
                connectivity_data.push_back( p01 );
            }
        }
    }
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"connectivity\" format=\"ascii\">\n";
    vtk_file << "          ";
    for ( size_t k = 0; k < connectivity_data.size(); ++k )
    {
        vtk_file << connectivity_data[k] << ( ( k == connectivity_data.size() - 1 ) ? "" : " " );
        if ( ( k + 1 ) % 3 == 0 && k < connectivity_data.size() - 1 )
        { // Newline for readability
            vtk_file << "\n          ";
        }
    }
    vtk_file << "\n";
    vtk_file << "        </DataArray>\n";

    // Offsets
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"offsets\" format=\"ascii\">\n";
    long long current_offset = 0;
    for ( long long c = 0; c < num_cells; ++c )
    {
        current_offset += ( 3 ); // Each triangle: 3 points
        vtk_file << "          " << current_offset << "\n";
    }
    vtk_file << "        </DataArray>\n";

    // Types (VTK_TRIANGLE = 5)
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< uint8_t >()
             << "\" Name=\"types\" format=\"ascii\">\n";
    for ( long long c = 0; c < num_cells; ++c )
    {
        vtk_file << "          5\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";

    vtk_file.close();
}

template <
    typename PointRealT,      // Type for surface coordinates and radii elements
    typename AttachedDataType // Number of components in the vector data (can be 0 if no vector data via optional)
    >
void write_surface_radial_extruded_to_wedge_vtu(
    grid::Grid2DDataVec< PointRealT, 3 > surface_points_device_view,
    grid::Grid1DDataScalar< PointRealT > radii_device_view,
    std::optional< AttachedDataType >    optional_attached_data_device_view,
    const std::string& vector_data_name, // Used only if vector_data_device_view has value & NumVecComponents > 0
    const std::string& filename,
    DiagonalSplitType  split_type )
{
    static_assert(
        std::is_same_v< AttachedDataType, grid::Grid3DDataScalar< double > > ||
        std::is_same_v< AttachedDataType, grid::Grid3DDataVec< double, 3 > > );

    const bool has_attached_data  = optional_attached_data_device_view.has_value();
    const int  is_scalar_data     = std::is_same_v< AttachedDataType, grid::Grid3DDataScalar< double > >;
    const int  num_vec_components = has_attached_data ? optional_attached_data_device_view.value().extent( 3 ) : 1;

    // --- 1. Create host mirrors and copy data ---
    auto surface_points_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, surface_points_device_view );
    auto radii_host          = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, radii_device_view );

    typename AttachedDataType::HostMirror vector_data_host;
    if ( has_attached_data )
    {
        vector_data_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, optional_attached_data_device_view.value() );
    }
    // Kokkos::fence(); // If using non-blocking deep_copy

    // --- 2. Get dimensions ---
    const int Ns       = surface_points_host.extent( 0 ); // Num points in 1st dim of surface
    const int Nt       = surface_points_host.extent( 1 ); // Num points in 2nd dim of surface
    const int Nw_radii = radii_host.extent( 0 );          // Num of radius values (defines layers of points)

    if ( Ns < 2 || Nt < 2 )
    {
        throw std::runtime_error( "Surface grid dimensions (Ns, Nt) must be >= 2 to form base triangles." );
    }
    if ( Nw_radii < 1 )
    {
        throw std::runtime_error( "Radii view must contain at least one radius value (Nw_radii >= 1)." );
    }

    // Validate vector data dimensions if provided
    if ( has_attached_data )
    { // Implies NumVecComponents > 0
        if ( vector_data_host.extent( 0 ) != Ns || vector_data_host.extent( 1 ) != Nt ||
             vector_data_host.extent( 2 ) != Nw_radii )
        {
            throw std::runtime_error(
                "VTK: Vector data dimensions (Ns, Nt, Nw_radii) do not match generated point structure." );
        }
        // extent(3) is NumVecComponents, checked by template/view type
        if ( vector_data_name.empty() )
        {
            throw std::runtime_error( "VTK: Vector data name must be provided if vector data is present." );
        }
    }

    // --- 3. Calculate total points and cells ---
    const long long num_total_points = static_cast< long long >( Ns ) * Nt * Nw_radii;
    long long       num_cells        = 0;
    if ( Nw_radii >= 2 )
    { // Need at least 2 radial layers to form wedges
        const long long num_quads_in_surface_plane = static_cast< long long >( Ns - 1 ) * ( Nt - 1 );
        const long long num_wedge_layers           = static_cast< long long >( Nw_radii - 1 );
        num_cells                                  = num_quads_in_surface_plane * 2 * num_wedge_layers;
    }

    // --- 4. Open VTK file and write header ---
    std::ofstream vtk_file( filename );
    if ( !vtk_file.is_open() )
    {
        throw std::runtime_error( "Failed to open file: " + filename );
    }
    vtk_file
        << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << num_total_points << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // --- 5. Points ---
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< PointRealT >()
             << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtk_file << std::fixed << std::setprecision( 10 );

    PointRealT base_s_pt_coords[3]; // To store coords of one surface_points_host(s,t,*)
    for ( int s_idx = 0; s_idx < Ns; ++s_idx )
    {
        for ( int t_idx = 0; t_idx < Nt; ++t_idx )
        {
            // Cache the base surface point coordinates
            base_s_pt_coords[0] = surface_points_host( s_idx, t_idx, 0 );
            base_s_pt_coords[1] = surface_points_host( s_idx, t_idx, 1 );
            base_s_pt_coords[2] = surface_points_host( s_idx, t_idx, 2 );
            for ( int w_rad_idx = 0; w_rad_idx < Nw_radii; ++w_rad_idx )
            {
                PointRealT current_radius_val = radii_host( w_rad_idx );
                vtk_file << "          " << base_s_pt_coords[0] * current_radius_val << " "
                         << base_s_pt_coords[1] * current_radius_val << " " << base_s_pt_coords[2] * current_radius_val
                         << "\n";
            }
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // --- 6. Cells (Connectivity, Offsets, Types) ---
    vtk_file << "      <Cells>\n";
    std::vector< long long > connectivity_data;
    if ( num_cells > 0 )
    {
        connectivity_data.reserve( num_cells * 6 ); //  6 indices per wedge

        auto pt_gid = [&]( int s_surf_idx, int t_surf_idx, int w_rad_layer_idx ) {
            return static_cast< long long >( s_surf_idx ) * ( Nt * Nw_radii ) +
                   static_cast< long long >( t_surf_idx ) * Nw_radii + static_cast< long long >( w_rad_layer_idx );
        };

        for ( int s = 0; s < Ns - 1; ++s )
        { // Iterate over quads in s-dim of surface
            for ( int t = 0; t < Nt - 1; ++t )
            { // Iterate over quads in t-dim of surface
                for ( int w_rad_layer_idx = 0; w_rad_layer_idx < Nw_radii - 1; ++w_rad_layer_idx )
                { // Iterate over wedge layers
                    int base_rad_idx = w_rad_layer_idx;
                    int top_rad_idx  = w_rad_layer_idx + 1;

                    long long p00_base = pt_gid( s, t, base_rad_idx );
                    long long p10_base = pt_gid( s + 1, t, base_rad_idx );
                    long long p01_base = pt_gid( s, t + 1, base_rad_idx );
                    long long p11_base = pt_gid( s + 1, t + 1, base_rad_idx );

                    long long p00_top = pt_gid( s, t, top_rad_idx );
                    long long p10_top = pt_gid( s + 1, t, top_rad_idx );
                    long long p01_top = pt_gid( s, t + 1, top_rad_idx );
                    long long p11_top = pt_gid( s + 1, t + 1, top_rad_idx );

                    if ( split_type == DiagonalSplitType::FORWARD_SLASH )
                    {
                        // Wedge 1 from Tri 1: (p00, p10, p11)_base -> (p00, p10, p11)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p11_top );
                        // Wedge 2 from Tri 2: (p00, p11, p01)_base -> (p00, p11, p01)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p11_top );
                        connectivity_data.push_back( p01_top );
                    }
                    else
                    { // BACKWARD_SLASH
                        // Wedge 1 from Tri 1: (p00, p10, p01)_base -> (p00, p10, p01)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p01_top );
                        // Wedge 2 from Tri 2: (p10, p11, p01)_base -> (p10, p11, p01)_top
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p11_top );
                        connectivity_data.push_back( p01_top );
                    }
                }
            }
        }
    }
    // Write Connectivity
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"connectivity\" format=\"ascii\">\n";
    if ( !connectivity_data.empty() )
    {
        vtk_file << "          ";
        for ( size_t k = 0; k < connectivity_data.size(); ++k )
        {
            vtk_file << connectivity_data[k] << ( ( k == connectivity_data.size() - 1 ) ? "" : " " );
            if ( ( k + 1 ) % 6 == 0 && k < connectivity_data.size() - 1 )
                vtk_file << "\n          ";
        }
        vtk_file << "\n";
    }
    vtk_file << "        </DataArray>\n";

    // Write Offsets
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"offsets\" format=\"ascii\">\n";
    if ( num_cells > 0 )
    {
        long long current_offset = 0;
        for ( long long c = 0; c < num_cells; ++c )
        {
            current_offset += 6; // Each wedge: 6 points
            vtk_file << "          " << current_offset << "\n";
        }
    }
    vtk_file << "        </DataArray>\n";

    // Write Types
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< uint8_t >()
             << "\" Name=\"types\" format=\"ascii\">\n";
    if ( num_cells > 0 )
    {
        for ( long long c = 0; c < num_cells; ++c )
        {
            vtk_file << "          13\n"; // VTK_WEDGE cell type
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    // --- 7. PointData (if vector data is provided) ---
    if ( has_attached_data )
    {
        std::string point_data_attributes_str;
        if ( is_scalar_data )
        {
            point_data_attributes_str = " Scalars=\"" + vector_data_name + "\"";
        }
        else
        {
            point_data_attributes_str = " Vectors=\"" + vector_data_name + "\"";
        }

        vtk_file << "      <PointData" << point_data_attributes_str << ">\n";
        vtk_file << "        <DataArray type=\"" << get_vtk_type_string< double >() << "\" Name=\"" << vector_data_name
                 << "\" NumberOfComponents=\"" << num_vec_components << "\" format=\"ascii\">\n";
        vtk_file << std::fixed << std::setprecision( 10 );

        // Iterate in the SAME order as points were written: s_idx, t_idx, w_rad_idx
        for ( int s_idx = 0; s_idx < Ns; ++s_idx )
        {
            for ( int t_idx = 0; t_idx < Nt; ++t_idx )
            {
                for ( int w_rad_idx = 0; w_rad_idx < Nw_radii; ++w_rad_idx )
                {
                    vtk_file << "          ";
                    if constexpr ( is_scalar_data )
                    {
                        vtk_file << vector_data_host( s_idx, t_idx, w_rad_idx ) << " ";
                    }
                    else
                    {
                        for ( size_t comp = 0; comp < num_vec_components; ++comp )
                        {
                            vtk_file << vector_data_host( s_idx, t_idx, w_rad_idx, comp )
                                     << ( comp == num_vec_components - 1 ? "" : " " );
                        }
                    }

                    vtk_file << "\n";
                }
            }
        }
        vtk_file << "        </DataArray>\n";
        vtk_file << "      </PointData>\n";
    }
    else
    {                                      // No vector data or NumVecComponents is 0
        vtk_file << "      <PointData>\n"; // Empty PointData section
        vtk_file << "      </PointData>\n";
    }

    // --- Footer ---
    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";
    vtk_file.close();
}
} // namespace terra::vtk