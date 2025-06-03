

#include <iostream>

#include "../src/terra/kokkos/kokkos_wrapper.hpp"
#include "Kokkos_OffsetView.hpp"
#include "terra/types.hpp"

using terra::real_t;



int main( int argc, char** argv )
{
    Kokkos::ScopeGuard kokkos_guard( argc, argv );

    const int rows = 5;
    const int cols = 10;

    Kokkos::View< double** > matrix( "matrix", rows, cols );
    std::cout << matrix.span_is_contiguous() << std::endl;

    Kokkos::parallel_for(
        "InitView2D", Kokkos::MDRangePolicy( { 0, 0 }, { rows, cols } ), KOKKOS_LAMBDA( const int i, const int j ) {
            matrix( i, j ) = i + j;
        } );
    Kokkos::fence();

    auto sub_view_col = Kokkos::subview( matrix, std::pair{0, 10}, std::pair{3, 4} );
    std::cout << sub_view_col.span_is_contiguous() << std::endl;

    auto sub_view_row = Kokkos::subview( matrix, 0, Kokkos::ALL );
    std::cout << sub_view_row.span_is_contiguous() << std::endl;

    Kokkos::View< double* > view_col( "view_col", rows );
    Kokkos::deep_copy( view_col, sub_view_col );
    std::cout << view_col.span_is_contiguous() << std::endl;

    auto host_view_col = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), sub_view_col );
    for ( int i = 0; i < rows; ++i )
    {
        std::cout << host_view_col( i ) << std::endl;
    }


    Kokkos::View< double*** > test( "test", 10, 10, 10 );



}