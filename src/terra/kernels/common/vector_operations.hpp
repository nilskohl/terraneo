#pragma once

#include "../../kokkos/kokkos_wrapper.hpp"
#include "terra/grid/grid_types.hpp"

namespace terra::kernels::common {

template < typename ScalarType >
void set_scalar( const grid::Grid3DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "set_scalar (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { x( i, j, k ) = value; } );
}

template < typename ScalarType >
void axpy(
    ScalarType                                  alpha,
    const grid::Grid3DDataScalar< ScalarType >& x,
    const grid::Grid3DDataScalar< ScalarType >& y )
{
    if ( x.extent( 0 ) != y.extent( 0 ) || x.extent( 1 ) != y.extent( 1 ) || x.extent( 2 ) != y.extent( 2 ) )
    {
        throw std::runtime_error( "axpy: x and y must have the same extent" );
    }

    Kokkos::parallel_for(
        "axpy (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { y( i, j, k ) += alpha * x( i, j, k ); } );
}

} // namespace terra::kernels::common
