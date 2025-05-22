#pragma once

#include "../terra/kokkos/kokkos_wrapper.hpp"

namespace terra::dense {

template < typename T, int N >
struct Vec
{
    T data[N] = {};

    Vec() = default;

    KOKKOS_INLINE_FUNCTION
    T& operator()( int i ) { return data[i]; }

    KOKKOS_INLINE_FUNCTION
    const T& operator()( int i ) const { return data[i]; }

    void fill( const T value )
    {
        for ( int i = 0; i < N; ++i )
        {
            data[i] = value;
        }
    }

    KOKKOS_INLINE_FUNCTION
    T dot( const Vec& other ) const
    {
        T sum = 0;
        for ( int i = 0; i < N; ++i )
            sum += data[i] * other( i );
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    Vec operator+( const Vec& rhs ) const
    {
        Vec out;
        for ( int i = 0; i < N; ++i )
            out( i ) = data[i] + rhs( i );
        return out;
    }

    KOKKOS_INLINE_FUNCTION
    Vec operator*( T scalar ) const
    {
        Vec out;
        for ( int i = 0; i < N; ++i )
            out( i ) = data[i] * scalar;
        return out;
    }

    KOKKOS_INLINE_FUNCTION
    T norm() const
    {
        T sum = 0;
        for ( int i = 0; i < N; ++i )
        {
            sum += data[i] * data[i];
        }
        return Kokkos::sqrt( sum );
    }
};

template < typename T, int N >
std::ostream& operator<<( std::ostream& os, const Vec< T, N >& v )
{
    for ( int i = 0; i < N; ++i )
    {
        os << v( i ) << '\n';
    }
    return os;
}

} // namespace terra::dense
