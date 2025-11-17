
#pragma once
#include "Kokkos_Macros.hpp"

namespace terra::util {

/// @brief Computes the linearly interpolated value at a specified point using two surrounding data points.
///
/// @param pos_a coordinate of the first data point
/// @param pos_b coordinate of the first data point
/// @param val_a value at pos_a
/// @param val_b value at pos_b
/// @param pos where to evaluate
/// @param clamp if true, clamps to [val_a, val_b]
/// @return The interpolated value at the specified x-coordinate.
KOKKOS_INLINE_FUNCTION
double interpolate_linear_1D(
    const double pos_a,
    const double pos_b,
    const double val_a,
    const double val_b,
    const double pos,
    const bool   clamp )
{
    const double dx = pos_b - pos_a;

    // Degenerate interval
    if ( dx == 0.0 )
    {
        return val_a;
    }

    const double t = ( pos - pos_a ) / dx;

    if ( clamp )
    {
        if ( t <= 0.0 )
        {
            return val_a;
        }

        if ( t >= 1.0 )
        {
            return val_b;
        }
    }

    return val_a + t * ( val_b - val_a );
}

} // namespace terra::util