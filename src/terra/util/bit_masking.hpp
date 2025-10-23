
#pragma once

#include <concepts>

#include "terra/kokkos/kokkos_wrapper.hpp"

namespace terra::util {

/// @concept FlagLike
/// @brief Concept for types that behave like bitmask flags.
///
/// This concept checks if a type `E` is an enum with an unsigned integral underlying type,
/// has a `NO_FLAG` value equal to 0, and supports bitwise OR (`|`) and AND (`&`) operations.
/// @tparam E The enum type to check.
template < typename E >
concept FlagLike = std::is_enum_v< E > && std::unsigned_integral< std::underlying_type_t< E > > &&
                   ( static_cast< std::underlying_type_t< E > >( E::NO_FLAG ) == 0 );

template < FlagLike E >
KOKKOS_INLINE_FUNCTION constexpr E operator|( E a, E b )
{
    using T = std::underlying_type_t< E >;
    return static_cast< E >( static_cast< T >( a ) | static_cast< T >( b ) );
}

template < FlagLike E >
KOKKOS_INLINE_FUNCTION constexpr E operator&( E a, E b )
{
    using T = std::underlying_type_t< E >;
    return static_cast< E >( static_cast< T >( a ) & static_cast< T >( b ) );
}

/// @brief Checks if a bitmask value contains a specific flag.
///
/// This function checks if the bitmask value `mask_value` has the flag `flag` set.
/// If `flag` is `E::NO_FLAG`, it checks if `mask_value` is also `E::NO_FLAG`.
/// @tparam E The enum type representing the bitmask.
/// @param mask_value The bitmask value to check.
/// @param flag The flag to check for in `mask_value`.
/// @return `true` if `mask_value` contains `flag`, otherwise `false`.
template < FlagLike E >
KOKKOS_INLINE_FUNCTION constexpr bool has_flag( E mask_value, E flag ) noexcept
{
    using U = std::underlying_type_t< E >;

    if ( flag == E::NO_FLAG )
    {
        return mask_value == E::NO_FLAG;
    }

    return static_cast< U >( mask_value & flag ) == static_cast< U >( flag );
}

} // namespace terra::util