

#pragma once
#include "terra/util/bit_masking.hpp"

namespace terra::grid::shell {

constexpr util::MaskAndValue mask_domain_boundary()
{
    return util::MaskAndValue{ 0b10, 0b10 };
}

constexpr util::MaskAndValue mask_domain_inner()
{
    return util::MaskAndValue{ 0b10, 0b00 };
}

static_assert( mask_domain_boundary().mask == mask_domain_inner().mask );
static_assert( mask_domain_boundary().value != mask_domain_inner().value );

constexpr util::MaskAndValue mask_domain_boundary_cmb()
{
    return util::MaskAndValue{ 0b110, 0b010 };
}

constexpr util::MaskAndValue mask_domain_boundary_surface()
{
    return util::MaskAndValue{ 0b110, 0b110 };
}

static_assert( mask_domain_boundary_cmb().mask == mask_domain_boundary_surface().mask );
static_assert( mask_domain_boundary_cmb().value != mask_domain_boundary_surface().value );

} // namespace terra::grid::shell