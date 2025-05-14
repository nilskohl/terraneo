#pragma once

#include <cmath>
#include <stdexcept>

#include "grid.hpp"
#include "kokkos_wrapper.hpp"
#include "types.hpp"

namespace terra {

GridData2D< double, 3 > unit_sphere_single_shell_subdomain_coords(
    int diamond_id,
    int global_refinements,
    int num_subdomains_per_side,
    int subdomain_i,
    int subdomain_j );

} // namespace terra
