
#pragma once

#include "kokkos_wrapper.hpp"
#include "types.hpp"

namespace terra {

template < typename ScalarType >
using GridDataScalar1D = Kokkos::View< ScalarType* >;

template < typename ScalarType, int VecDim >
using GridData1D = Kokkos::View< ScalarType* [VecDim] >;

template < typename ScalarType, int VecDim >
using GridData2D = Kokkos::View< ScalarType** [VecDim] >;

template < typename ScalarType, int VecDim >
using GridData3D = Kokkos::View< ScalarType*** [VecDim] >;

} // namespace terra