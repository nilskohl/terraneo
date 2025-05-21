add_library(terra)

target_include_directories(terra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(terra Kokkos::kokkos)

add_subdirectory(grid)
add_subdirectory(vtk)
