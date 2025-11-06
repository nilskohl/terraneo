add_subdirectory(src)

add_executable(mantlecirculation mantlecirculation.cpp)
target_link_libraries(mantlecirculation terra MPI::MPI_CXX)
target_include_directories(mantlecirculation PRIVATE ${CMAKE_SOURCE_DIR}/src)

