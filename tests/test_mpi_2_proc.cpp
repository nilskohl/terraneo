

#include <array>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const auto rank          = terra::mpi::rank();
    const auto num_processes = terra::mpi::num_processes();

    if ( num_processes != 2 )
    {
        throw std::logic_error( "This test requires 2 MPI processes." );
    }

    std::cout << "Hello from rank " << rank << " out of " << num_processes << " processes\n";

    // Init src
    if ( rank == 0 )
    {
        Kokkos::View< double* > src( "src", 10 );

        Kokkos::parallel_for(
            "fill_src", Kokkos::RangePolicy( 0, 10 ), KOKKOS_LAMBDA( const int i ) {
                src( i ) = static_cast< double >( i );
            } );

        Kokkos::fence();

        std::cout << "Sending View from rank " << rank << " to rank 1.\n";

        MPI_Send( src.data(), src.span(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD );
    }

    if ( rank == 1 )
    {
        Kokkos::View< double* > dst( "dst", 10 );

        MPI_Recv( dst.data(), dst.span(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        std::cout << "Received View from rank 0 at rank " << rank << ".\n";

        auto dst_host = Kokkos::create_mirror_view( dst );
        Kokkos::deep_copy( dst_host, dst );

        for ( int i = 0; i < dst_host.extent( 0 ); i++ )
        {
            if ( ( dst_host( i ) - i ) > 1e-16 )
            {
                throw std::logic_error( "Communication did not succeed." );
            }
            std::cout << dst_host( i ) << std::endl;
        }
    }

    return 0;
}
