#pragma once

#include <ranges>
#include <variant>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/communication/buffer_copy_kernels.hpp"
#include "util/timer.hpp"

using terra::grid::shell::SubdomainNeighborhood;

namespace terra::communication::shell {

constexpr int MPI_TAG_BOUNDARY_DATA = 100;

/// @brief Send and receive buffers for all process-local subdomain boundaries.
///
/// Allocates views for all boundaries of local subdomains. Those are the nodes that overlap with values from
/// neighboring subdomains.
///
/// One buffer per local boundary + neighbor is allocated. So, for instance, for an edge shared with several
/// neighbors, just as many buffers as neighbors are allocated. This facilitates the receiving step since all
/// neighbors that a subdomain receives data from can send their data simultaneously.
///
/// Can be reused after communication (send + recv) has been completed to avoid unnecessary reallocation.
template < typename ScalarType, int VecDim = 1 >
class SubdomainNeighborhoodSendRecvBuffer
{
  public:
    /// @brief Constructs a SubdomainNeighborhoodSendRecvBuffer for the passed distributed domain object.
    explicit SubdomainNeighborhoodSendRecvBuffer( const grid::shell::DistributedDomain& domain )
    {
        setup_buffers( domain );
    }

    /// @brief Const reference to the view that is a buffer for a vertex of a subdomain.
    const grid::Grid0DDataVec< ScalarType, VecDim >& buffer_vertex(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryVertex        local_boundary_vertex,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryVertex        neighbor_boundary_vertex ) const
    {
        return buffers_vertex_.at(
            { local_subdomain, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex } );
    }

    /// @brief Const reference to the view that is a buffer for an edge of a subdomain.
    const grid::Grid1DDataVec< ScalarType, VecDim >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryEdge          neighbor_boundary_edge ) const
    {
        return buffers_edge_.at( { local_subdomain, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge } );
    }

    /// @brief Const reference to the view that is a buffer for a face of a subdomain.
    const grid::Grid2DDataVec< ScalarType, VecDim >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryFace          neighbor_boundary_face ) const
    {
        return buffers_face_.at( { local_subdomain, local_boundary_face, neighbor_subdomain, neighbor_boundary_face } );
    }

    /// @brief Mutable reference to the view that is a buffer for a vertex of a subdomain.
    grid::Grid0DDataVec< ScalarType, VecDim >& buffer_vertex(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryVertex        local_boundary_vertex,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryVertex        neighbor_boundary_vertex )
    {
        return buffers_vertex_.at(
            { local_subdomain, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex } );
    }

    /// @brief Mutable reference to the view that is a buffer for an edge of a subdomain.
    grid::Grid1DDataVec< ScalarType, VecDim >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryEdge          neighbor_boundary_edge )
    {
        return buffers_edge_.at( { local_subdomain, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge } );
    }

    /// @brief Mutable reference to the view that is a buffer for a face of a subdomain.
    grid::Grid2DDataVec< ScalarType, VecDim >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryFace          neighbor_boundary_face )
    {
        return buffers_face_.at( { local_subdomain, local_boundary_face, neighbor_subdomain, neighbor_boundary_face } );
    }

  private:
    /// @brief Helper called in the ctor that allocates the buffers.
    void setup_buffers( const grid::shell::DistributedDomain& domain )
    {
        for ( const auto& [subdomain_info, data] : domain.subdomains() )
        {
            const auto& [local_subdomain_idx, neighborhood] = data;

            for ( const auto& [local_boundary_vertex, neighbor] : neighborhood.neighborhood_vertex() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_vertex, mpi_rank] : neighbor )
                {
                    buffers_vertex_[{
                        subdomain_info, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex }] =
                        grid::Grid0DDataVec< ScalarType, VecDim >( "recv_buffer" );
                }
            }

            for ( const auto& [local_boundary_edge, neighbor] : neighborhood.neighborhood_edge() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_edge, _, mpi_rank] : neighbor )
                {
                    const int buffer_size = grid::is_edge_boundary_radial( local_boundary_edge ) ?
                                                domain.domain_info().subdomain_num_nodes_radially() :
                                                domain.domain_info().subdomain_num_nodes_per_side_laterally();

                    buffers_edge_[{ subdomain_info, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge }] =
                        grid::Grid1DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size );
                }
            }

            for ( const auto& [local_boundary_face, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_subdomain, neighbor_boundary_face, _, mpi_rank] = neighbor;

                const int buffer_size_i = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int buffer_size_j = grid::is_face_boundary_normal_to_radial_direction( local_boundary_face ) ?
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                              domain.domain_info().subdomain_num_nodes_radially();

                buffers_face_[{ subdomain_info, local_boundary_face, neighbor_subdomain, neighbor_boundary_face }] =
                    grid::Grid2DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size_i, buffer_size_j );
            }
        }
    }

    /// Key ordering: local subdomain, local boundary, neighbor subdomain, neighbor-local boundary
    std::map<
        std::
            tuple< grid::shell::SubdomainInfo, grid::BoundaryVertex, grid::shell::SubdomainInfo, grid::BoundaryVertex >,
        grid::Grid0DDataVec< ScalarType, VecDim > >
        buffers_vertex_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryEdge, grid::shell::SubdomainInfo, grid::BoundaryEdge >,
        grid::Grid1DDataVec< ScalarType, VecDim > >
        buffers_edge_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryFace, grid::shell::SubdomainInfo, grid::BoundaryFace >,
        grid::Grid2DDataVec< ScalarType, VecDim > >
        buffers_face_;
};

namespace detail {

// Build an unmanaged view with the *same* data_type/layout/device as a Grid*DDataVec,
// pointing into a raw pointer slice.
// This lets us reuse copy_to_buffer(...) and still pack into a contiguous rank buffer.

template < class GridViewT >
auto make_unmanaged_like( typename GridViewT::value_type* ptr, int n0 = 0, int n1 = 0, int n2 = 0 )
{
    using data_type    = typename GridViewT::data_type;
    using array_layout = typename GridViewT::array_layout;
    using device_type  = typename GridViewT::device_type;

    using unmanaged_view =
        Kokkos::View< data_type, array_layout, device_type, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

    if constexpr ( GridViewT::rank == 1 )
    {
        return unmanaged_view( ptr, n0 );
    }
    else if constexpr ( GridViewT::rank == 2 )
    {
        return unmanaged_view( ptr, n0, n1 );
    }
    else if constexpr ( GridViewT::rank == 3 )
    {
        return unmanaged_view( ptr, n0, n1, n2 );
    }
    else
    {
        static_assert( GridViewT::rank >= 1 && GridViewT::rank <= 3, "Unsupported rank for unmanaged-like helper." );
    }
}

} // namespace detail

/// @brief Packs, sends and recvs local subdomain boundaries using two sets of buffers.
///
/// Communication works like this:
/// - data is packed from the boundaries of the grid data structure into send buffers
/// - the send buffers are sent via MPI
/// - the data is received in receive buffers
/// - the receive buffers are unpacked into the grid data structure (and the data is potentially rotated if necessary)
///
/// @note Must be complemented with `unpack_and_reduce_local_subdomain_boundaries()` to complete communication.
///       This function waits until all recv buffers are filled - but does not unpack.
///
/// Performs "additive" communication.
///
/// IMPLEMENTED OPTIMIZATION (Point 1):
///   **Aggregate messages per neighbor rank**:
///   We pack all boundary pieces destined for the same neighbor rank into one contiguous send buffer,
///   and post exactly one Isend/Irecv per neighbor rank (instead of per boundary piece).
///
/// The existing boundary_recv_buffers are still filled (scatter) so that the existing unpack routine stays unchanged.
///
template < typename GridDataType >
void pack_send_and_recv_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    [[maybe_unused]] SubdomainNeighborhoodSendRecvBuffer<
        typename GridDataType::value_type,
        grid::grid_data_vec_dim< GridDataType >() >& boundary_send_buffers,
    SubdomainNeighborhoodSendRecvBuffer<
        typename GridDataType::value_type,
        grid::grid_data_vec_dim< GridDataType >() >& boundary_recv_buffers )
{
    constexpr bool enable_local_comm = true;

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    using ScalarType                = typename GridDataType::value_type;
    static constexpr int VecDim     = grid::grid_data_vec_dim< GridDataType >();
    using memory_space              = typename GridDataType::memory_space;
    using rank_buffer_view          = Kokkos::View< ScalarType*, memory_space >;

    struct SendRecvPair
    {
        int                        boundary_type = -1; // 0 vertex, 1 edge, 2 face
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_boundary;
        int                        local_subdomain_id;

        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_boundary;
    };

    std::vector< SendRecvPair > send_recv_pairs;
    send_recv_pairs.reserve( 1024 );
  
    {
    util::Timer          timer_kernel( "setup_send_recv_pairs" );
      
    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

        for ( const auto& [local_vertex_boundary, neighbors] : neighborhood.neighborhood_vertex() )
        {
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                send_recv_pairs.push_back( SendRecvPair{
                    .boundary_type               = 0,
                    .local_rank                  = mpi::rank(),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_vertex_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) } );
            }
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, _, neighbor_rank] = neighbor;

                send_recv_pairs.push_back( SendRecvPair{
                    .boundary_type               = 1,
                    .local_rank                  = mpi::rank(),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_edge_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) } );
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, _, neighbor_rank] = neighbor;

            send_recv_pairs.push_back( SendRecvPair{
                .boundary_type               = 2,
                .local_rank                  = mpi::rank(),
                .local_subdomain             = local_subdomain_info,
                .local_subdomain_boundary    = static_cast< int >( local_face_boundary ),
                .local_subdomain_id          = local_subdomain_id,
                .neighbor_rank               = neighbor_rank,
                .neighbor_subdomain          = neighbor_subdomain_info,
                .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) } );
        }
    }
}

    // -------------------------------------------------------------------------
    // Build per-neighbor-rank receive layout (using the same ordering as before)
    // -------------------------------------------------------------------------

    auto recv_pairs = send_recv_pairs;

    std::sort( recv_pairs.begin(), recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        // Must match original Irecv ordering to be compatible with sender-side ordering.
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        if ( a.neighbor_subdomain != b.neighbor_subdomain )
            return a.neighbor_subdomain < b.neighbor_subdomain;
        if ( a.neighbor_subdomain_boundary != b.neighbor_subdomain_boundary )
            return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
        if ( a.local_subdomain != b.local_subdomain )
            return a.local_subdomain < b.local_subdomain;
        return a.local_subdomain_boundary < b.local_subdomain_boundary;
    } );

    auto piece_num_scalars = [&]( const SendRecvPair& p ) -> int {
        if ( p.boundary_type == 0 )
        {
            return VecDim;
        }
        else if ( p.boundary_type == 1 )
        {
            const auto local_edge_boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
            const int  n_nodes             = grid::is_edge_boundary_radial( local_edge_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_radially() :
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally();
            return n_nodes * VecDim;
        }
        else if ( p.boundary_type == 2 )
        {
            const auto local_face_boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
            const int  ni                  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
            const int  nj                  = grid::is_face_boundary_normal_to_radial_direction( local_face_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                                 domain.domain_info().subdomain_num_nodes_radially();
            return ni * nj * VecDim;
        }
        Kokkos::abort( "Unknown boundary type" );
        return 0;
    };

    // Per neighbor rank: contiguous recv buffer + list of (pair, offset, size)
    struct ChunkInfo
    {
        SendRecvPair pair;
        int         offset = 0; // in scalars
        int         size   = 0; // in scalars
    };

    std::map< mpi::MPIRank, std::vector< ChunkInfo > > recv_chunks_by_rank;
    std::map< mpi::MPIRank, int >                       recv_total_by_rank;

    {
    util::Timer          timer_kernel( "recv_chunks_by_rank" );
    for ( const auto& p : recv_pairs )
    {
        if ( enable_local_comm && p.local_rank == p.neighbor_rank )
            continue;

        const int sz = piece_num_scalars( p );
        auto&     chunks = recv_chunks_by_rank[p.neighbor_rank];

        const int off = recv_total_by_rank[p.neighbor_rank];
        recv_total_by_rank[p.neighbor_rank] += sz;

        chunks.push_back( ChunkInfo{ .pair = p, .offset = off, .size = sz } );
    }
}

    std::map< mpi::MPIRank, rank_buffer_view > recv_rank_buffers;
    std::vector< MPI_Request >                 data_recv_requests;
    data_recv_requests.reserve( recv_chunks_by_rank.size() );
  {
    util::Timer          timer_kernel( "recv_rank_buffers" );
  
    for ( const auto& [rank, total_sz] : recv_total_by_rank )
    {
        if ( total_sz <= 0 )
            continue;

        recv_rank_buffers[rank] = rank_buffer_view( "rank_recv_buffer", total_sz );

        MPI_Request req;
        MPI_Irecv(
            recv_rank_buffers[rank].data(),
            total_sz,
            mpi::mpi_datatype< ScalarType >(),
            rank,
            MPI_TAG_BOUNDARY_DATA,
            MPI_COMM_WORLD,
            &req );
        data_recv_requests.push_back( req );
    }
}

    // -------------------------------------------------------------------------
    // Build per-neighbor-rank send layout (using the same ordering as before)
    // -------------------------------------------------------------------------

    auto send_pairs = send_recv_pairs;
  {
    util::Timer          timer_kernel( "sort" );
 
    std::sort( send_pairs.begin(), send_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        // Must match original Isend ordering.
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        if ( a.local_subdomain != b.local_subdomain )
            return a.local_subdomain < b.local_subdomain;
        if ( a.local_subdomain_boundary != b.local_subdomain_boundary )
            return a.local_subdomain_boundary < b.local_subdomain_boundary;
        if ( a.neighbor_subdomain != b.neighbor_subdomain )
            return a.neighbor_subdomain < b.neighbor_subdomain;
        return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
    } );
}

    std::map< mpi::MPIRank, std::vector< ChunkInfo > > send_chunks_by_rank;
    std::map< mpi::MPIRank, int >                       send_total_by_rank;
      std::map< mpi::MPIRank, rank_buffer_view > send_rank_buffers;

  {
    util::Timer          timer_kernel( "send_chunks_by_rank" );
 
    for ( const auto& p : send_pairs )
    {
        if ( enable_local_comm && p.local_rank == p.neighbor_rank )
            continue;

        const int sz = piece_num_scalars( p );
        auto&     chunks = send_chunks_by_rank[p.neighbor_rank];

        const int off = send_total_by_rank[p.neighbor_rank];
        send_total_by_rank[p.neighbor_rank] += sz;

        chunks.push_back( ChunkInfo{ .pair = p, .offset = off, .size = sz } );
    }

  
    for ( const auto& [rank, total_sz] : send_total_by_rank )
    {
        if ( total_sz <= 0 )
            continue;

        send_rank_buffers[rank] = rank_buffer_view( "rank_send_buffer", total_sz );
    }
}

    // -------------------------------------------------------------------------
    // Local communication path stays as before (direct copy into recv buffers)
    // -------------------------------------------------------------------------
     {
    util::Timer          timer_kernel( "local_comm" );
  for ( const auto& p : send_pairs )
    {
        const auto local_comm = enable_local_comm && p.local_rank == p.neighbor_rank;
        if ( !local_comm )
            continue;

        // For local comm: copy directly into the existing per-boundary recv buffers (as before).
        if ( !domain.subdomains().contains( p.neighbor_subdomain ) )
            Kokkos::abort( "Subdomain not found locally - but it should be there..." );

        const auto local_subdomain_id_of_neighboring_subdomain =
            std::get< 0 >( domain.subdomains().at( p.neighbor_subdomain ) );

        if ( p.boundary_type == 0 )
        {
            auto& recv_buf = boundary_recv_buffers.buffer_vertex(
                p.local_subdomain,
                static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ),
                p.neighbor_subdomain,
                static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary ) );

            copy_to_buffer<VecDim>(
                recv_buf,
                data,
                local_subdomain_id_of_neighboring_subdomain,
                static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary ) );
        }
        else if ( p.boundary_type == 1 )
        {
            auto& recv_buf = boundary_recv_buffers.buffer_edge(
                p.local_subdomain,
                static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary ),
                p.neighbor_subdomain,
                static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary ) );

            copy_to_buffer<VecDim>(
                recv_buf,
                data,
                local_subdomain_id_of_neighboring_subdomain,
                static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary ) );
        }
        else if ( p.boundary_type == 2 )
        {
            auto& recv_buf = boundary_recv_buffers.buffer_face(
                p.local_subdomain,
                static_cast< grid::BoundaryFace >( p.local_subdomain_boundary ),
                p.neighbor_subdomain,
                static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary ) );

            copy_to_buffer<VecDim>(
                recv_buf,
                data,
                local_subdomain_id_of_neighboring_subdomain,
                static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary ) );
        }
        else
        {
            Kokkos::abort( "Unknown boundary type" );
        }
    }
}

    // -------------------------------------------------------------------------
    // Pack all remote sends into the per-rank contiguous send buffers
    // -------------------------------------------------------------------------     
    {
    util::Timer          timer_kernel( "packing" );

    for ( const auto& [rank, chunks] : send_chunks_by_rank )
    {
        auto& rank_buf = send_rank_buffers.at( rank );

        for ( const auto& ch : chunks )
        {
            const auto& p = ch.pair;
            ScalarType* base_ptr = rank_buf.data() + ch.offset;

            if ( p.boundary_type == 0 )
            {
                using BufT = grid::Grid0DDataVec< ScalarType, VecDim >;
                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr );

                copy_to_buffer<VecDim>(
                    unmanaged,
                    data,
                    p.local_subdomain_id,
                    static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ) );
            }
            else if ( p.boundary_type == 1 )
            {
                using BufT = grid::Grid1DDataVec< ScalarType, VecDim >;
                const auto local_edge_boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                const int  n_nodes             = grid::is_edge_boundary_radial( local_edge_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_radially() :
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally();
                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, n_nodes );

                copy_to_buffer<VecDim>( unmanaged, data, p.local_subdomain_id, local_edge_boundary );
            }
            else if ( p.boundary_type == 2 )
            {
                using BufT = grid::Grid2DDataVec< ScalarType, VecDim >;
                const auto local_face_boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                const int  ni                  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int  nj                  = grid::is_face_boundary_normal_to_radial_direction( local_face_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                                 domain.domain_info().subdomain_num_nodes_radially();
                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, ni, nj );

                copy_to_buffer<VecDim>( unmanaged, data, p.local_subdomain_id, local_face_boundary );
            }
            else
            {
                Kokkos::abort( "Unknown boundary type" );
            }
        }
    }

    // Ensure send buffers are ready before MPI reads them.
    Kokkos::fence( "pack_rank_send_buffers" );
}

    // -------------------------------------------------------------------------
    // Post one Isend per neighbor rank (remote only)
    // -------------------------------------------------------------------------
    std::vector< MPI_Request > data_send_requests;
    data_send_requests.reserve( send_rank_buffers.size() );
 {
    util::Timer          timer_kernel( "posting sends" );

    for ( const auto& [rank, buf] : send_rank_buffers )
    {
        const int total_sz = static_cast< int >( buf.extent( 0 ) );
        if ( total_sz <= 0 )
            continue;

        MPI_Request req;
        MPI_Isend(
            buf.data(),
            total_sz,
            mpi::mpi_datatype< ScalarType >(),
            rank,
            MPI_TAG_BOUNDARY_DATA,
            MPI_COMM_WORLD,
            &req );
        data_send_requests.push_back( req );
    }
}

    // Wait for sends + receives (same semantics as before)
    if ( !data_send_requests.empty() )
        MPI_Waitall( data_send_requests.size(), data_send_requests.data(), MPI_STATUSES_IGNORE );
    if ( !data_recv_requests.empty() )
        MPI_Waitall( data_recv_requests.size(), data_recv_requests.data(), MPI_STATUSES_IGNORE );

    // -------------------------------------------------------------------------
    // Scatter aggregated recv buffers into the existing per-boundary recv buffers
    // (so unpack_and_reduce_local_subdomain_boundaries() can remain unchanged)
    // -------------------------------------------------------------------------
     {
    util::Timer          timer_kernel( "scatter buffers" );

    for ( const auto& [rank, chunks] : recv_chunks_by_rank )
    {
        auto& rank_buf = recv_rank_buffers.at( rank );

        for ( const auto& ch : chunks )
        {
            const auto& p = ch.pair;
            ScalarType* base_ptr = rank_buf.data() + ch.offset;

            if ( p.boundary_type == 0 )
            {
                using BufT = grid::Grid0DDataVec< ScalarType, VecDim >;
                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr );

                auto& recv_buf = boundary_recv_buffers.buffer_vertex(
                    p.local_subdomain,
                    static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary ) );

                Kokkos::deep_copy( recv_buf, unmanaged );
            }
            else if ( p.boundary_type == 1 )
            {
                using BufT = grid::Grid1DDataVec< ScalarType, VecDim >;
                const auto local_edge_boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                const int  n_nodes             = grid::is_edge_boundary_radial( local_edge_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_radially() :
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally();

                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, n_nodes );

                auto& recv_buf = boundary_recv_buffers.buffer_edge(
                    p.local_subdomain,
                    static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary ) );

                Kokkos::deep_copy( recv_buf, unmanaged );
            }
            else if ( p.boundary_type == 2 )
            {
                using BufT = grid::Grid2DDataVec< ScalarType, VecDim >;
                const auto local_face_boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                const int  ni                  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int  nj                  = grid::is_face_boundary_normal_to_radial_direction( local_face_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                                 domain.domain_info().subdomain_num_nodes_radially();

                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, ni, nj );

                auto& recv_buf = boundary_recv_buffers.buffer_face(
                    p.local_subdomain,
                    static_cast< grid::BoundaryFace >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary ) );

                Kokkos::deep_copy( recv_buf, unmanaged );
            }
            else
            {
                Kokkos::abort( "Unknown boundary type" );
            }
        }
    }

    // Make sure recv buffers are populated before caller proceeds to unpack.
    Kokkos::fence( "scatter_rank_recv_buffers" );
}
}

/// @brief Unpacks and reduces local subdomain boundaries.
///
/// The recv buffers must be the same instances as used during sending in `pack_send_and_recv_local_subdomain_boundaries()`.
///
/// See `pack_send_and_recv_local_subdomain_boundaries()` for more details on how the communication works.
template < typename GridDataType >
void unpack_and_reduce_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
                           boundary_recv_buffers,
    CommunicationReduction reduction = CommunicationReduction::SUM )
{
    util::Timer          timer_kernel( "unpacking and reduce" );

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

        for ( const auto& [local_vertex_boundary, neighbors] : neighborhood.neighborhood_vertex() )
        {
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                auto recv_buffer = boundary_recv_buffers.buffer_vertex(
                    local_subdomain_info, local_vertex_boundary, neighbor_subdomain_info, neighbor_local_boundary );

                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, local_subdomain_id, local_vertex_boundary, reduction );
            }
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, boundary_direction, neighbor_rank] =
                    neighbor;

                auto recv_buffer = boundary_recv_buffers.buffer_edge(
                    local_subdomain_info, local_edge_boundary, neighbor_subdomain_info, neighbor_local_boundary );

                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, local_subdomain_id, local_edge_boundary, boundary_direction, reduction );
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, boundary_directions, neighbor_rank] =
                neighbor;

            auto recv_buffer = boundary_recv_buffers.buffer_face(
                local_subdomain_info, local_face_boundary, neighbor_subdomain_info, neighbor_local_boundary );

            copy_from_buffer_rotate_and_reduce(
                recv_buffer, data, local_subdomain_id, local_face_boundary, boundary_directions, reduction );
        }
    }

    Kokkos::fence();
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// @note THIS MAY COME WITH A PERFORMANCE PENALTY.
///       This function (re-)allocates send and receive buffers for each call, which could be inefficient.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain& domain,
    grid::Grid4DDataScalar< ScalarType >& grid,
    const CommunicationReduction          reduction = CommunicationReduction::SUM )
{
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buffers( domain );
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buffers( domain );

    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// Send and receive buffers must be passed. This is the preferred way to execute communication since the buffers
/// can be reused.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain&              domain,
    grid::Grid4DDataScalar< ScalarType >&              grid,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& send_buffers,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& recv_buffers,
    const CommunicationReduction                       reduction = CommunicationReduction::SUM )
{
    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

} // namespace terra::communication::shell
