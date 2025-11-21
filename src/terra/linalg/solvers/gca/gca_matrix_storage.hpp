
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_wedges_per_hex_cell;

namespace terra::linalg::solvers {

///
template < GCACapable Operator, typename ScalarT >
class GCAMatrixStorage
{
  public:
    using ScalarType = ScalarT;

  private:
    // coarsest grid boolean field for elements on which a GCA hierarchy has to be built
    grid::Grid4DDataScalar< ScalarType > GCAElements_;
    const int                            level_range_;
    grid::shell::DistributedDomain       domain_;
    terra::grid::Grid4DDataMatrices< ScalarType, Operator::LocalMatrixDim, Operator::LocalMatrixDim, 2 >
                                     local_matrices_;

  public:
    GCAMatrixStorage(
        const grid::shell::DistributedDomain& domain,
        const int                             level_range,
        grid::Grid4DDataScalar< ScalarType >& GCAElements)
    : domain_( domain )
    , GCAElements_( GCAElements )
    , level_range_( level_range )
    , local_matrices_(
          "local_matrices_",
          domain_.subdomains().size(),
          domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          domain_.domain_info().subdomain_num_nodes_radially() - 1 )
    {}

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                                        local_subdomain_id,
        const int                                                                        x_cell,
        const int                                                                        y_cell,
        const int                                                                        r_cell,
        const int                                                                        wedge,
        const dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim >& mat ) const
    {
        assert( lmatrices_.data() != nullptr );
        for ( int i = 0; i < Operator::LocalMatrixDim; ++i )
        {
            for ( int j = 0; j < Operator::LocalMatrixDim; ++j )
            {
                lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j ) = mat( i, j );
            }
        }
    }

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > ijslice;
        for ( int i = 0; i < Operator::LocalMatrixDim; ++i )
        {
            for ( int j = 0; j < Operator::LocalMatrixDim; ++j )
            {
                ijslice( i, j ) = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j );
            }
        }
        return ijslice;
    }
};
} // namespace terra::linalg::solvers