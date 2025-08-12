

#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT >
class IdentitySolver
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        assign( x, b );
        (void) A;
    }
};

static_assert( SolverLike< IdentitySolver< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers