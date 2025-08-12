

#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

template < Block2x2OperatorLike OperatorT, SolverLike Block11Preconditioner, SolverLike Block22Preconditioner >
class BlockDiagonalPreconditioner2x2
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "The solution vector of the BlockPreconditioner2x2 must be Block2VectorLike." );
    static_assert(
        Block2VectorLike< RHSVectorType >,
        "The RHS vector of the BlockPreconditioner2x2 must be Block2VectorLike." );

    BlockDiagonalPreconditioner2x2(
        const Block11Preconditioner& block11_preconditioner,
        const Block22Preconditioner& block22_preconditioner )
    : block11_preconditioner_( block11_preconditioner )
    , block22_preconditioner_( block22_preconditioner )
    {}

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        solve( block11_preconditioner_, A.block_11(), x.block_1(), b.block_1() );
        solve( block22_preconditioner_, A.block_22(), x.block_2(), b.block_2() );
    }

  private:
    Block11Preconditioner block11_preconditioner_;
    Block22Preconditioner block22_preconditioner_;
};

static_assert( SolverLike< BlockDiagonalPreconditioner2x2<
                   linalg::detail::DummyConcreteBlock2x2Operator,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block11Type >,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block22Type > > > );

} // namespace terra::linalg::solvers