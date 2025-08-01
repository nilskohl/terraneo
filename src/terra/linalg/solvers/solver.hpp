
#pragma once

#include <optional>

#include "iterative_solver_info.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::util {
class Table;
}
namespace terra::linalg::solvers {

template < typename T >
concept SolverLike = requires(
    // TODO: Cannot make solver const since we may have temporaries as members.
    T& self,
    // TODO: See OperatorLike for why A is not const.
    typename T::OperatorType&                      A,
    typename T::OperatorType::SrcVectorType&       x,
    const typename T::OperatorType::DstVectorType& b ) {
    // Require exposing the operator type.
    typename T::OperatorType;

    // Require that the operator type satisfy OperatorLike
    requires OperatorLike< typename T::OperatorType >;

    { self.solve_impl( A, x, b ) } -> std::same_as< void >;
};

template < SolverLike Solver >
using SolutionOf = SrcOf< typename Solver::OperatorType >;

template < SolverLike Solver >
using RHSOf = DstOf< typename Solver::OperatorType >;

template < SolverLike Solver, OperatorLike Operator, VectorLike SolutionVector, VectorLike RHSVector >
void solve( Solver& solver, Operator& A, SolutionVector& x, const RHSVector& b )
{
    solver.solve_impl( A, x, b );
}

namespace detail {

template < OperatorLike OperatorT >
class DummySolver
{
  public:
    using OperatorType = OperatorT;

    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    void solve_impl( const OperatorType& A, SolutionVectorType& x, const RHSVectorType& b ) const
    {
        (void) A;
        (void) x;
        (void) b;
    }
};

static_assert( SolverLike< DummySolver< linalg::detail::DummyConcreteOperator > > );

} // namespace detail

} // namespace terra::linalg::solvers