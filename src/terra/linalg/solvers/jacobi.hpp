#pragma once

#include "solver.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT >
class Jacobi
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = SolutionVectorType::ScalarType;

    Jacobi(
        const SolutionVectorType& inverse_diagonal,
        const int                 iterations,
        const SolutionVectorType& tmp,
        const double              omega = 1.0 )
    : inverse_diagonal_( inverse_diagonal )
    , iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    {}

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            scale_in_place( tmp_, inverse_diagonal_ );
            lincomb( x, { 1.0, omega_ }, { x, tmp_ } );
        }
    }

  private:
    SolutionVectorType inverse_diagonal_;
    int                iterations_;
    SolutionVectorType tmp_;
    double             omega_;
};

static_assert( SolverLike< Jacobi< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers