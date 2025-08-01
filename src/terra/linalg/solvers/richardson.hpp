#pragma once

#include <iostream>

#include "solver.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT >
class Richardson
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    Richardson( const int iterations, const double omega, const RHSVectorType& r_tmp )
    : iterations_( iterations )
    , omega_( omega )
    , r_( r_tmp )
    {}

    void solve_impl(
        OperatorType&                                          A,
        SolutionVectorType&                                    x,
        const RHSVectorType&                                   b,
        std::optional< std::reference_wrapper< util::Table > > statistics )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            assign( r_, 0 );
            apply( A, x, r_ );
            lincomb( x, { 1.0, omega_, -omega_ }, { x, b, r_ } );
        }
    }

  private:
    int           iterations_;
    double        omega_;
    RHSVectorType r_;
};

static_assert( SolverLike< Richardson< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers