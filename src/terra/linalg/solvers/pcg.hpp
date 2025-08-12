

#pragma once

#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PCG
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = typename SolutionVectorType::ScalarType;

    PCG( const IterativeSolverParameters&         params,
         const std::shared_ptr< util::Table >&    statistics,
         const std::vector< SolutionVectorType >& tmps )
    : PCG( params, statistics, tmps, IdentitySolver< OperatorT >() )
    {}

    PCG( const IterativeSolverParameters&         params,
         const std::shared_ptr< util::Table >&    statistics,
         const std::vector< SolutionVectorType >& tmps,
         const PreconditionerT                    preconditioner )
    : tag_( "pcg_solver" )
    , params_( params )
    , statistics_( statistics )
    , tmps_( tmps )
    , preconditioner_( preconditioner )
    {
        if ( tmps.size() < 4 )
        {
            throw std::runtime_error( "PCG: tmps.size() < 4. Need at least 4 tmp vectors." );
        }
    }

    void set_tag( const std::string& tag ) { tag_ = tag; }

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        auto& r_  = tmps_[0];
        auto& p_  = tmps_[1];
        auto& ap_ = tmps_[2];
        auto& z_  = tmps_[3];

        apply( A, x, r_ );

        lincomb( r_, { 1.0, -1.0 }, { b, r_ } );

        solve( preconditioner_, A, z_, r_ );

        assign( p_, z_ );

        // TODO: should this be dot(z, z) instead or dot(r, r)?
        const ScalarType initial_residual = std::sqrt( dot( r_, r_ ) );

        if ( statistics_ )
        {
            statistics_->add_row(
                { { "tag", tag_ },
                  { "iteration", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual } } );
        }

        if ( initial_residual < params_.absolute_residual_tolerance() )
        {
            return;
        }

        for ( int iteration = 1; iteration <= params_.max_iterations(); ++iteration )
        {
            const ScalarType alpha_num = dot( z_, r_ );

            apply( A, p_, ap_ );
            const ScalarType alpha_den = dot( ap_, p_ );

            const ScalarType alpha = alpha_num / alpha_den;

            lincomb( x, { 1.0, alpha }, { x, p_ } );
            lincomb( r_, { 1.0, -alpha }, { r_, ap_ } );

            // TODO: is this the correct term for the residual check?
            const ScalarType absolute_residual = std::sqrt( dot( r_, r_ ) );

            const ScalarType relative_residual = absolute_residual / initial_residual;

            if ( statistics_ )
            {
                statistics_->add_row(
                    { { "tag", tag_ },
                      { "iteration", iteration },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual } } );
            }

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                return;
            }

            if ( absolute_residual < params_.absolute_residual_tolerance() )
            {
                return;
            }

            solve( preconditioner_, A, z_, r_ );

            const ScalarType beta_num = dot( z_, r_ );
            const ScalarType beta     = beta_num / alpha_num;

            lincomb( p_, { 1.0, beta }, { z_, p_ } );
        }
    }

  private:
    std::string tag_;

    IterativeSolverParameters params_;

    std::shared_ptr< util::Table > statistics_;

    std::vector< SolutionVectorType > tmps_;

    PreconditionerT preconditioner_;
};

static_assert(
    SolverLike<
        PCG< linalg::detail::
                 DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers