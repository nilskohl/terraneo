

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

    PCG( const IterativeSolverParameters& params,
         const RHSVectorType&             r_tmp,
         const SolutionVectorType&        p_tmp,
         const RHSVectorType&             ap_tmp,
         const SolutionVectorType&        z_tmp )
    : PCG( params, r_tmp, p_tmp, ap_tmp, z_tmp, IdentitySolver< OperatorT >() )
    {}

    PCG( const IterativeSolverParameters& params,
         const RHSVectorType&             r_tmp,
         const SolutionVectorType&        p_tmp,
         const RHSVectorType&             ap_tmp,
         const SolutionVectorType&        z_tmp,
         const PreconditionerT            preconditioner )
    : tag_( "pcg_solver" )
    , params_( params )
    , r_( r_tmp )
    , p_( p_tmp )
    , ap_( ap_tmp )
    , z_( z_tmp )
    , preconditioner_( preconditioner )
    {}

    void set_tag( const std::string& tag ) { tag_ = tag; }

    void solve_impl(
        OperatorType&                                          A,
        SolutionVectorType&                                    x,
        const RHSVectorType&                                   b,
        std::optional< std::reference_wrapper< util::Table > > statistics )
    {
        apply( A, x, r_ );

        lincomb( r_, { 1.0, -1.0 }, { b, r_ } );

        solve( preconditioner_, A, z_, r_ );

        assign( p_, z_ );

        // TODO: should this be dot(z, z) instead or dot(r, r)?
        const ScalarType initial_residual = std::sqrt( dot( r_, r_ ) );

        if ( statistics.has_value() )
        {
            statistics->get().add_row(
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

            if ( statistics.has_value() )
            {
                statistics->get().add_row(
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

    RHSVectorType      r_;
    SolutionVectorType p_;
    RHSVectorType      ap_;
    SolutionVectorType z_;

    PreconditionerT preconditioner_;
};

static_assert(
    SolverLike<
        PCG< linalg::detail::
                 DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers