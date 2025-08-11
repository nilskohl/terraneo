

#pragma once

#include "solver.hpp"
#include "util/table.hpp"
#include "vtk/vtk.hpp"

namespace terra::linalg::solvers {

template <
    OperatorLike OperatorT,
    OperatorLike ProlongationT,
    OperatorLike RestrictionT,
    OperatorLike CoarseGridOperatorT,
    SolverLike   SmootherT,
    SolverLike   CoarseGridSolverT >
class Multigrid
{
  public:
    using OperatorType           = OperatorT;
    using ProlongationType       = ProlongationT;
    using RestrictionType        = RestrictionT;
    using CoarseGridOperatorType = CoarseGridOperatorT;
    using SmootherType           = SmootherT;
    using CoarseGridSolverType   = CoarseGridSolverT;

    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = SolutionVectorType::ScalarType;

  private:
    std::vector< ProlongationType >       P_additive_;
    std::vector< RestrictionType >        R_;
    std::vector< CoarseGridOperatorType > A_c_;
    std::vector< SolutionVectorType >     tmp_r_;
    std::vector< SolutionVectorType >     tmp_e_;
    std::vector< SolutionVectorType >     tmp_;
    std::vector< SmootherType >           smoothers_pre_;
    std::vector< SmootherType >           smoothers_post_;
    CoarseGridSolverType                  coarse_grid_solver_;

    int        num_cycles_;
    ScalarType relative_residual_threshold_;

    std::shared_ptr< util::Table > statistics_;
    std::string                    tag_ = "multigrid";

  public:
    Multigrid(
        const std::vector< ProlongationType >&       P_additive,
        const std::vector< RestrictionType >&        R,
        const std::vector< CoarseGridOperatorType >& A_c,
        const std::vector< SolutionVectorType >&     tmp_r,
        const std::vector< SolutionVectorType >&     tmp_e,
        const std::vector< SolutionVectorType >&     tmp,
        const std::vector< SmootherType >&           smoothers_pre,
        const std::vector< SmootherType >&           smoothers_post,
        const CoarseGridSolverType&                  coarse_grid_solver,
        int                                          num_cycles,
        ScalarType                                   relative_residual_threshold )
    : P_additive_( P_additive )
    , R_( R )
    , A_c_( A_c )
    , tmp_r_( tmp_r )
    , tmp_e_( tmp_e )
    , tmp_( tmp )
    , smoothers_pre_( smoothers_pre )
    , smoothers_post_( smoothers_post )
    , coarse_grid_solver_( coarse_grid_solver )
    , num_cycles_( num_cycles )
    , relative_residual_threshold_( relative_residual_threshold )
    {}

    void set_tag( const std::string& tag ) { tag_ = tag; }
    void collect_statistics( const std::shared_ptr< util::Table >& statistics ) { statistics_ = statistics; }

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        if ( P_additive_.size() != A_c_.size() || R_.size() != A_c_.size() || tmp_r_.size() != A_c_.size() ||
             tmp_e_.size() != A_c_.size() || tmp_.size() != A_c_.size() + 1 )
        {
            throw std::runtime_error(
                "Multigrid: P_additive, R, and A_c must be available for all coarse levels. tmp_0, tmp_1, tmp_2 "
                "require the finest grid allocated, too." );
        }

        const int max_level = P_additive_.size();

        ScalarType initial_residual = 0.0;

        if ( statistics_ )
        {
            apply( A, x, tmp_[max_level] );
            lincomb( tmp_[max_level], { 1.0, -1.0 }, { b, tmp_[max_level] } );
            initial_residual = norm_2( tmp_[max_level] );

            statistics_->add_row(
                { { "tag", tag_ },
                  { "cycle", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual },
                  { "residual_convergence_rate", 1.0 } } );
        }

        ScalarType previous_absolut_residual = initial_residual;

        for ( int cycle = 1; cycle <= num_cycles_; ++cycle )
        {
            solve_recursive( A, x, b, max_level );

            if ( statistics_ )
            {
                apply( A, x, tmp_[max_level] );
                lincomb( tmp_[max_level], { 1.0, -1.0 }, { b, tmp_[max_level] } );
                const auto absolute_residual = norm_2( tmp_[max_level] );

                const auto relative_residual = absolute_residual / initial_residual;

                statistics_->add_row(
                    { { "tag", tag_ },
                      { "cycle", cycle },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual },
                      { "residual_convergence_rate", absolute_residual / previous_absolut_residual } } );

                if ( relative_residual <= relative_residual_threshold_ )
                {
                    return;
                }

                previous_absolut_residual = absolute_residual;
            }
        }
    }

  private:
    void solve_recursive( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b, int level )
    {
        if ( level == 0 )
        {
            solve( coarse_grid_solver_, A, tmp_e_[0], tmp_r_[0] );
            return;
        }

        // relax on Ax = b
        solve( smoothers_pre_[level], A, x, b );

        // compute the residual r = b - Ax
        apply( A, x, tmp_[level] );
        lincomb( tmp_[level], { 1.0, -1.0 }, { b, tmp_[level] } );

        // restrict the residual r_c = R r_f
        apply( R_[level - 1], tmp_[level], tmp_r_[level - 1] );

        // solve (recursively) A_c e_c = r_c
        assign( tmp_e_[level - 1], 0.0 );
        solve_recursive( A_c_[level - 1], tmp_e_[level - 1], tmp_r_[level - 1], level - 1 );

        // apply the coarse grid correction x = x + P e_c
        apply( P_additive_[level - 1], tmp_e_[level - 1], x );

        // relax on A x = b
        solve( smoothers_post_[level], A, x, b );
    }
};
} // namespace terra::linalg::solvers