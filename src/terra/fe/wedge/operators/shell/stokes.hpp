
#pragma once

#include "communication/shell/communication.hpp"
#include "divergence.hpp"
#include "gradient.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "vector_laplace.hpp"
#include "zero.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim = 3 >
class Stokes
{
  public:
    using SrcVectorType = linalg::VectorQ1IsoQ2Q1< double, VecDim >;
    using DstVectorType = linalg::VectorQ1IsoQ2Q1< double, VecDim >;
    using ScalarType    = ScalarT;

    using Block11Type = VectorLaplace< double, VecDim >;
    using Block12Type = Gradient< double >;
    using Block21Type = Divergence< double >;
    using Block22Type = Zero< double >;

  private:
    VectorLaplace< double, VecDim > A_;
    VectorLaplace< double, VecDim > A_diagonal_;
    Gradient< double >              B_T_;
    Divergence< double >            B_;
    Zero< double >                  O_;

    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

  public:
    Stokes(
        const grid::shell::DistributedDomain&   domain_fine,
        const grid::shell::DistributedDomain&   domain_coarse,
        const grid::Grid3DDataVec< double, 3 >& grid,
        const grid::Grid2DDataScalar< double >& radii,
        bool                                    treat_boundary,
        bool                                    diagonal,
        linalg::OperatorApplyMode               operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode       operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : A_( domain_fine,
          grid,
          radii,
          treat_boundary,
          false,
          linalg::OperatorApplyMode::Replace,
          linalg::OperatorCommunicationMode::SkipCommunication )
    , A_diagonal_( domain_fine, grid, radii, treat_boundary, true, operator_apply_mode, operator_communication_mode )
    , B_T_(
          domain_fine,
          domain_coarse,
          grid,
          radii,
          treat_boundary,
          linalg::OperatorApplyMode::Add,
          operator_communication_mode )
    , B_( domain_fine, domain_coarse, grid, radii, treat_boundary, operator_apply_mode, operator_communication_mode )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( diagonal_ )
        {
            apply( A_diagonal_, src.block_1(), dst.block_1() );
            return;
        }

        apply( A_, src.block_1(), dst.block_1() );
        apply( B_T_, src.block_2(), dst.block_1() );
        apply( B_, src.block_1(), dst.block_2() );
    }

    const Block11Type& block_11() const { return A_; }
    const Block12Type& block_12() const { return B_T_; }
    const Block21Type& block_21() const { return B_; }
    const Block22Type& block_22() const { return O_; }

    Block11Type& block_11() { return A_; }
    Block12Type& block_12() { return B_T_; }
    Block21Type& block_21() { return B_; }
    Block22Type& block_22() { return O_; }
};

static_assert( linalg::Block2x2OperatorLike< Stokes< double, 3 > > );

} // namespace terra::fe::wedge::operators::shell