
#pragma once

#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Zero
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

    Zero() = default;

    void apply_impl(
        const SrcVectorType&            src,
        DstVectorType&                  dst,
        int                             level,
        const linalg::OperatorApplyMode operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        if ( operator_apply_mode == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0, level );
        }
    }
};

static_assert( linalg::OperatorLike< Zero< double > > );

} // namespace terra::fe::wedge::operators::shell