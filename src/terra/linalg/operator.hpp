
#pragma once
#include "vector.hpp"

namespace terra::linalg {

enum class OperatorApplyMode
{
    Replace,
    Add,
};

enum class OperatorCommunicationMode
{
    SkipCommunication,
    CommunicateAdditively,
};

template < typename T >
concept OperatorLike =

    requires(
        const T&                         self_const,
        T&                               self,
        const typename T::SrcVectorType& src,
        typename T::DstVectorType&       dst,
        dense::Vec< int, 2 >             block,
        OperatorApplyMode                operator_apply_mode,
        OperatorCommunicationMode        operator_communication_mode ) {
        // Requires exposing the vector types.
        typename T::SrcVectorType;
        typename T::DstVectorType;

        // Require that Src and Dst vector types satisfy VectorLike
        requires VectorLike< typename T::SrcVectorType >;
        requires VectorLike< typename T::DstVectorType >;

        // Required matvec implementation
        // TODO: T& self is not const because having apply_impl const is not convenient - mostly because
        //       it is handy to reuse send/recv buffers that are members of an operator implementation.
        //       To modify these members (i.e., to communicate) we cannot be const :(
        { self.apply_impl( src, dst, operator_apply_mode, operator_communication_mode ) } -> std::same_as< void >;
    };

template < typename T >
concept Block2x2OperatorLike = OperatorLike< T > &&

                               requires( const T& self_const, T& self ) {
                                   typename T::Block11Type;
                                   typename T::Block12Type;
                                   typename T::Block21Type;
                                   typename T::Block22Type;

                                   requires OperatorLike< typename T::Block11Type >;
                                   requires OperatorLike< typename T::Block11Type >;
                                   requires OperatorLike< typename T::Block21Type >;
                                   requires OperatorLike< typename T::Block22Type >;

                                   { self_const.block_11() } -> std::same_as< const typename T::Block11Type& >;
                                   { self_const.block_12() } -> std::same_as< const typename T::Block12Type& >;
                                   { self_const.block_21() } -> std::same_as< const typename T::Block21Type& >;
                                   { self_const.block_22() } -> std::same_as< const typename T::Block22Type& >;

                                   { self.block_11() } -> std::same_as< typename T::Block11Type& >;
                                   { self.block_12() } -> std::same_as< typename T::Block12Type& >;
                                   { self.block_21() } -> std::same_as< typename T::Block21Type& >;
                                   { self.block_22() } -> std::same_as< typename T::Block22Type& >;
                               };

template < OperatorLike Operator >
using SrcOf = typename Operator::SrcVectorType;

template < OperatorLike Operator >
using DstOf = typename Operator::DstVectorType;

template < OperatorLike Operator >
void apply(
    Operator&                       A,
    const SrcOf< Operator >&        src,
    DstOf< Operator >&              dst,
    const OperatorApplyMode         operator_apply_mode         = OperatorApplyMode::Replace,
    const OperatorCommunicationMode operator_communication_mode = OperatorCommunicationMode::CommunicateAdditively )
{
    A.apply_impl( src, dst, operator_apply_mode, operator_communication_mode );
}

namespace detail {

template < VectorLike SrcVectorT, VectorLike DstVectorT >
class DummyOperator
{
  public:
    using SrcVectorType = SrcVectorT;
    using DstVectorType = DstVectorT;

    void apply_impl(
        const SrcVectorType&            src,
        DstVectorType&                  dst,
        const OperatorApplyMode         operator_apply_mode,
        const OperatorCommunicationMode operator_communication_mode ) const
    {
        (void) src;
        (void) dst;
        (void) operator_apply_mode;
        (void) operator_communication_mode;
    }
};

class DummyConcreteOperator
{
  public:
    using SrcVectorType = DummyVector< double >;
    using DstVectorType = DummyVector< double >;

    void apply_impl(
        const SrcVectorType&            src,
        DstVectorType&                  dst,
        const OperatorApplyMode         operator_apply_mode,
        const OperatorCommunicationMode operator_communication_mode ) const
    {
        (void) src;
        (void) dst;
        (void) operator_apply_mode;
        (void) operator_communication_mode;
    }
};

class DummyConcreteBlock2x2Operator
{
  public:
    using SrcVectorType = DummyBlock2Vector< double >;
    using DstVectorType = DummyBlock2Vector< double >;

    using Block11Type = DummyConcreteOperator;
    using Block12Type = DummyConcreteOperator;
    using Block21Type = DummyConcreteOperator;
    using Block22Type = DummyConcreteOperator;

    void apply_impl(
        const SrcVectorType&            src,
        DstVectorType&                  dst,
        const OperatorApplyMode         operator_apply_mode,
        const OperatorCommunicationMode operator_communication_mode ) const
    {
        (void) src;
        (void) dst;
        (void) operator_apply_mode;
        (void) operator_communication_mode;
    }

    const Block11Type& block_11() const { return block_11_; }
    const Block12Type& block_12() const { return block_12_; }
    const Block21Type& block_21() const { return block_21_; }
    const Block22Type& block_22() const { return block_22_; }

    Block11Type& block_11() { return block_11_; }
    Block12Type& block_12() { return block_12_; }
    Block21Type& block_21() { return block_21_; }
    Block22Type& block_22() { return block_22_; }

  private:
    DummyConcreteOperator block_11_;
    DummyConcreteOperator block_12_;
    DummyConcreteOperator block_21_;
    DummyConcreteOperator block_22_;
};

static_assert( OperatorLike< DummyOperator< DummyVector< double >, DummyVector< double > > > );
static_assert( OperatorLike< DummyConcreteOperator > );
static_assert( Block2x2OperatorLike< DummyConcreteBlock2x2Operator > );

} // namespace detail

} // namespace terra::linalg