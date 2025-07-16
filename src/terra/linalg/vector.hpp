
#pragma once

namespace terra::linalg {

template < typename T >
concept VectorLike = requires(
    const T&                                     self_const,
    T&                                           self,
    const std::vector< typename T::ScalarType >& c,
    const T&                                     x,
    T&                                           x_non_const,
    const std::vector< T >&                      xx,
    const typename T::ScalarType                 c0,
    const int                                    level ) {
    // Requires exposing the scalar type.
    typename T::ScalarType;

    // Required lincomb overload with 4 args
    { self.lincomb_impl( c, xx, c0, level ) } -> std::same_as< void >;

    // Required dot product
    { self_const.dot_impl( x, level ) } -> std::same_as< typename T::ScalarType >;

    // Required max magnitude
    { self_const.max_abs_entry_impl( level ) } -> std::same_as< typename T::ScalarType >;

    // Required nan check
    { self_const.has_nan_impl( level ) } -> std::same_as< bool >;

    // Required swap operation
    { self.swap_impl( x_non_const ) } -> std::same_as< void >;
};

template < typename T >
concept Block2VectorLike = VectorLike< T > && requires( const T& self_const, T& self ) {
    typename T::Block1Type;
    typename T::Block2Type;

    requires VectorLike< typename T::Block1Type >;
    requires VectorLike< typename T::Block2Type >;

    { self_const.block_1() } -> std::same_as< const typename T::Block1Type& >;
    { self_const.block_2() } -> std::same_as< const typename T::Block2Type& >;

    { self.block_1() } -> std::same_as< typename T::Block1Type& >;
    { self.block_2() } -> std::same_as< typename T::Block2Type& >;
};

template < VectorLike Vector >
using ScalarOf = typename Vector::ScalarType;

template < VectorLike Vector >
void lincomb(
    Vector&                                  y,
    const std::vector< ScalarOf< Vector > >& c,
    const std::vector< Vector >&             x,
    const ScalarOf< Vector >&                c0,
    const int                                level )
{
    y.lincomb_impl( c, x, c0, level );
}

template < VectorLike Vector >
void lincomb( Vector& y, const std::vector< ScalarOf< Vector > >& c, const std::vector< Vector >& x, const int level )
{
    lincomb( y, c, x, static_cast< ScalarOf< Vector > >( 0 ), level );
}

template < VectorLike Vector >
void assign( Vector& y, const ScalarOf< Vector >& c0, const int level )
{
    lincomb( y, {}, {}, c0, level );
}

template < VectorLike Vector >
void assign( Vector& y, const Vector& x, const int level )
{
    lincomb( y, { static_cast< ScalarOf< Vector > >( 1 ) }, { x }, level );
}

template < VectorLike Vector >
ScalarOf< Vector > dot( const Vector& y, const Vector& x, const int level )
{
    return y.dot_impl( x, level );
}

template < VectorLike Vector >
ScalarOf< Vector > inf_norm( const Vector& y, const int level )
{
    return y.max_abs_entry_impl( level );
}

template < VectorLike Vector >
bool has_nan( const Vector& y, const int level )
{
    return y.has_nan_impl( level );
}

template < VectorLike Vector >
void swap( Vector& x, Vector& y )
{
    y.swap_impl( x );
}

namespace detail {

template < typename ScalarT >
class DummyVector
{
  public:
    using ScalarType = ScalarT;

    void lincomb_impl(
        const std::vector< ScalarType >&  c,
        const std::vector< DummyVector >& x,
        const ScalarType                  c0,
        const int                         level )
    {
        (void) c;
        (void) x;
        (void) c0;
        (void) level;
    }

    ScalarType dot_impl( const DummyVector& x, const int level ) const
    {
        (void) x;
        (void) level;
        return 0;
    }

    ScalarType max_abs_entry_impl( const int level ) const
    {
        (void) level;
        return 0;
    }

    bool has_nan_impl( const int level ) const
    {
        (void) level;
        return false;
    }

    void swap_impl( DummyVector< ScalarType >& other ) { (void) other; }
};

template < typename ScalarT >
class DummyBlock2Vector
{
  public:
    using ScalarType = ScalarT;

    using Block1Type = DummyVector< ScalarType >;
    using Block2Type = DummyVector< ScalarType >;

    void lincomb_impl(
        const std::vector< ScalarType >&        c,
        const std::vector< DummyBlock2Vector >& x,
        const ScalarType                        c0,
        const int                               level )
    {
        (void) c;
        (void) x;
        (void) c0;
        (void) level;
    }

    ScalarType dot_impl( const DummyBlock2Vector& x, const int level ) const
    {
        (void) x;
        (void) level;
        return 0;
    }

    ScalarType max_abs_entry_impl( const int level ) const
    {
        (void) level;
        return 0;
    }

    bool has_nan_impl( const int level ) const
    {
        (void) level;
        return false;
    }

    void swap_impl( DummyBlock2Vector& other ) { (void) other; }

    const DummyVector< ScalarType >& block_1() const { return block_1_; }
    const DummyVector< ScalarType >& block_2() const { return block_2_; }

    DummyVector< ScalarType >& block_1() { return block_1_; }
    DummyVector< ScalarType >& block_2() { return block_2_; }

  private:
    DummyVector< ScalarType > block_1_;
    DummyVector< ScalarType > block_2_;
};

static_assert( VectorLike< DummyVector< double > > );
static_assert( Block2VectorLike< DummyBlock2Vector< double > > );

} // namespace detail

} // namespace terra::linalg