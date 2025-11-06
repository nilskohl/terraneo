#pragma once

#include <string>
#include <variant>

namespace terra::util {

struct Ok
{};

template < typename T = Ok, typename E = std::string >
class Result
{
  public:
    // Success constructor
    Result( T value )
    : data( std::move( value ) )
    {}

    // Error constructor
    Result( E error )
    : data( std::move( error ) )
    {}

    // Check if it's a success
    [[nodiscard]] bool is_ok() const { return std::holds_alternative< T >( data ); }
    [[nodiscard]] bool is_err() const { return std::holds_alternative< E >( data ); }

    // Access value (call only if is_ok())
    T&       unwrap() { return std::get< T >( data ); }
    const T& unwrap() const { return std::get< T >( data ); }

    // Access error (call only if is_err())
    E&       error() { return std::get< E >( data ); }
    const E& error() const { return std::get< E >( data ); }

    // Return value or default if error
    T unwrap_or( T default_value ) const { return is_ok() ? std::get< T >( data ) : default_value; }

    // Transform the value if ok
    template < typename Func >
    auto map( Func f ) const -> Result< decltype( f( std::declval< T >() ) ), E >
    {
        if ( is_ok() )
            return f( std::get< T >( data ) );
        return std::get< E >( data );
    }

  private:
    std::variant< T, E > data;
};

} // namespace terra::util
