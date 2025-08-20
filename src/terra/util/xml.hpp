

#pragma once

namespace terra::util {

#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class XML
{
  public:
    explicit XML(
        std::string                                 name,
        const std::map< std::string, std::string >& attributes = {},
        std::string                                 content    = "" )
    : name_( std::move( name ) )
    , content_( std::move( content ) )
    {
        attributes_.insert( attributes.begin(), attributes.end() );
    }

    XML& add_child( const XML& child )
    {
        children_.push_back( child );
        return *this;
    }

    [[nodiscard]] std::string to_string() const { return to_string( 0 ); }

  private:
    std::string                          name_;
    std::string                          content_;
    std::vector< XML >                   children_;
    std::map< std::string, std::string > attributes_;

    [[nodiscard]] std::string to_string( int indent ) const
    {
        std::ostringstream oss;
        std::string        indent_str( indent, ' ' );

        oss << indent_str << "<" << name_;
        for ( const auto& attr : attributes_ )
        {
            oss << " " << attr.first << "=\"" << escape_xml( attr.second ) << "\"";
        }

        if ( children_.empty() && content_.empty() )
        {
            oss << " />\n";
        }
        else
        {
            oss << ">";
            if ( !content_.empty() )
            {
                oss << escape_xml( content_ );
            }
            if ( !children_.empty() )
            {
                oss << "\n";
                for ( const auto& child : children_ )
                {
                    oss << child.to_string( indent + 2 );
                }
                oss << indent_str;
            }
            oss << "</" << name_ << ">\n";
        }

        return oss.str();
    }

    static std::string escape_xml( const std::string& data )
    {
        std::ostringstream escaped;
        for ( char c : data )
        {
            switch ( c )
            {
            case '&':
                escaped << "&amp;";
                break;
            case '\"':
                escaped << "&quot;";
                break;
            case '\'':
                escaped << "&apos;";
                break;
            case '<':
                escaped << "&lt;";
                break;
            case '>':
                escaped << "&gt;";
                break;
            default:
                escaped << c;
                break;
            }
        }
        return escaped.str();
    }
};

} // namespace terra::util