
#pragma once
#include <chrono>
#include <ctime>

namespace terra::util {

/// @brief Get the current timestamp as a string.
/// @return Timestamp string.
inline std::string current_timestamp()
{
    using namespace std::chrono;
    const auto  now = system_clock::now();
    std::time_t t   = system_clock::to_time_t( now );
    std::tm     buf{};
#ifdef _WIN32
    localtime_s( &buf, &t );
#else
    localtime_r( &t, &buf );
#endif
    char str[32];
    std::strftime( str, sizeof( str ), "%Y-%m-%d %H:%M:%S", &buf );
    return { str };
}

} // namespace terra::util
