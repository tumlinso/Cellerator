#pragma once
#include <cstdlib>
#include <iostream>

// Test acceptance must execute in Release as well as Debug. Evaluate once,
// retain the expression and location, and never depend on the NDEBUG macro.
#define SPINE_REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << __FILE__ << ':' << __LINE__ \
                      << ": failed: " << #condition << '\n'; \
            std::abort(); \
        } \
    } while (false)
