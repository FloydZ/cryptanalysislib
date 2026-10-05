#pragma once
#include <utility>


#ifdef __cpp_static_call_operator
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC static
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST 
#else
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST const
#endif

// TODO add concept for T1

/// Generic hash function implementation for basic types
/// Uses a simple but effective combination of XOR, addition, and bit shifts
/// This implementation is designed for performance and distribution quality
/// 
/// \tparam T1 Type of value to be hashed (must support arithmetic operations)
template<class T1>
struct hash { 
public:
    /// Computes a hash value for the given input
    /// Uses the standard multiplicative hashing technique with the golden ratio constant
    /// 
    /// \param p[in] Value to be hashed
    /// \return Computed hash value as size_t
    constexpr inline CRYPTANALYSISLIB_HASH_STANDARD_STATIC 
    size_t operator()(const T1 &p) CRYPTANALYSISLIB_HASH_STANDARD_CONST noexcept {
        return p ^ (p + 0x9e3779b9 + (p<<6) + (p>>2));
    }
}; 
