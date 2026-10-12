#pragma once
#include <concepts>
#include <cstddef>
#include <utility>


#ifdef __cpp_static_call_operator
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC static
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST 
#else
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST const
#endif

/// the value type of `hash`: an integer (the hash uses ^, +, << and >>)
template<class T>
concept StandardHashAble = std::integral<T>;

/// Generic hash function implementation for basic types
/// Uses a simple but effective combination of XOR, addition, and bit shifts
/// This implementation is designed for performance and distribution quality
/// 
/// \tparam T1 Type of value to be hashed (an integer, see `StandardHashAble`)
template<class T1>
#if __cplusplus > 201709L
	requires StandardHashAble<T1>
#endif
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
