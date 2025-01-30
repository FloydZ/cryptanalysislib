#pragma once
#include <utility>


#ifdef __cpp_static_call_operator
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC static
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST 
#else
#define CRYPTANALYSISLIB_HASH_STANDARD_STATIC
#define CRYPTANALYSISLIB_HASH_STANDARD_CONST const
#endif


/// 
template<class T1>
struct hash { 
public:
    constexpr inline CRYPTANALYSISLIB_HASH_STANDARD_STATIC size_t operator()(const T1 &p) CRYPTANALYSISLIB_HASH_STANDARD_CONST noexcept {
        return p ^ (p + 0x9e3779b9 + (p<<6) + (p>>2));
    }
}; 
