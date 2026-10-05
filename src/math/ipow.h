#ifndef CRYPTANALYSISLIB_IPOW_H
#define CRYPTANALYSISLIB_IPOW_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>
#include "helper.h"

namespace cryptanalysislib::math {

	template <typename T, typename T2>
	__device__ __host__
#if __cplusplus > 201709L
    	requires std::is_arithmetic_v<T> &&
    	         std::is_integral_v<T2>
#endif
	constexpr T ipow(T x, T2 n) {
	    return (n == 0) ? T{1} :
	           n == 1 ? x :
	           n > 1 ? ((n & 1) ? x * ipow(x, n-1) : ipow(x, n/2) * ipow(x, n/2)) :
	           T{1} / ipow(x, -n);
	}

    /// square and multiply
    /// \return a**ex
    template <typename Type1, 
              typename Type2>
    Type1 ipow_v2(Type1 a,
                  Type2 ex) {
        if (ex == 0) {
            return 1;
        }

        Type1 z = a;
        Type1 y = 1;
        while (1){
            if (ex & 1u) { 
                y *= z;
            }

            ex /= 2;
            if ( 0==ex ){ 
                break;
            }

            z *= z;
        }
        return y;
    }
}

#endif //CRYPTANALYSISLIB_IPOW_H
