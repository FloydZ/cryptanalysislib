#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstdio>

/// wrapper class/struct arounf __m256i for better debugging
union U256i {
    __m256i v;
    uint32_t a[8];
    uint64_t b[4];
};

/// prints a `__m256i` as 8 `u32`
static void print_m256i_u32(const __m256i v){
    const U256i u = { v };

    for (uint32_t i = 0; i < 8; ++i) {
        printf("%d ", u.a[i]);

	}
	
	printf("\n");
}
