#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H

#include <cstdint>

// Transpose 8x8 bit array packed into a single quadword 
constexpr uint64_t transpose_b8x8(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;
    x = x ^ t ^ (t << 28);
	return x;
}

// Transpose 8x8 bit array along the diagonal from upper right
constexpr uint64_t transpose_b8x8_be(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 9)) & 0x0055005500550055LL;
    x = x ^ t ^ (t << 9);
    t = (x ^ (x >> 18)) & 0x0000333300003333LL;
    x = x ^ t ^ (t << 18);
    t = (x ^ (x >> 36)) & 0x000000000F0F0F0FLL;
    x = x ^ t ^ (t << 36);
	return x;
}
#endif
