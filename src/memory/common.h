#ifndef CRYPTANALYSISLIB_MEMORY_COMMON_H
#define CRYPTANALYSISLIB_MEMORY_COMMON_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory/memory.h>`"
#endif

#include <cstddef>
#include <cstdlib>
#include <cstdint>

#define POINTER_IS_32BYTES_ALIGNED(ptr) ((((uintptr_t)(ptr)) & (0b11111)) == 0)

// basic alignment cofnig
struct AlignmentConfig {
	// alignment in bytes
	constexpr static size_t alignment = 8;
} configAlignment;

#endif
