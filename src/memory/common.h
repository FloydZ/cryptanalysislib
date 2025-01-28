#ifndef CRYPTANALYSISLIB_MEMORY_COMMON_H
#define CRYPTANALYSISLIB_MEMORY_COMMON_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory/memory.h>`"
#endif

#include <cstddef>
#include <cstdlib>
#include <cstdint>

// TODO move to traits
// basic alignment cofnig
struct AlignmentConfig {
	// alignment in bytes
	constexpr static size_t alignment = 8;
} configAlignment;

#endif
