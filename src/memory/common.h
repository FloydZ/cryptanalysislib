#ifndef CRYPTANALYSISLIB_MEMORY_COMMON_H
#define CRYPTANALYSISLIB_MEMORY_COMMON_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory.h>`"
#endif

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#define POINTER_IS_32BYTES_ALIGNED(ptr) ((((uintptr_t)(ptr)) & (0b11111)) == 0)

#endif//CRYPTANALYSISLIB_MEMORY_H
