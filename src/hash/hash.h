#ifndef CRYPTANALYSISLIB_HASH_H
#define CRYPTANALYSISLIB_HASH_H

#include <concepts>

#if __cplusplus > 201709L
// defines the concept of a Hash function
template<class H, class T>
concept HashFunction =
        std::invocable<H, const T&>;

template<class C, class T>
concept CompareFunction =
	std::regular_invocable<C, const T&, const T&>;
#endif


#include "adler32.h"
#include "cityhash.h"
#include "crc.h"
#include "fnv1a.h"
#include "simple.h"
#include "xxh3.h"

#endif//CRYPTANALYSISLIB_HASH_H
