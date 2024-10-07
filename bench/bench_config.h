#ifndef CRYPTANALYSISLIB_BENCH_CONFIG_H
#define CRYPTANALYSISLIB_BENCH_CONFIG_H

// external includes
#include <cstdint>
#include <vector>
#include <array>

// internal includes
#include "tree.h"
#include "matrix/fq_matrix.h"

#define SORT_INCREASING_ORDER

// some dummy values
constexpr uint32_t n = 100;
constexpr uint32_t k = 100;
constexpr uint32_t q = 3;

using kAryType          = kAry_Type_T<q>;
using kAryContainer     = FqNonPackedVector< n, q, uint8_t>;
using kAryContainer2    = FqNonPackedVector< k, q, uint8_t>;
using kAryLabel         = kAryContainer2;
using kAryValue         = kAryContainer;

//using kAryMatrix        = fplll::ZZ_mat<kAryType>;
using kAryMatrix        = FqMatrix<uint64_t, n, k, q, false>;
using kAryElement       = Element_T<kAryValue, kAryLabel, kAryMatrix>;
using kAryList          = List_T<kAryElement>;
using kAryTree          = Tree_T<kAryList>;

using BinContainer      = BinaryVector<n>;
using BinContainer2     = BinaryVector<k>;
using BinaryLabel       = BinContainer2;
using BinaryValue       = BinContainer;
using BinaryMatrix      = FqMatrix<uint64_t, n, k, 2>;
using BinaryElement     = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList        = List_T<BinaryElement>;
using BinaryTree        = Tree_T<BinaryList>;

static  std::vector<uint32_t>   __level_translation_array{{0, 10, 20, 30, k}};
#endif
