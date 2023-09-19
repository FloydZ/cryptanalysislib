#ifndef SMALLSECRETLWE_BEHCN_CONFIG_H
#define SMALLSECRETLWE_BEHCN_CONFIG_H

// external includes
#include <cstdint>
#include <vector>
#include <array>

// internal includes
#include "tree.h"
#include "matrix/fq_matrix.h"

#define SORT_INCREASING_ORDER
#define VALUE_KARY

// some dummy values
constexpr uint32_t n = 100;
constexpr uint32_t k = 100;

constexpr uint32_t q = 3;
using kAryType          = kAry_Type_T<uint32_t, uint64_t, q>;
/// TODO support this using kAryContainer     = Vector<kAryType, n>;
/// TODO support this using kAryContainer2    = Vector<kAryType, k>;
using kAryContainer     = kAryContainer_T<uint8_t, n, q>;
using kAryContainer2    = kAryContainer_T<uint8_t, k, q>;
using kAryLabel         = kAryContainer2;
using kAryValue         = kAryContainer;

//using kAryMatrix        = fplll::ZZ_mat<kAryType>;
using kAryMatrix        = FqMatrix<uint64_t, n, k, 15, false>;
using kAryElement       = Element_T<kAryValue, kAryLabel, kAryMatrix>;
using kAryList          = List_T<kAryElement>;
using kAryTree          = Tree_T<kAryList>;

using BinContainer      = BinaryContainer<n>;
using BinContainer2     = BinaryContainer<k>;
using BinaryLabel       = BinContainer2;
using BinaryValue       = BinContainer;
using BinaryMatrix      = FqMatrix<uint64_t, n, k, 2>;
using BinaryElement     = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList        = List_T<BinaryElement>;
using BinaryTree        = Tree_T<BinaryList>;

static  std::vector<uint64_t>   __level_translation_array{{0, 10, 20, 30, k}};
#endif //SMALLSECRETLWE_BEHCN_CONFIG_H
