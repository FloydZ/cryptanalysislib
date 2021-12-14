#ifndef SMALLSECRETLWE_BEHCN_CONFIG_H
#define SMALLSECRETLWE_BEHCN_CONFIG_H

// external includes
#include <cstdint>
#include <vector>
#include <array>

// internal includes
#include <tree.h>

#define SORT_INCREASING_ORDER
#define VALUE_KARY

// some dummy values
constexpr uint32_t n = 100;
constexpr uint32_t k = 100;

using kAryType          = kAry_Type_T<uint32_t, uint64_t, 3>;
using kAryContainer     = kAryContainer_T<kAryType, n>;
using kAryContainer2    = kAryContainer_T<kAryType, k>;
using kAryLabel         = Label_T<kAryContainer2>;
using kAryValue         = Value_T<kAryContainer>;
#ifdef USE_FPLLL
using kAryMatrix        = fplll::ZZ_mat<kAryType>;
using kAryElement       = Element_T<kAryValue, kAryLabel, kAryMatrix>;
using kAryList          = List_T<kAryElement>;
using kAryTree          = Tree_T<kAryList>;
#endif

using BinContainer      = BinaryContainer<n>;
using BinaryLabel       = Label_T<BinContainer>;
using BinaryValue       = Value_T<BinContainer>;
using BinaryMatrix      = mzd_t *;
using BinaryElement     = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList        = List_T<BinaryElement>;
using BinaryTree        = Tree_T<BinaryList>;

static  std::vector<uint64_t>   __level_translation_array{{0, 10, 20, 30, k}};
#endif //SMALLSECRETLWE_BEHCN_CONFIG_H
