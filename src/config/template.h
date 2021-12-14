//
// Created by FloydZ on 05.08.20.
//
#ifndef SMALLSECRETLWE_TEMPLATE_H
#define SMALLSECRETLWE_TEMPLATE_H

#include <vector>
#include <array>
#include <cstdint>

// make sure that its not possible to load/create another config
#ifndef SSLWE_CONFIG_SET
#define SSLWE_CONFIG_SET

#define G_l             0u
#define G_k             0u
#define G_g             0u
#define G_n             1u
#define LOG_Q           1u
#define G_q             (1u << LOG_Q)
#define G_w             1u

// sort and searching options.
#define SORT_INCREASING_ORDER

// if nothing set 'SORT_DECREASING_ORDER' will be used
#if !defined(SORT_INCREASING_ORDER) && !defined(SORT_DECREASING_ORDER)
#define SORT_DECREASING_ORDER
#endif

// enable parallel sort/search
// #define PARALLEL
// #define SORT_PARALLEL
// #define SEARCH_PARALLEL

#if defined(PARALLEL)
#define SORT_PARALLEL
#define SEARCH_PARALLEL
#endif


// this will represent each 'bit within a 'Value' vector as a 'k_ary' number. Which means each limb will be between
// [-k, k] \mod q
#if  !defined(VALUE_KARY) && !defined(VALUE_BINARY)
#define VALUE_KARY
#endif

// The other options is:
// #define VLAUE_BINARY
// this will set the 'Value' Class to a binary based, rather a k-ary based one. So all calculations of 'Value' will be
// over $F_2^{n}$
#if  !defined(VALUE_KARY) && !defined(VALUE_BINARY)
#define VALUE_BAINRY
#endif

// IMPORTANT: this vector  __MUST__ be defined
static  std::vector<uint64_t> __level_translation_array{{0, G_n/4, G_n/2, G_n}};
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{0,0,0}} }};


#endif
#endif //SMALLSECRETLWE_TEMPLATE_H
