#ifndef TEST_INCLUDE_H
#define TEST_INCLUDE_H

#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>

#define SSLWE_CONFIG_SET
#define G_l                     0u                  // unused Parameter
#define G_k                     0u                  // unused Parameter
#define G_d                     0u                  // unused Parameter
#define G_n                     32u
#define LOG_Q                   1u                  // unused Parameter
#define G_q                     1u                  // unused Parameter
#define G_w                     1u                  // unused Parameter
#define SORT_INCREASING_ORDER
#define VALUE_BINARY

//static  std::vector<uint64_t>                     __level_translation_array{{0,  G_n}};
//constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

// Hack for testing private functions (C++ god)
//#define private public

#include "../binary.h"

#endif //SIEVER_PYX_INCLUDE_H
