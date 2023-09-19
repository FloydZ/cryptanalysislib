#ifndef TEST_INCLUDE_H
#define TEST_INCLUDE_H

#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>
#define SORT_INCREASING_ORDER
#define VALUE_BINARY

//static  std::vector<uint64_t>                     __level_translation_array{{0,  G_n}};
//constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

// Hack for testing private functions (C++ god)
//#define private public

#include "../binary.h"

#endif //SIEVER_PYX_INCLUDE_H
