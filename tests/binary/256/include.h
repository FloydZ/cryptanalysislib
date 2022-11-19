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

// #define private public

#include "../binary.h"

#endif //SIEVER_PYX_INCLUDE_H
