#ifndef SMALLSECRETLWE_AM_TOY_H
#define SMALLSECRETLWE_AM_TOY_H
// a toy example of alex may.

// make sure that its not possible to load/create another config. So every problem dimension is hardcoded so the
// compiler can optimize the shit out of it.
#ifndef SSLWE_CONFIG_SET
#define SSLWE_CONFIG_SET

#include <tuple>
#include <vector>
#include <cstdint>

// searchspace of 83 bits.
#define G_l             0u
#define G_k             0u
#define G_g             0u
#define G_d             3u
#define G_n             107u
#define LOG_Q           10u
#define G_q             (1u << LOG_Q)
#define G_w             25u

#define SORT_INCREASING_ORDER
#define VALUE_KARY

// different maximal depth for testing
static std::vector<uint64_t> __levels{{3, 4, 5}};

// these values are calcultaed with the script 'deps/coortinates.sage'
static std::vector<uint64_t> __level3_translation_array{{0, 15, 31, G_n}};
static std::vector<uint64_t> __level4_translation_array{{0, 13, 28, 57, G_n}};
static std::vector<uint64_t> __level5_translation_array{{0, 6, 13, 28, 57, G_n}};

// List constructing and guessing complexity
// given as an array of depths from d = 3, 4, 5
static std::vector<std::tuple<int, int>> complexity{ {29, 9}, { 26, 11}, {26, 11} };

// IMPORTANT: only useful in the k-ary contract (_DVALUE_KARY). In the binary (-DVALUE_BINARY) this is ignored.
static std::vector<std::vector<uint8_t>> __level3_filter_array{ std::vector<uint8_t>{4,0,0}, {1,0,0}, {0,0,0} };
static std::vector<std::vector<uint8_t>> __level4_filter_array{ {4,1,0}, {4,0,0}, {1,0,0}, {0,0,0} };
static std::vector<std::vector<uint8_t>> __level5_filter_array{ {4,1,0}, {4,0,0}, {1,0,0}, {0,0,0}, {0,0,0} };

#if G_d == 3
static std::vector<uint64_t> __level_translation_array          = __level3_translation_array;
static std::vector<std::vector<uint8_t>> __level_filter_array   = __level3_filter_array;
#elif G_d == 4
static std::vector<uint64_t> __level_translation_array          = __level4_translation_array;
static std::vector<std::vector<uint8_t>> __level_filter_array   = __level4_filter_array;
#elif G_d == 5
static std::vector<uint64_t> __level_translation_array          = __level5_translation_array;
static std::vector<std::vector<uint8_t>> __level_filter_array   = __level5_filter_array;
#else
#error WRONG D
#endif

#endif
#endif //SMALLSECRETLWE_AM_TOY_H
