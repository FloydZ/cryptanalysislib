#ifndef SMALLSECRETLWE_AM_TOY_H
#define SMALLSECRETLWE_AM_TOY_H
// a small ntru example made by alex may.

// make sure that its not possible to load/create another config
#ifndef SSLWE_CONFIG_SET
#define SSLWE_CONFIG_SET

#include <vector>
#include <cstdint>

// searchspace of 115 bits.
#define G_l             0u
#define G_k             0u
#define G_g             0u
#define G_d             3u
#define G_n             107u
#define LOG_Q           10u
#define G_q             (1u << LOG_Q)
#define G_w             30u

#define SORT_INCREASING_ORDER
#define VALUE_KARY

static std::vector<uint64_t> __levels{{3, 4, 5}};
static std::vector<uint64_t> __level3_translation_array{{0, 17, 37, G_n}};
static std::vector<uint64_t> __level4_translation_array{{0, 10, 17, 48, G_n}};
static std::vector<uint64_t> __level5_translation_array{{0, 6, 23, 41, 60, G_n}};

static std::vector< std::vector<uint8_t>> __level3_filter_array{  std::vector<uint8_t>{4,0,0}, {1,0,0}, {0,0,0} };
static std::vector< std::vector<uint8_t>> __level4_filter_array{ {4,0,1}, {2,0,0}, {1,0,0}, {0,0,0} };
static std::vector< std::vector<uint8_t>> __level5_filter_array{ {4,1,1}, {2,0,1}, {2,0,0}, {1,0,0}, {0,0,0} };

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
