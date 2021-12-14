#ifndef SMALLSECRETLWE_TEST_H
#define SMALLSECRETLWE_TEST_H

#define TESTSIZE 100
#define SORT_INCREASING_ORDER

#include "value.h"
#include "label.h"
#include "element.h"
#include "matrix.h"
#include "list.h"
#include "tree.h"

#ifndef n
#define n 20
#endif

constexpr uint32_t k    = 20;
constexpr uint32_t q    = 3;
constexpr uint32_t d    = 2;
using kAryType          = kAry_Type_T<uint32_t, uint64_t, 3>;
using kAryContainer     = kAryContainer_T<kAryType, n>;
using kAryContainer2    = kAryContainer_T<kAryType, k>;
using Label             = Label_T<kAryContainer2>;
using Value             = Value_T<kAryContainer>;

#ifdef USE_FPLLL
using kAryMatrix        = fplll::ZZ_mat<kAryType>;
using Element           = Element_T<Value, Label, kAryMatrix>;
using List              = List_T<Element>;
using Tree              = Tree_T<List>;
#endif

static std::vector<uint64_t> __level_translation_array{{0, 5, 10, 15, n}};
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};
#endif //SMALLSECRETLWE_TEST_H
