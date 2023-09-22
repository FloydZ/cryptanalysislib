#ifndef SMALLSECRETLWE_BINARY_H
#define SMALLSECRETLWE_BINARY_H

#define SORT_INCREASING_ORDER

#include "element.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "matrix/fq_matrix.h"
#include "tree.h"

constexpr uint32_t n  = 100;
using BinaryValue     = BinaryContainer<n>;
using BinaryLabel     = BinaryContainer<n>;
using BinaryMatrix    = FqMatrix<uint64_t, n, n, 2>;
using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList      = List_T<BinaryElement>;
using BinaryTree      = Tree_T<BinaryList>;

static std::vector<uint64_t> __level_translation_array{{0, 5, 10, 15, n}};
// DO NOT USE THIS
#ifdef EXTERNAL_MAIN
#define TESTSIZE 100

//TODO not available for n=100 #include "build_tree.cpp"
#include "container.cpp"
#include "container_avx.cpp"
#include "container_cmp.cpp"
#include "list.cpp"
// TODO not available for n=100 #include "tree.cpp"
#endif

#endif //SMALLSECRETLWE_BINARY_H
