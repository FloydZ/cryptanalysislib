#ifndef SMALLSECRETLWE_BINARY_H
#define SMALLSECRETLWE_BINARY_H

#define SORT_INCREASING_ORDER

#include "value.h"
#include "label.h"
#include "element.h"
#include "matrix.h"
#include "list.h"
#include "tree.h"

#define n 200
using BinaryValue     = Value_T<BinaryContainer<n>>;
using BinaryLabel     = Label_T<BinaryContainer<n>>;
using BinaryMatrix    = mzd_t *;
using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList      = List_T<BinaryElement>;
using BinaryTree      = Tree_T<BinaryList>;

// DO NOT USE THIS
#ifdef EXTERNAL_MAIN
// #include "build_tree.cpp"
#include "container.cpp"
#include "container_avx.cpp"
#include "container_cmp.cpp"
#include "container_label.cpp"
#include "label.cpp"
#include "list.cpp"
// #include "tree.cpp"
#include "value.cpp"
#endif

#include "../test.h"
#endif //SMALLSECRETLWE_BINARY_H
