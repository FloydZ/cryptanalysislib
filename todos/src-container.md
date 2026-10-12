# TODOs: src-container

89 entries. Format: `- [ ] file:line [MARKER] text`.

## src/container/aa_tree.h

- [ ] `src/container/aa_tree.h:12` **TODO** implement everything: https://www.nayuki.io/res/aa-tree-set/aatreeset.rs
- [ ] `src/container/aa_tree.h:56` **TODO** free the stuff
- [ ] `src/container/aa_tree.h:85` **TODO** what if full
- [ ] `src/container/aa_tree.h:101` **TODO** rest — `}`

## src/container/array.h

- [ ] `src/container/array.h:12` **TODO** doc and tests

## src/container/avl_tree.h

- [ ] `src/container/avl_tree.h:24` **TODO** use allocator class

## src/container/binary_heap.h

- [ ] `src/container/binary_heap.h:5` **TODO** (no description)

## src/container/binary_packed_vector.h

- [ ] `src/container/binary_packed_vector.h:20` **TODO** alle funktionen gleich anordnen wie in der fq klasse
- [ ] `src/container/binary_packed_vector.h:46` **TODO** const BinaryVectorConfig &config=binaryVectorConfig>
- [ ] `src/container/binary_packed_vector.h:83` **TODO** config this via config
- [ ] `src/container/binary_packed_vector.h:162` **TODO** better formula
- [ ] `src/container/binary_packed_vector.h:204` **TODO** use _bextr_u64 (needs bmi2)
- [ ] `src/container/binary_packed_vector.h:868` **TODO** avx512 support
- [ ] `src/container/binary_packed_vector.h:956` **TODO** optimize with avx
- [ ] `src/container/binary_packed_vector.h:1173` **TODO** avx512 support
- [ ] `src/container/binary_packed_vector.h:1248` **TODO** missing rol/ror
- [ ] `src/container/binary_packed_vector.h:1295` **TODO** add to concept and add to kAryTypes
- [ ] `src/container/binary_packed_vector.h:1300` **TODO** simd — `for (uint32_t i = 0; i < limbs(); ++i) {`
- [ ] `src/container/binary_packed_vector.h:1689` **TODO** make this an field in the config

## src/container/bk_tree.h

- [ ] `src/container/bk_tree.h:9` **TODO** add namespace
- [ ] `src/container/bk_tree.h:10` **TODO** add concept for T, as T needs a dist functions

## src/container/btree_set.h

- [ ] `src/container/btree_set.h:21` **TODO** add namespace
- [ ] `src/container/btree_set.h:22` **TODO** add Allocator to class
- [ ] `src/container/btree_set.h:23` **TODO** add iterators to classes
- [ ] `src/container/btree_set.h:57` **TODO** iterators and stuff

## src/container/cartesian_tree.h

- [ ] `src/container/cartesian_tree.h:5` **TODO** add tests — TODO: add docs
- [ ] `src/container/cartesian_tree.h:6` **TODO** add docs — TODO: add benchs
- [ ] `src/container/cartesian_tree.h:7` **TODO** add benchs — TODO: add namespace
- [ ] `src/container/cartesian_tree.h:8` **TODO** add namespace
- [ ] `src/container/cartesian_tree.h:9` **TODO** add constexpr
- [ ] `src/container/cartesian_tree.h:10` **TODO** add noexcept

## src/container/critbit.h

- [ ] `src/container/critbit.h:15` **TODO** the insertion is not working for the first element, where only the key is inserted into the tree, but not the node.
- [ ] `src/container/critbit.h:18` **TODO** move somewhere useful
- [ ] `src/container/critbit.h:39` **TODO** static () Operator
- [ ] `src/container/critbit.h:58` **TODO** move somewhere usefull
- [ ] `src/container/critbit.h:67` **TODO** doc — `critbit_tree() noexcept : free_arg(nullptr) {`
- [ ] `src/container/critbit.h:73` **TODO** doc — inline size_t critbit_buf_keylen(const uint8_t *a) noexcept {
- [ ] `src/container/critbit.h:79` **TODO** doc — inline size_t critbit_str_keylen(const uint8_t *a) noexcept {
- [ ] `src/container/critbit.h:128` **TODO** abstract away as class template function
- [ ] `src/container/critbit.h:183` **TODO** returns? — `inline critbit_key *`
- [ ] `src/container/critbit.h:312` **TODO** `static void`

## src/container/ctrie.h

- [ ] `src/container/ctrie.h:7` **TODO** Ideas — debug funcitons schreiben, die den Ctrie plotten
- [ ] `src/container/ctrie.h:67` **TODO** define NUMBER_CPUS 2
- [ ] `src/container/ctrie.h:218` **TODO** enable — some flags to enable/disable features
- [ ] `src/container/ctrie.h:1027` **TODO** here is some problem
- [ ] `src/container/ctrie.h:1110` **TODO** jump list — `while (i < usedLength(current_)) {`
- [ ] `src/container/ctrie.h:2213` **TODO** wie löschen — `return oldsn->value;`
- [ ] `src/container/ctrie.h:2225` **TODO** == FSNOde — `} else if (txn == nullptr) {`

## src/container/dancing_links.h

- [ ] `src/container/dancing_links.h:8` **TODO** iterator — \tparam T

## src/container/fq_packed_vector.h

- [ ] `src/container/fq_packed_vector.h:117` **TODO** was damit machen nochmal?

## src/container/fq_packed_vector_v2.h

- [ ] `src/container/fq_packed_vector_v2.h:87` **TODO** `return 0;`
- [ ] `src/container/fq_packed_vector_v2.h:93` **TODO** `}`

## src/container/hashmap/growth_policy.h

- [ ] `src/container/hashmap/growth_policy.h:292` **TODO** `(void)min_bucket_count_in_out;`

## src/container/hashmap/hopscotch.h

- [ ] `src/container/hashmap/hopscotch.h:19` **TODO** #include "lock_free_hopscotch.h"
- [ ] `src/container/hashmap/hopscotch.h:23` **TODO** ranges constructor
- [ ] `src/container/hashmap/hopscotch.h:24` **TODO** - new buckets in cache?
- [ ] `src/container/hashmap/hopscotch.h:170` **TODO** generic over this

## src/container/hashmap/hopscotch_hash.h

- [ ] `src/container/hashmap/hopscotch_hash.h:1592` **TODO** use std::optional when
- [ ] `src/container/hashmap/hopscotch_hash.h:1689` **TODO** Try to optimize the function.
- [ ] `src/container/hashmap/hopscotch_hash.h:1817` **TODO** Remove m_buckets_data and only use a pointer+size instead of a

## src/container/hashmap/simple_compressed.h

- [ ] `src/container/hashmap/simple_compressed.h:37` **TODO** requires HashFunction<Hash, valueType>
- [ ] `src/container/hashmap/simple_compressed.h:114` **TODO** what happens if the old value is bigger?

## src/container/hashmap/ska_flat.h

- [ ] `src/container/hashmap/ska_flat.h:13` **TODO** doc
- [ ] `src/container/hashmap/ska_flat.h:15` **TODO** remove — `template<typename T, bool>`
- [ ] `src/container/hashmap/ska_flat.h:179` **TODO** simplify — \tparam T
- [ ] `src/container/hashmap/ska_flat.h:427` **TODO** `assert(false);`
- [ ] `src/container/hashmap/ska_flat.h:474` **TODO** implement — `assert(false);`
- [ ] `src/container/hashmap/ska_flat.h:494` **TODO** use hash — `template<class K>`
- [ ] `src/container/hashmap/ska_flat.h:502` **TODO** use hash — `template<class K>`
- [ ] `src/container/hashmap/ska_flat.h:618` **TODO** `(void) key;`
- [ ] `src/container/hashmap/ska_flat.h:734` **TODO** generic

## src/container/imap.h

- [ ] `src/container/imap.h:22` **TODO** iterators — TODO remove? use __uint128?
- [ ] `src/container/imap.h:23` **TODO** remove? use __uint128?
- [ ] `src/container/imap.h:55` **TODO** templates for bsr/popcount etc algorithm
- [ ] `src/container/imap.h:76` **TODO** move to algorithms bits
- [ ] `src/container/imap.h:90` **TODO** neon and doc

## src/container/kAry_type.h

- [ ] `src/container/kAry_type.h:31` **TODO** rename to Fqelement
- [ ] `src/container/kAry_type.h:1375` **TODO** remove — `template<const uint64_t q,`
- [ ] `src/container/kAry_type.h:1400` **TODO** `template<typename T,`
- [ ] `src/container/kAry_type.h:1430` **TODO** remove — `if (i == 15) {std::cout << " ";}`

## src/container/kd_tree.h

- [ ] `src/container/kd_tree.h:10` **TODO** custom distance function

## src/container/left_right_array.h

- [ ] `src/container/left_right_array.h:7` **TODO** actual use them for tests
- [ ] `src/container/left_right_array.h:31` **TODO** allocator — `fl_ = new size_t[n_];`

## src/container/linkedlist/const_linkedlist.h

- [ ] `src/container/linkedlist/const_linkedlist.h:137` **TODO** replace with allocator

## src/container/linkedlist/linkedlist.h

- [ ] `src/container/linkedlist/linkedlist.h:25` **TODO** allocator — if __cplusplus > 201709L
- [ ] `src/container/linkedlist/linkedlist.h:32` **TODO** something like this template<template<typename> typename A=std::atomic>

## src/container/queue.h

- [ ] `src/container/queue.h:184` **TODO** `template <typename Type>`

## src/container/signal_tree/tree.h

- [ ] `src/container/signal_tree/tree.h:11` **TODO** type rich — `using signal_index = std::uint64_t;`
- [ ] `src/container/signal_tree/tree.h:89` **TODO** hack.  write better when have time
