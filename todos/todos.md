# TODO / FIXME / XXX / HACK markers

130 entries in 55 files.

| area | count |
|---|---|
| (root) | 2 |
| bench/loop_fusion | 1 |
| bench/subsetsum | 1 |
| bench/thread | 1 |
| cryptanalysislib | 3 |
| src (top level) | 17 |
| src/algorithm | 20 |
| src/atomic | 1 |
| src/container | 31 |
| src/crypto | 1 |
| src/hash | 2 |
| src/list | 2 |
| src/nn | 1 |
| src/simd | 1 |
| src/sort | 14 |
| src/thread | 17 |
| src/tree | 5 |
| tests/algorithm | 8 |
| tests/container | 1 |
| tests/thread | 1 |

## (root) (2)

| location | marker | text |
|---|---|---|
| `CMakeLists.txt:226` | TODO | port this to the cmake project |
| `README.md:171` | TODO | === |

## bench/loop_fusion (1)

| location | marker | text |
|---|---|---|
| `bench/loop_fusion/simple.cpp:21` | TODO | independe code |

## bench/subsetsum (1)

| location | marker | text |
|---|---|---|
| `bench/subsetsum/rho.cpp:18` | TODO | currently the python optimizer outputs wrong values |

## bench/thread (1)

| location | marker | text |
|---|---|---|
| `bench/thread/sum_tree.cpp:74` | TODO | port `heartbeatsum` once the heartbeat scheduler is usable again |

## cryptanalysislib (3)

| location | marker | text |
|---|---|---|
| `cryptanalysislib/main.py:22` | TODO | generate this dict from files from a subfolder: so the whole thing |
| `cryptanalysislib/optimizers/subsetsum/bcj.py:24` | TODO | (no text) next: `return False, {}` |
| `cryptanalysislib/optimizers/subsetsum/hgj.py:23` | TODO | (no text) next: `return False, {}` |

## src (top level) (17)

| location | marker | text |
|---|---|---|
| `src/element.h:196` | TODO | matrix_view? |
| `src/tree.h:1223` | TODO | test    out list |
| `src/tree.h:1492` | TODO | this function is implemented twice (ahh except for the weight. but that should be in the f function) |
| `src/tree.h:1495` | TODO | doc probalby remove and do it via the lambda function |
| `src/tree.h:1567` | TODO | only find a single solution |
| `src/tree.h:1702` | TODO | test, and filter weight, and filter function |
| `src/tree.h:2069` | TODO | tests out |
| `src/tree.h:2176` | TODO | tests,docs out |
| `src/tree.h:2411` | TODO | join2lists_on_iT_v2<k_lower1, k_upper1>(iL, L1, L2, iT, false); |
| `src/tree.h:2499` | TODO | docs/tests |
| `src/tree.h:2543` | TODO | docs/tests, remove the _f in name |
| `src/tree.h:2613` | TODO | doc, test out |
| `src/tree.h:2726` | TODO | doc img test |
| `src/tree.h:2871` | TODO | test, doc ┌───┐ |
| `src/tree.h:2992` | TODO | test, doc ┌───┐ |
| `src/tree.h:3387` | TODO | (no text) next: `lists[2].template sort_level<0, 16>();` |
| `src/tree.h:3469` | TODO | sort out list |

## src/algorithm (20)

| location | marker | text |
|---|---|---|
| `src/algorithm/matrix.h:15` | TODO | (no text) next: `inline double A[MAXN], B[MAXN], C[MAXN], D[MAXN], X[MAXN];` |
| `src/algorithm/mq/fes.h:289` | FIXME | consider getting rid of this. This function is internal ! |
| `src/algorithm/mq/fes.h:605` | TODO | to fix the issue with 10 is to greate two more kernels which only enumerate 6 or 7 variables |
| `src/algorithm/mq/fes.h:631` | TODO | iterativer revolving door ansatz |
| `src/algorithm/mq/fes.h:650` | TODO | gamma is not correct, need to proper understand it. |
| `src/algorithm/partition.h:17` | TODO | simd versions |
| `src/algorithm/pcs.h:10` | TODO | multithreading and then pcs |
| `src/algorithm/pcs.h:11` | TODO | gospers cycle finding |
| `src/algorithm/pcs.h:258` | TODO | probaly also safe the length of the walk within the distinguished point data set |
| `src/algorithm/pcs.h:264` | TODO | flavour |
| `src/algorithm/pcs.h:289` | TODO | (no text) next: `std::vector<std::future<bool>> futures;` |
| `src/algorithm/pcs.h:299` | TODO | reconstruct or whatever |
| `src/algorithm/rsa.h:12` | TODO | generic Fq numbers |
| `src/algorithm/sat/bruteforce.h:52` | TODO | write python script which unrolls everything |
| `src/algorithm/sat/cdcl.h:32` | TODO | class: with allocator |
| `src/algorithm/sat/cdcl.h:33` | TODO | add PR https://github.com/marijnheule/microsat/pulls |
| `src/algorithm/sat/cdcl.h:34` | TODO | class for literal |
| `src/algorithm/set_intersection.h:13` | TODO | simd version |
| `src/algorithm/set_intersection.h:14` | TODO | parrallel version |
| `src/algorithm/subsetsum.h:560` | TODO | factor * (Enumerator::max_list_size >> (instance.l1)); |

## src/atomic (1)

| location | marker | text |
|---|---|---|
| `src/atomic/semaphore.h:248` | TODO | std::atomic a template |

## src/container (31)

| location | marker | text |
|---|---|---|
| `src/container/binary_packed_vector.h:20` | TODO | alle funktionen gleich anordnen wie in der fq klasse |
| `src/container/binary_packed_vector.h:46` | TODO | const BinaryVectorConfig &config=binaryVectorConfig> |
| `src/container/binary_packed_vector.h:83` | TODO | config this via config |
| `src/container/binary_packed_vector.h:1325` | TODO | add to concept and add to kAryTypes |
| `src/container/binary_packed_vector.h:1720` | TODO | make this an field in the config |
| `src/container/btree_set.h:21` | TODO | add Allocator to class |
| `src/container/critbit.h:21` | TODO | move somewhere useful |
| `src/container/critbit.h:42` | TODO | static () Operator |
| `src/container/critbit.h:61` | TODO | move somewhere usefull |
| `src/container/critbit.h:192` | TODO | abstract away as class template function |
| `src/container/ctrie.h:7` | TODO | Ideas debug funcitons schreiben, die den Ctrie plotten |
| `src/container/ctrie.h:224` | TODO | enable some flags to enable/disable features |
| `src/container/ctrie.h:1033` | TODO | here is some problem |
| `src/container/ctrie.h:1116` | TODO | jump list |
| `src/container/ctrie.h:2219` | TODO | wie löschen |
| `src/container/ctrie.h:2231` | TODO | == FSNOde |
| `src/container/fq_packed_vector_v2.h:87` | TODO | (no text) next: `return 0;` |
| `src/container/fq_packed_vector_v2.h:93` | TODO | (no text) next: `}` |
| `src/container/hashmap/hopscotch.h:20` | TODO | #include "lock_free_hopscotch.h" |
| `src/container/hashmap/hopscotch.h:24` | TODO |   - new buckets in cache? |
| `src/container/hashmap/hopscotch.h:170` | TODO | generic over this |
| `src/container/hashmap/hopscotch_hash.h:1592` | TODO | use std::optional when |
| `src/container/hashmap/hopscotch_hash.h:1689` | TODO | Try to optimize the function. |
| `src/container/hashmap/hopscotch_hash.h:1817` | TODO | Remove m_buckets_data and only use a pointer+size instead of a |
| `src/container/hashmap/ska_flat.h:166` | TODO | simplify \tparam T |
| `src/container/hashmap/ska_flat.h:720` | TODO | generic |
| `src/container/imap.h:25` | TODO | remove? use __uint128? |
| `src/container/kAry_type.h:1373` | TODO | remove |
| `src/container/kAry_type.h:1398` | TODO | (no text) next: `template<typename T,` |
| `src/container/linkedlist/linkedlist.h:37` | TODO | something like this template<template<typename> typename A=std::atomic> |
| `src/container/signal_tree/tree.h:11` | TODO | type rich |

## src/crypto (1)

| location | marker | text |
|---|---|---|
| `src/crypto/sha1.h:17` | TODO | all 3 algorithms move somewhere useful |

## src/hash (2)

| location | marker | text |
|---|---|---|
| `src/hash/adler32.h:13` | TODO | move the internal horizontally addition functions to the simd interface |
| `src/hash/fnv1a.h:12` | TODO | SIMD version |

## src/list (2)

| location | marker | text |
|---|---|---|
| `src/list/enumeration/fq.h:888` | TODO | not fully correct: with this the last coordinate will not be enumerated |
| `src/list/enumeration/fq.h:1054` | TODO | also hash the 3 element |

## src/nn (1)

| location | marker | text |
|---|---|---|
| `src/nn/nn.cuh:259` | TODO | here the selection process for the correct nn algorithm |

## src/simd (1)

| location | marker | text |
|---|---|---|
| `src/simd/avx512.h:3322` | TODO | make generic |

## src/sort (14)

| location | marker | text |
|---|---|---|
| `src/sort/ips4o.h:75` | TODO | allocator + thread template |
| `src/sort/ips4o.h:84` | TODO | move somewhere usefull |
| `src/sort/ips4o.h:90` | TODO | (?): make into template to allow other key types. |
| `src/sort/ips4o.h:107` | TODO | (?): currently AVX-512 specific to avoid Gather |
| `src/sort/ips4o.h:110` | TODO | (?): adapt to base case sorter (exact match size of sorting network) |
| `src/sort/ips4o.h:219` | TODO | template<class Allocator = std::allocator<int>> |
| `src/sort/ips4o.h:250` | TODO | (?): to allow SIMD writes, we should add >=1 vector of padding to the |
| `src/sort/ips4o.h:494` | TODO | (?): for verification, we should narrow the scope/visibility of these. |
| `src/sort/ips4o.h:562` | TODO | (?): this is a bottleneck, should be replaced with SIMD writes. |
| `src/sort/ips4o.h:628` | TODO | was 16 for avx 512 |
| `src/sort/ips4o.h:702` | TODO | tail mngt |
| `src/sort/sorting_network/hunter.h:3` | TODO | improvements:  https://github.com/HJLebbink/SorterHunter (256 hannels) |
| `src/sort/sorting_network/hunter.h:4` | TODO | add a function to `Pair_t` which checks if its a simd swap pattern |
| `src/sort/sorting_network/hunter.h:7` | TODO | find until Level=91 oder so |

## src/thread (17)

| location | marker | text |
|---|---|---|
| `src/thread/old/heartbeat_scheduler.h:29` | TODO | apple doesnt support jthread |
| `src/thread/old/heartbeat_scheduler.h:32` | TODO | translate the spice binarytree example |
| `src/thread/old/heartbeat_scheduler.h:35` | TODO | use this future thingy, probably want to switch to std::future |
| `src/thread/old/heartbeat_scheduler.h:104` | TODO | implement |
| `src/thread/old/heartbeat_scheduler.h:108` | TODO | implement |
| `src/thread/old/heartbeat_scheduler.h:238` | TODO | this = Job();                  // Reset the current job. |
| `src/thread/old/heartbeat_scheduler.h:331` | TODO | correct reference |
| `src/thread/old/heartbeat_scheduler.h:335` | TODO | return std::invoke(func, &t, arg...); |
| `src/thread/old/heartbeat_scheduler.h:411` | TODO | ? |
| `src/thread/old/heartbeat_scheduler.h:484` | TODO | memory allocator |
| `src/thread/old/heartbeat_scheduler.h:489` | TODO | this seams wrong |
| `src/thread/old/heartbeat_scheduler.h:510` | TODO | better ways. |
| `src/thread/old/heartbeat_scheduler.h:552` | TODO | best_worker->shared_job->reset(); |
| `src/thread/old/heartbeat_scheduler_v2.h:159` | TODO | weak probably not needed? |
| `src/thread/old/heartbeat_scheduler_v2.h:281` | TODO | (no text) next: `using T = uint64_t;` |
| `src/thread/simple.h:293` | TODO | replace with `ConstVectorQueue` |
| `src/thread/work_contract.h:484` | TODO | remove the expected case. invoke the work function |

## src/tree (5)

| location | marker | text |
|---|---|---|
| `src/tree/d1.h:470` | TODO | test see tree.h for doc |
| `src/tree/d2_stream.h:82` | TODO | remove the weight |
| `src/tree/d2_stream.h:493` | TODO | diesen teil der function wird irgendwie hundert mal copy&pasted |
| `src/tree/d2_stream.h:520` | TODO | abstract this away? |
| `src/tree/dissection.h:4` | TODO | hashmap version, and non constexpr version |

## tests/algorithm (8)

| location | marker | text |
|---|---|---|
| `tests/algorithm/histogram.cpp:139` | TODO | `histogram_less` is unfinished (mask literal `01` instead of `b1`, |
| `tests/algorithm/mq/fes.cpp:112` | TODO | `feslite_avx2_enum_16x16_w` is unfinished (see the TODOs in |
| `tests/algorithm/pcs.cpp:44` | TODO | move to subesetsum folder |
| `tests/algorithm/pcs.cpp:247` | TODO | do something |
| `tests/algorithm/pcs.cpp:265` | TODO | return something |
| `tests/algorithm/subsetsum/join2lists.cpp:21` | TODO | non power of two prime extension are not really supported. |
| `tests/algorithm/subsetsum/join4lists.cpp:93` | TODO | not finished |
| `tests/algorithm/subsetsum/tmto.cpp:19` | TODO | simplify the enumerator interface to only need the lists as inputs and not the offset. |

## tests/container (1)

| location | marker | text |
|---|---|---|
| `tests/container/rb_tree.cpp:5` | TODO | currently dev |

## tests/thread (1)

| location | marker | text |
|---|---|---|
| `tests/thread/heartbead_scheduler.cpp:8` | TODO | not finished implementing |
