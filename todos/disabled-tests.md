# Commented-out / disabled tests

53 entries in 18 files.


## `CMakeLists.txt` (2)

| lines | test | how |
|---|---|---|
| 320 | `add_executable(test_cuda tests/test_cuda.cu)` | CMake line commented out |
| 321 | `target_link_libraries(test_cuda ${LINK_TEST_FLAGS})` | CMake line commented out |

## `bench/algorithm/copy.cpp` (6)

| lines | test | how |
|---|---|---|
| 65 | `BENCHMARK(BM_stdcopy)` | commented out |
| 66 | `BENCHMARK(BM_stdcopy)` | commented out |
| 69 | `BENCHMARK(BM_copy)` | commented out |
| 70 | `BENCHMARK(BM_copy)` | commented out |
| 73 | `BENCHMARK(BM_copy_multithreaded)` | commented out |
| 74 | `BENCHMARK(BM_copy_multithreaded)` | commented out |

## `bench/algorithm/equal.cpp` (6)

| lines | test | how |
|---|---|---|
| 72 | `BENCHMARK(BM_stdequal)` | commented out |
| 73 | `BENCHMARK(BM_stdequal)` | commented out |
| 76 | `BENCHMARK(BM_equal)` | commented out |
| 77 | `BENCHMARK(BM_equal)` | commented out |
| 80 | `BENCHMARK(BM_equal_multithreaded)` | commented out |
| 81 | `BENCHMARK(BM_equal_multithreaded)` | commented out |

## `bench/algorithm/prefixsum.cpp` (2)

| lines | test | how |
|---|---|---|
| 114 | `BENCHMARK(BM_stupid_prefixsum)` | commented out |
| 115 | `BENCHMARK(BM_prefixsum)` | commented out |

## `bench/container/ctrie.cpp` (1)

| lines | test | how |
|---|---|---|
| 50 | `BENCHMARK(BM_std_unordered_map)` | commented out |

## `bench/sort/sortingnetwork.cpp` (5)

| lines | test | how |
|---|---|---|
| 50 | `BENCHMARK(bench_stdsort)` | commented out |
| 51 | `BENCHMARK(bench_qsort)` | commented out |
| 52 | `BENCHMARK(bench_stdsort)` | commented out |
| 53 | `BENCHMARK(bench_skasort)` | commented out |
| 234 | `BENCHMARK(bench_djb_sort)` | commented out |

## `bench/thread/join.cpp` (15)

| lines | test | how |
|---|---|---|
| 45 | `BENCHMARK(stealingscheduler_create_destroy)` | commented out |
| 64 | `BENCHMARK(submit_detach_packaged_task)` | commented out |
| 83 | `BENCHMARK(submit_detach_void_lambda)` | commented out |
| 103 | `BENCHMARK(submit_void_lambda)` | commented out |
| 123 | `BENCHMARK(submit_void_lambda_future)` | commented out |
| 144 | `BENCHMARK(submit_int_lambda_future)` | commented out |
| 160 | `BENCHMARK(run_1k_packaged_tasks)` | commented out |
| 175 | `BENCHMARK(run_1k_void_lambdas)` | commented out |
| 191 | `BENCHMARK(run_1k_int_lambdas)` | commented out |
| 280 | `BENCHMARK(BM_WorkContract)` | commented out |
| 285 | `BENCHMARK(BM_pthread)` | commented out |
| 286 | `BENCHMARK(BM_Simple)` | commented out |
| 287 | `BENCHMARK(BM_Stealing)` | commented out |
| 288 | `BENCHMARK(BM_jthread)` | commented out |
| 289 | `BENCHMARK(BM_stdthread)` | commented out |

## `tests/algorithm/histogram.cpp` (1)

| lines | test | how |
|---|---|---|
| 141 | `TEST(avx512_hist256_2, DISABLED_single)` | DISABLED_ prefix |

## `tests/algorithm/mq/fes.cpp` (1)

| lines | test | how |
|---|---|---|
| 115 | `TEST(mq, DISABLED_weight)` | DISABLED_ prefix |

## `tests/algorithm/pcs.cpp` (1)

| lines | test | how |
|---|---|---|
| 47-53 | `TEST(PCS, RhoSubSetSum)` | commented out — "`walk_len`=8 steps only rarely ends in a golden collision." |

## `tests/algorithm/search.cpp` (1)

| lines | test | how |
|---|---|---|
| 56-59 | `TYPED_TEST_P(Search, multithreading)` | commented out |

## `tests/algorithm/subsetsum/join8lists.cpp` (1)

| lines | test | how |
|---|---|---|
| 34-39 | `TEST(SubSetSum, join8lists)` | commented out — "`iL` with L1, L2 instead of L3, L4, and L[2], L[6] are never used." |

## `tests/algorithm/subsetsum/parameters.cpp` (2)

| lines | test | how |
|---|---|---|
| 65-67 | `TEST(SubSetSum, n32_d2_baselists4)` | commented out — "four of them never reach the output weight n/2 = 16" |
| 98-101 | `TEST(SubSetSum, n48)` | commented out — "the output weight n/2 = 24, and p = 6 (190051 elements per list) is too slow" |

## `tests/container/rb_tree.cpp` (1)

| lines | test | how |
|---|---|---|
| 21 | `TEST(RBTree, first)` | inside `#if 0` (line 6) |

## `tests/nn/full_length.cu` (2)

| lines | test | how |
|---|---|---|
| 447 | `TEST(FullLengthBench, DISABLED_filter)` | DISABLED_ prefix |
| 457 | `TEST(FullLengthBench, DISABLED_join)` | DISABLED_ prefix |

## `tests/nn/tensor.cu` (2)

| lines | test | how |
|---|---|---|
| 449 | `TEST(TensorBench, DISABLED_bench)` | DISABLED_ prefix |
| 478 | `TEST(TensorBench, DISABLED_tune)` | DISABLED_ prefix |

## `tests/sort/sorting_algorithms.cpp` (1)

| lines | test | how |
|---|---|---|
| 31-36 | `TYPED_TEST_P(TestSort, CountingSort)` | commented out |

## `tests/thread/scheduler.cpp` (3)

| lines | test | how |
|---|---|---|
| 191-193 | `TEST(Thread, TaskExceptionDoesntKill)` | commented out |
| 272-275 | `TEST(Thread, CreateFewerThread)` | commented out — "__cpp_lib_jthread" |
| 277-279 | `TEST(Thread, CreateFewerThreadComplete)` | commented out |
