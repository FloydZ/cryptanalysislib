# cryptanalysislib bug audit: branch `dev` (57ce1f9)

This audit covers branch `dev` at `57ce1f9`, done on 2026-10-07. It replaces the earlier report against `master` (`5dffd79`), which is kept as `BUGS-master-5dffd79.md`. No file under `src/` or `tests/` was changed.

## Method

- **Toolchain:** clang 21.1.8, C++23, x86-64. The CPU has AVX2 and AVX-512 (F/BW/VL/VBMI/BITALG/VPOPCNTDQ).
- **Differential sweep:** each of roughly 300 audit reproducers ran against `master` and against `dev`, built with ASan and UBSan, and the outputs were compared.
- **Targeted tests:** every function that `dev` added or rewrote was tested against a reference: the `std` algorithms, zlib, upstream komihash, `__int128`, and brute force.
- **Builds:**
  - the stock build of unmodified `dev`;
  - `dev` with AVX-512 disabled and with the one-line shim for N1, so that the test suite can run.
- **Tools and data:** scripts, tests, raw notes and sweep outputs are in `audit-dev-recheck/`.
- **Labels:**
  - **[C]** means confirmed by running code.
  - **[R]** means confirmed by reading only. This applies mainly to NEON, which cannot run here.
  - **[P]** means plausible but not verified.
- **Line numbers** refer to `dev`. Files that `dev` did not change keep their `master` line numbers.

## Build and test status

| Build of `dev` | Targets built | Compile errors | Tests passing |
|---|---|---|---|
| Stock (AVX-512 on), unmodified | 48 | 937 | 35 / 180 |
| AVX-512 off, with the one-line N1 shim | 223 | 27 | 171 / 180 |

In the stock build, 704 of the errors are blocker B1, the AVX-512 popcount builtins, and 194 are the new blocker N1.

In the shim build, the remaining compile errors are:
- `tests/simd/test_simd.h`: `t11` and `t12` are not constant expressions, so `test_simd_simd` is not built;
- `bench/algorithm/argmin.cpp` uses three removed `argmin` names;
- the `b63` benchmark submodule is missing, which is environment setup.

The 9 failing tests in the shim build:
- `algorithm_zip` segfaults (`zip.h`, still present);
- `container_ctrie` segfaults or aborts (still present);
- `algorithm_mq_fes` fails `mq.weight`;
- the tree and subset-sum joins fail: `test_build_tree`, `binary_build_tree`, `subsetsum_join4lists` and `subsetsum_parameters`;
- `container_queue_spsc` is "Not Run" because its CMake registration is broken (S6);
- `simd_simd` does not compile.

Many of the passing tests assert very little. For example, the AES, futex and pipe tests pass although those bugs are still present.

---

## 1. Build blockers

- **N1 (new) [C]** `src/alloc/alloc.h:116`: `const uint64_t HPAGE_SIZE = 1u<<13;`.
  - `dev` now defines the macro `HPAGE_SIZE` in `helper.h`, so this line expands to `const uint64_t (1 << 21) = …` and fails with "expected unqualified-id".
  - Every file that includes `alloc.h` fails to compile, which covers most of the library.
  - The fix is to delete the line. The local constant is wrong anyway: it is 8 KiB where a huge page is 2 MiB.
- **B1 [C]** `src/simd/avx512.h:696, 1545, 2175, 2828` (file unchanged): these use `__builtin_ia32_vpopcnt{b,w,d,q}_512`, which clang 21 does not provide. Every file that includes `simd.h` on an AVX-512 host fails to compile.
- **B4 [C]:** a global `using namespace cryptanalysislib;` sits in 18 headers, on `master` too. The earlier report listed only 6.
  - The headers are: `algorithm/bits/pdep.h`, `algorithm/bits/pext.h`, `algorithm/random_index.h`, `algorithm/rsa.h`, `algorithm/sat/bruteforce.h`, `container/fq_vector.h`, `container/imap.h`, `container/kAry_type.h`, `list/enumeration/random.h`, `matrix/binary_matrix.h`, `nn/limb.h`, `nn/nn.h`, `simd/avx2.h`, `simd/generic.h`, `simd/simd.h`, `sort/counting_sort.h`, `sort/sorting_network/hunter.h` and `thread/mythread.h`.
  - Effect, verified: after including the sort headers, an ordinary user call `memcpy(p, q, n*4)` binds to `cryptanalysislib::memcpy<uint32_t>`, which takes an element count, and ASan reports a heap overflow.
  - Inside the library the same mechanism breaks `rhmergesort` (see S3).
- **B5 [C]:** in a test that includes each header from two translation units, only `thread/mythread.h` still fails to link. It defines `mythread_q_search(int)`, `mythread_idle(void*)` and `debug_futex` non-inline.
- **Include guards:** `src/compression/lzf.h` and `src/simd/avx2_ternary.h` still have none.
- **B6:** the submodules are not checked out, so cmake fails without `deps/cmake_optimize_for_architecture`. This is environment setup.
- **B7:** `USE_HOST_AVX512*=OFF` is ignored. This lives in the external cmake repository.

## 2. New bugs on `dev`

- **N2 [C]** `src/algorithm/partition.h:14-150`: `is_partitioned`, `partition_point`, `partition`, `partition_copy` and `stable_partition` are in the global namespace. An unqualified call with `std` iterators is ambiguous with `std::`; verified: "call to 'stable_partition' is ambiguous". The functions themselves are correct.
- **N3 [C]** `src/algorithm/set.h:64`: `set_difference` on `uint16_t` data segfaults. The new SIMD set code now reaches the old aligned-store bug S1-#10. The u8, u32 and u64 set operations are correct.
- **N4 [C]** `src/thread/work_contract.h`: `stop()` now wakes waiters, but teardown still hangs in other phases. A phase reproducer hangs at iteration 144, and a destructor reproducer hangs. `test_thread_work_contract` itself passes.
- **N5 [C]** pipe reproducer: `audit/t_pipe` now times out on `dev`; it finished on `master`. `atomic/pipe.h` is unchanged, but `atomic_primitives.h` changed. The pipe's reversed CAS (T5) is still present.
- **N6 [C]:** the library's global `adler32()` and `crc32()` collide with zlib. Including `<zlib.h>` before `compression/deflate.h` gives "call to 'adler32' is ambiguous" at `deflate.h:726`.
- **N7 [C]** `tests/container/queue/CMakeLists.txt:8`: `add_test` registers `test_container_hashmap_${file}`, which should be `test_container_queue_${file}`, so the SPSC queue test never runs. This is also on `master`. The binary passes when run by hand.
- **N8 [R]** `src/search/interpolation.h:80`: `lower_bound_interpolation_3p_search` requires only `std::forward_iterator` but uses `last - first` and `first[pos]`, so it does not compile with true forward iterators.

## 3. Still present on `dev`

### S1 SIMD

- **`avx2.h`:**
  - `:1128` `_Xint16x8_t::all_equal` returns a vector from a `bool` function, a compile error.
  - Scatter writes only 8 lanes: 8 of 16 for `uint16x16_t`, 8 of 32 for `uint8x32_t`.
  - `:2928` `uint16x16_t::store<false>` calls `aligned_store`. It crashes on misaligned pointers, now also through `set_difference<u16>` (N3).
  - `div` is wrong: u16 gives `7/2=4` and `1000/1=64536`; u64x4 gives `100000/10=62430`.
  - `cmp` returns byte masks (`0xffffffff`) for 16-, 32- and 64-bit lanes, while `gt`, `lt` and `move` return per-lane masks.
  - `uint32x8_t::pack` and `cvtepu8` are wrong.
  - `uint64x4_t::gather` ignores `scale`.
  - `set_bit` uses `1u << pos`, which is UB for pos ≥ 32, at `:4141` (u64x4) and `:1754` (u64x2).
  - `avx2_load_f32x8` returns 2.139e9 where its documentation says +inf.
  - `permute` semantics differ between types: u16x16 scatters, u8x32 gathers.
- **`simd.h:136` [C]:** the `Mask<>` operators assign to members in `const` methods, a compile error.
- **Generic fallback [C]:**
  - `eq_` returns 1 per lane instead of `0xFF`.
  - The generic `_Xint32x4_t` and `_Xint64x2_t` have no `set1`.
- **`generic.h` [C]:** `TxN_t::cmp_` is inverted between the SIMD path and the scalar tail. `TxN_t<u32,12>::cmp_(1,2)` gives `0xffffffff` in lane 0 and `0` in lane 11.
- **Unchanged files [C]:**
  - `avx512.h`: permute, min/max, reduce, `lzcnt`, rotate, `srli`, `slli`, `cmp` widths, `div` and `test()`; master items 3.1 #24-33.
  - `avx512_intrinsics.h`: `adds`/`subs_epi32` and `setr_epi16`.
  - `bits/bits.h`: the `pdep` fallback.
  - `float/avx2.h` and `float/simd.h`: the signed or rounding conversions; `f32x8_t(3e9)` gives -1294967296.
- **NEON [R]:**
  - `neon.h:469, 471` still use `vmulq_n_u8` and `vmulq_n_s8`, which are not NEON intrinsics.
  - Unqualified `rng()` is called outside the namespace at `neon.h:1373, 2142, 2872, 3657`.

### S2 Containers

- **`binary_packed_vector.h` [C]:**
  - `hash(l,h)` for T narrower than 64 bits: `is_hashable` allows 64-bit windows that span 3 limbs, but `:1661` asserts at most 2. With `NDEBUG` it returns wrong values in 1400 to 2500 cases per size.
  - `:447` `random_with_weight(w,m,offset)` ignores `offset`, with around 2700 failures per size.
  - `random()` leaves a limb unrandomised for narrow T.
  - `:1559` `reference::get_data()` and `data()` return `bool()`.
- **`fq_vector.h` [C]:** the q=4 `mul_T` (`:1316`) collapses all lanes into lane 0.
- **`kAry_type.h` [C]:**
  - `:416` `operator-=(const T)` adds.
  - `operator=(int32_t/int64_t)` does not normalise negative values.
  - `:1051` `sub256_T` wraps before the modular reduction.
- **`fq_packed_vector.h` [C]:** `hash` on a 64-bit window triggers an assertion.
- ~~**`hashmap/simple2.h:194, 247` [C]:** `load(e)` and `find_without_hash` disagree with the real bucket load in 3511 of 4096 buckets.~~ **Not a bug on `dev` (re-examined 2026-10-08):** `find_without_hash` already uses `load_without_hash`, and the FAA counter no longer wraps. The remaining mismatch comes from the reproducer calling `load(b)` with a bucket index, but `load(e)` is documented as the load of the bucket `e` hashes into.
- **`hashmap/hopscotch_hash.h:980` [C]:** `at()` on a missing key returns `*nullptr`.
- **Hashmap concepts [C]:** `SimpleCompressedHashMap` and `Simple2HashMap` fail the `HashMapAble` concept.
- **`ctrie.h` [C]:** the test segfaults or aborts. Equal hashes cause a bad free, and the multi-threaded test leaks.
- **Unchanged files [C]:**
  - `critbit.h:293`: `remove` matches a prefix of a key.
  - `bk_tree.h`: the fake root.
  - `trie.h:619`: the index is not range-checked.
  - `imap.h`: `assign()` overflows.
  - linked lists: leaks.
  - `dancing_links.h`: leaks.
- **`binary_indexed_tree.h:27` [C]:** `update(0)` now asserts, but still loops forever under `NDEBUG`.

### S3 Sort

- **`robinhoodsort.h:37` [C]:** `memcpy(aux, a, l*sizeof(T))` binds to the element-count `cryptanalysislib::memcpy`, so `rhmergesort` overflows the heap for u16, u32, i32 and u64.
- **`float_radixsort.h` [C]:**
  - `:87` the empty destructor leaks `mIndices` and `mIndices2`.
  - `:281-282` frees `new[]` memory with plain `delete`.
- **`sort/heap/is_heap.h:53` [C]:** `is_heap_rnd` reads out of bounds.
- **`sorting_network/avx2.h:1913-2027` (unchanged) [C]:** `sortingnetwork_small_uint32_t` and `_int32_t` are wrong in 1071 and 1059 of 1100 cases. The u8x224 and odd-even networks were not rerun.
- **`sort.h:404-411` [C]:** `find_next_empty_slot` overwrites live entries.

### S4 Search

- **`interpolation.h:316` [C]:** `interpolation_search_dispatch` still returns `binary_search_dispatch`; the computed dispatch result `d` is unused. The function statics are not thread-safe.
- **API changes rather than bugs:**
  - `bsearch_leq` now requires an array sorted in descending order. It is correct under that contract.
  - `lower_bound_linear_search` and `upper_bound_linear_search` now behave like `std::lower_bound` and `std::upper_bound` with a less-than comparator.

### S5 Lists, enumeration, trees, nn

- **`list/list.h:455` `find()` [C]:** still counts 1 for 5 duplicates.
- **`list/common.h:381` `is_correct()` [C]:** returns true when one element is wrong.
- **`list/parallel_full.h:170-176` [C]:** cross-limb `sort_level` shifts by 64, which is UB.
- **`list/enumeration/binary.h:805` [C]:** the `noreps` reset hits `assert(element.value[unset])`.
- **Tree and subset-sum joins [C]:**
  - `subsetsum_join4lists` and `subsetsum_parameters` produce 0 results.
  - `binary_build_tree` produces far too many: 1047767 against a bound of 2048.
  - `test_build_tree` `join2lists` gives a mismatch.

### S6 Matrix and math

- **Binary matrix [C]:**
  - `gaus` and `m4ri` under-report pivots: `matrix_echelonize_partial` around `binary_matrix.h:1162` returned 3 where 4 leading pivots exist.
  - `:733` `sub_matrix` uses `1u << (ncols%64)`, which is UB.
- **Fq matrix [C]:**
  - `m4ri` for q=5 is not systematic in 10 of 10 runs.
  - `swap` writes the wrong cell.
  - `gaus` of the 6×6 identity returns 5.
- **Bits [C]:** `clz` and `bsr` for u32, u16 and u8 (`clz.h:20`, unchanged): `clz<u32>(1)=63`, `bsr<u32>(1)=4294967264`. The 128-bit branches are also wrong.
- **`popcount.h:25` (unchanged) [C]:** sign-extends, so `popcount<int8_t>(-1)=64`.
- **Integral `log2` [C]:** truncates: `log2<int>(8)=2`, `log2<u64>(1024)=8`, `log2<u64>(2)=0`.
- **Non-terminating math [C]:**
  - `math::log` (`log.h:28`) never terminates for 50, 999 and 1024.
  - `math::exp(120)` (`exp.h`, unchanged) never terminates.
  - `HH(0.8)` never terminates.
  - `log(0)` overflows the stack.
- **`prime.h` [C]:** `prev_prime(1)` returns 1.
- **`bigint.h:242, 258` [C]:** `operator<` compares from the low limb first, wrong in 62405 cases; `operator==` compares only `min(N,M)` limbs.
- **Algorithms [C]:**
  - `histogram<uint8_t>` overwrites counts (`histogram.h:23-29` `HISTEND`).
  - `transpose8` (`transpose.h:97`) does not mask its outputs.
  - `b8x8_be` and `b64x64` are wrong.
  - `prefixsum_i32_avx2` (`prefixsum.h:128`) reads `v[-1]` for small n.
- **`zip.h:93` [C]:** an aligned store into a 2-byte-aligned pointer makes `algorithm_zip` segfault.
- **`mq/fes.h` [C]:** `mq.weight` fails, from the early return in the weight variant.

### S7 Hashing, crypto, compression

- **`crypto/aes.h` (unchanged) [C]:** AES returns the plaintext unchanged on the NIST F.1.1 vector.
- **`hash/crc.h:74-197` [C]:**
  - `sse42_crc32` ignores the `len%16` tail and is wrong for 1065 of 1136 lengths from 64 up.
  - `:114` reads 64 bytes unconditionally, a heap overflow for inputs under 64 bytes.
  - It is still gated on the never-defined `USE_PCLMULDQD`.
- **`hash/simple.h` (unchanged) [C]:** `extract<60,70>` does not mask; `Hash<q not a power of two>` reads the wrong digits.
- **Compression [C]:**
  - `leb128` round trips fail for signed types: 49,868 failures for `int32_t`, around `leb128.h:24-45`.
  - `deflate` of empty input produces a truncated, invalid zlib stream.
  - `deflate` calls `__builtin_clz(0)` through `log.h:114`.
  - `lzss.h:111` `DataCompare` returns prefix length + 1.
  - lzss `CompressData` writes past an output buffer of exactly the input size.
  - `smaz2.h:193-213` decompression hangs, or overflows the heap when the output is too small.
  - lz77 does a misaligned store (unchanged).
  - `lzmat.h` does misaligned `uint32_t` loads and over-reads.
  - `lzf.h:185` over-reads.
  - LCP (`bwt.h:45`) reads out of bounds.
  - BWT now documents that `$` must be smaller than every symbol, so it cannot process bytes below `0x24`. This is a usage restriction.

### S8 Memory, allocators, threads, misc

- **Allocators (unchanged) [C]:**
  - `StackAllocator` deallocate overflows.
  - `STDAllocatorWrapper` treats an element count as a byte count.
  - `FreeListAllocator::deallocateAll` leaves a dangling root, a use-after-free.
  - The page allocator leaks.
  - `cache.h:93` over-copies, a stack overflow.
  - `gc_simple.h:133` mismatches alloc and dealloc.
- **Threads and atomics [C]:**
  - `atomic/futex.h:45, 59` has swapped CAS arguments: `down()` never blocks and the mutex loses updates (798344 of 800000).
  - `atomic/pipe.h` has reversed CAS arguments, so readers spin (and N5).
  - `steal.h` moves from caller lvalues, `pause()` has no effect, and a 0-thread pool drops tasks.
  - `simple.h:98` loses wakeups after `clear_tasks`, and dropped child tasks raise `broken_promise`.
  - `execution.h:259` calls `std::terminate` on task exceptions.
- **`random.h:376` [C]:** `rng_weighted` uses `1u << pos`, which is UB. `random_device(0)` asserts. The other `random.h` items were not rerun.
- **Combinatorics [C]:**
  - `necklace.h`: wrong for 8- and 32-bit types (5 of 36), `next_lyn` never terminates, and `:51` reads out of bounds.
  - `lexicographic.h:297` `negidx2lexrev` is wrong.
- **Misc [C]:**
  - `permutation.h:37`: double free.
  - `helper.h` `translate_level`: out-of-bounds read.
  - `traits.h:106-135`: the static closure copy is shared across calls.
  - `jmp/x86.h:107`: misaligned store.
  - `graph/digraph.h:38-40` [R]: the out-parameters are assigned before allocation.
  - `graph/digraph.h:51` [R]: the destructor frees `e_` with a count of 1.
  - `digraph` constructor: leak [C].
  - `bench/thread/join.cpp:297`: returns a counter as its exit code.

## 4. Fixed on `dev` (verified)

- **Build:**
  - B2 `CUSTOM_PAGE_SIZE`;
  - B3 the `lzf` `expect` macro;
  - `inline` on the 256 ternary specialisations in each file;
  - the multiple-definition link errors in `simd.h`, `math.h`, `memory.h`, `access_profiler.h` and seven compression headers;
  - the compile failures in reflection, the loop fusion runtime, latch, paren, revolving door, deque, AVL tree, ringbuffer, heap, the algorithm headers, and the `crt`, `miller_rabin`, `tonelli`, `lucas` and `primitive_root` templates.
- **Core helpers:** `memcmp` tails; the `memcpy` misaligned tail; the u16 `memset`; `count`; `equal`; the new `memmove`, tested on 36,000 overlap cases.
- **Search:**
  - All binary searches match their documented contracts in 9,000 random cases. That includes the n==1 cases, the Khuong out-of-bounds read, monobound, tripletapped, and `binary_search` equality.
  - All interpolation searches are correct in 8,970 cases.
  - The `search` namespace clash is resolved.
- **Sort:**
  - `radix_sort`, `vv_radix_sort` (including the use-after-free), `heap_sort`, `quick_sort`, `selection_sort`, `merge_sort` and `merge_sort4`;
  - `rhsort32`, including 64-bit keys;
  - the `float_radixsort` order;
  - `ips4o`, which loses no elements;
  - the scalar `uint32_MINMAX`.
- **Lists and trees:**
  - Lists: `binary_search`, `search_boundaries`, `copy()`, `random(m,tid)` and `size(tid)`.
  - Trees: the two-level stream joins, which now match the naive join exactly.
- **Containers:**
  - Binary vector: `set_bit`, `is_zero`, `zero`, `one`, `add`, `add_weight`, `mul`, `scalar`, `random(l,u)`, `slr` and `sll`.
  - Fq packed: the q=3 `neg`, `mod256_T`, `accessMask` and `filter2count_range`.
  - Fq vector: `rol` and `hash`.
  - FqElement: `addmul` and `popcnt`.
  - Compressed hashmap, the `simple2` load total and FAA wrap, and `SimpleHashMap::clear(tid)`.
  - Stack and vector queue.
  - New tests pass for heap, `priority_queue`, segment tree, sparse table and BIT.
- **SIMD:**
  - u8x16 `rol`/`ror`/`slli`/`srli`;
  - the `_Xint32x4` union size;
  - int8x16, int32x4 and int64x2 compile again;
  - int16x8 arithmetic;
  - u16x16 `andnot`, `slli(x,0)` and `eq`;
  - u32x8 `min`/`max` and `mulhi`;
  - u64x4 `max`;
  - generic `slli`/`srli`, `move` and `ror`;
  - `generic.h` `andnot`, `move` and `gather`/`scatter`;
  - NEON: the duplicate `lt`, the 256-bit `andnot`, `cmp_` and `reverse` [R].
- **Math:** `bc` overflow, `ceil_log2(2^64-1)`, `crt`, `next_prime(0/1)`, `gcd`, `round`, `cceil`, `fastmod` with negative d, and `fastdiv<1>`.
- **Hashing:** adler32 matches zlib for all 1,200 lengths; komihash matches upstream.
- **Compression:**
  - zlib accepts all non-empty deflate streams;
  - inflate decodes real zlib streams;
  - lzss round trips and no longer hangs;
  - `bwt_inplace` returns n;
  - unsigned `leb128` round trips;
  - the compressed-hashmap `leb128` desync is fixed.
- **Enumeration:** p=3 Chase, `shifts`, and narrow-type `colex`.
- **Threads:** the stealing scheduler's early return from `wait_for_tasks`; `work_contract::stop()` now notifies; `random_index` no longer hangs.

## 5. Not rechecked on `dev`

- **NEON runtime:** `neon.h` now uses intrinsics that the x86 emulation shim lacks, so the NEON runtime items were checked only by reading.
- **AVX-512 runtime:** the code is unchanged and blocked by B1.
- **Not rerun individually:**
  - the u8x224 and odd-even sorting networks;
  - the `nn.h` tails;
  - `fq_packed_vector_v2.h`;
  - the `random.h` signed ranges and seeding;
  - `gc_simple` details.

  The corresponding files are unchanged or barely changed, so their earlier findings very likely still hold.
- **`proposed-fixes/`:** these patches were written for `master` and do not apply to `dev`. They need to be regenerated against `dev`.
