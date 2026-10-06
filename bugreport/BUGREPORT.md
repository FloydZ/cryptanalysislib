# cryptanalysislib bug hunt — 2026-10-05

State examined: branch `dev` @ `5dffd79 "stuff (#35)"`, on an Apple M1 Max (arm64, NEON path), using the nix
toolchain from `shell.nix` (clang 21.1.7, libc++ 19, gtest 1.17, gbenchmark 1.9.4).

Every finding listed under "confirmed" was reproduced with a small standalone program, mostly a differential test against
`std::` or a reference implementation. Repro sources are in `repros/<area>/` (see [How to rerun](#how-to-rerun)).
Items marked *(inspection)* were confirmed by reading the code only, usually because the template cannot be instantiated.
AVX2/AVX-512 code could not be executed on this machine, so those items are "likely".

- `build-fixes.patch`: minimal workaround patch for the build blockers (B1–B4). It applies with `git apply bugreport/build-fixes.patch`.
  It is a workaround meant for testing, not a polished fix (see the notes on B4).

---

## 0. Test-suite status

| build | targets built | ctest |
|---|---|---|
| `5dffd79` as-is, fresh build dir, clang/libc++ | 23 / ~250 | — |
| with `build-fixes.patch` + `-DCMAKE_CXX_FLAGS='-Dulong="unsigned long" -fexperimental-library'` | 169 / ~250 | **143 / 247 pass**; ~76 *Not Run* (still don't compile); ~28 fail at runtime |

Runtime failures (patched build, Debug):
`test_tree`, `test_build_tree`, `binary_tree`, `binary_build_tree`, `algorithm_subsetsum_{BCJ,dissection,generic,join2lists,join2lists_multithreading,join4lists,join_stream,parameters}`,
`compression_{bwt,deflate,lzss}`, `container_stack`, `algorithm_count`, `nn_simd`, `nn_simd_bruteforce`, `simd_simd`,
`sort_sorting_algorithms` (gtest: "You forgot to list test HeapSort"), `log`, `bench_thread_join`, `bench_nn_bruteforce` (assert `d < 7`, nn.h:2352),
`b63_bench_container_ctrie` (SIGTRAP), `b63_bench_search_search` (timeout).
Asserts hit in tree/subsetsum tests (`upper <= length` in binary_packed_vector.h:383, `pos < length` :1589, `i < size()` list/common.h:587)
match the `binary_packed_vector` range bugs in §3.2. These are probably the shared root cause.

Environment notes (not library bugs):
- The existing `build/` dir fails because reflect-cpp's PCH was generated in a nix shell with different hardening flags
  (`_LIBCPP_HARDENING_MODE_EXTENSIVE` vs `_FAST`). A fresh dir with `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` avoids it.
- nix libc++ 19 needs `-fexperimental-library` for `std::stop_token`/`jthread`.
- ASan binaries hang in this sandbox, even on hello-world. UBSan plus `DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib` (Guard Malloc)
  and `-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG` were used instead.
- nix sets `NIX_ENFORCE_NO_NATIVE`, so `-march=native` is silently dropped.
- The `deps/b63` submodule is not checked out in this working copy: it contains only its `.git` pointer, and `git status` shows
  41 staged deletions inside it. That alone breaks every `bench_b63_*` target. Fix: `git submodule update --init deps/b63`.

---

## 1. Build blockers (highest leverage)

| # | where | problem |
|---|---|---|
| B1 | `simd/neon.h:3278-3311` | `Xint32x8_t::lt` / `lt_` defined twice (added in 5dffd79). This is a hard error on any compiler and breaks ~120 ARM targets. Fix: delete the duplicate. |
| B2 | `atomic/atomic_primitives.h:250-390` | `__atomic_impl` copies libstdc++ internals (`_GLIBCXX_ALWAYS_INLINE`, `std::__conditional_t`, `std::__addressof`, `std::__atomic_wait_address_v`, `std::__atomic_notify_address`), so it fails with libc++. `atomic/latch.h` has the same problem (`std::__detail`, `std::__atomic_wait_address`, unqualified `memory_order`, `__atomic_impl::fetch_sub` is commented out). |
| B3 | `helper.h` | 5dffd79 deleted the `CUSTOM_PAGE_SIZE` / `HPAGE_SIZE` macro block (it was at helper.h:135-146 in d023b29), but `matrix/binary_matrix.h:180`, `nn/nn.h:300`, `sort.h:503` still use it. This fails on **every** platform. Restoring it also requires deleting the local `const uint64_t CUSTOM_PAGE_SIZE = 1u<<13` in `alloc/alloc.h:70`, which would otherwise be macro-expanded into a syntax error. That local also used the wrong page size (8 KiB) for the `/proc/self/pagemap` offset. |
| B4 | `thread/performance.h:4` | The whole file is inside `#ifndef __APPLE__`, so `details::`, `SchedulerConfig` and `SchedulerPerformanceManager` vanish on macOS and all of `thread/`, `list/list.h` and `list/parallel.h` break. The inner `#ifdef __APPLE__ → std::thread` branch is dead code. The only real Linux-ism is `RUSAGE_THREAD` (:78). The patch uses `#if 1` plus an `RUSAGE_THREAD→RUSAGE_SELF` fallback; a proper fix should guard only the rusage part. |
| B5 | `container/queue.h:191`, `search/binary.h:208,270,294,324,356,362`, `sort/radixsort.h` | `ulong` is a glibc-only typedef. Use `unsigned long`/`size_t`. (Not in the patch; worked around with `-Dulong=...`.) |
| B6 | `loop_fusion/runtime/looper.hpp:20,24`, `looper_union.hpp:19,24` | The constructor initializes `rng`, but the member is named `rang`. |
| B7 | `compression/lzf.h:128` | Leaks an `expect` macro that breaks `reflection.h`. |

Other headers that still don't compile once instantiated (test/header drift, likely on Linux too):
- `algorithm/accumulate.h:106`: assigns to a `const init` parameter. This breaks the op-overload and parallel `exclusive_scan`. `:73` passes an iterator where a `const T*` is expected, so `accumulate(v.begin(), v.end(), x)` never compiles.
- `algorithm/min.h:100`: unqualified, nonexistent `min_simd_uXX`. Tests in `tests/algorithm/{min,max}.cpp` call `min_simd_uXX`/`max_simd_uXX`.
- `algorithm/argmax.h:186`: calls `internal::argmax`, which is declared only. `argmin.h:354` only accepts `const uint32_t*`.
- `algorithm/find.h:380`: `cryptanalysislib::search(...)` names a namespace.
- `algorithm/minmax.h:49,64`: `S::loadu`/`storeu` don't exist.
- `algorithm/partition.h:3`: includes the nonexistent header `<pair>`.
- `algorithm/set.h:51,142`: missing the `template` keyword.
- `algorithm/search.h:127,214`: wrong `regular_invocable` constraint.
- `algorithm/int2weight.h:114`: wrong argument count.
- `algorithm/histogram.h:1035`: calls the nonexistent `enqueue`.
- `algorithm/transform.h:187`: broken unary `transform_reduce`.
- Four parallel `mismatch` overloads: wrong `parallel_chunk_for_2` arguments and no return value.
- `container/deque.h`: `Allocator::allocator(...)` should be `allocate`; `swap2` is undeclared; non-static calls.
- `container/ringbuffer.h` (`allocator.allocator`) and `container/heap.h` (wrong `heapify` arity; `tests/container/heap.cpp` uses the wrong argument count).
- `sparse_table.h`, `binary_indexed_tree.h` (vector ctor), `segment_tree.h`, `Heap2`: undefined `pb`/`sz` macros, or all members private.
- `priorityqueue.h`: includes the missing `fxttypes.h`.
- `const_array::begin() const`.
- `container/avl_tree.h:388`: invalid template template argument.
- `combination/revolving_door.h:141-142`: no matching `min`/`max`.
- `thread/work_contract.h:286,501`: missing `<exception>`.
- `tests/thread/thread.cpp`: `mythread_*` and `pthread_t::tid` are Linux-only.
- `tests/math/avg.cpp`: `floor_average`/`ceil_average` don't exist.
- `tests/simd/swar.cpp` / `simd/swar.h:13`: `LogTypeTemplate<T>` is given a type, and `broadcast` doesn't exist.
- `math/crt.h`, `miller_rabin.h`, `tonelli_shanks.h`, `primitive_root.h`: `mod_pow`/`mod` are undeclared, `vector::pb` is used, and `crt` calls `EEA` with the wrong signature. Their tests are empty, so nothing notices.
- `sort/selectionsort.h`, `sort/quicksort.h` (take `const Type*` and swap through it; `if (8)` at quicksort.h:43 always picks selection sort), `merge_sort.h:68,112` (discards the `allocate()` result; `merge_sort4` calls the wrong recursion), `heapsort.h:19` (`heapify(p,n,1)` should use `m`; nonexistent `Heap(x,n)`/`swap2`), `radixsort.h:38` (`calloc(1, nb)` allocates bytes, not `nb*sizeof(size_t)`).
- Several sources include x86-only `<immintrin.h>`/`<mmintrin.h>` unconditionally, which fails on ARM.
- `reflection/reflection.h:363,367,974-1061` (vendored qlibs/reflect): fails with clang 21.

---

## 2. SIMD (`src/simd/`) — repros: `repros/simd/`

| # | where | defect → observed vs expected |
|---|---|---|
| S1 | `neon.h ~3409` `uint32x8_t::move` | second half shifted `<< i*8` (should be `i*4`). All-MSB gives 0xf0f instead of 0xff; lanes 0 and 4 give 0x101 instead of 0x11. |
| S2 | `neon.h ~4075-4086` `uint64x4_t::eq`, `~4139` `move` | `<< i*4` (should be `i*2`). All-equal gives 0x33 instead of 0xf; lane 2 gives 0x10 instead of 0x4. `operator==` and `find<u64>` are wrong (match at index 2 reports 4). |
| S3 | `neon.h 2688, ~3405, ~4135` `cmp()` packing (16x16, 32x8, 64x4) | high half always `<<16`. 16x16 gives 0xff00ff instead of 0xffff; 32x8 gives 0xf000f instead of 0xff. |
| S4 | `neon.h Xint8x32 eq / operator==` | returns signed `int`, so 0xffffffff becomes -1, and `popcount` sign-extends it to 64 bits. `count_uXX_simd<uint8_t>` on 100 ones gives **196** (this is the `algorithm_count` test failure). |
| S5 | `neon.h ~3847` `uint64x4_t::srli`, ~3837 | `vshlq_u64` with a positive count shifts **left**; `assert(in2<=8)` should be `<=64`. 8>>1 gives 0x10 instead of 4. |
| S6 | `neon.h 505` `_Xint8x16_t::srli` | same left-shift bug. 0x10>>1 gives 0x20 instead of 0x08. |
| S7 | `neon.h ~3448, ~4201` `popcnt` (32x8, 64x4) | upper partial sums are never masked. 0xffffffff gives 0x100020 instead of 32. |
| S8 | `neon.h 2619` `uint16x16_t::le_` | uses `vcltq_u16` instead of `vcleq_u16`. le_(7,7) gives 0. |
| S9 | `neon.h` signed 256-bit compares (8/16/32/64) | always use `_u` intrinsics. `int8x32_t::gt(-1,1)` gives true. |
| S10 | `neon.h 2118, 2824, 3563` `set`/`setr` (16x16, 32x8, 64x4) | lane order is the reverse of `uint8x32_t` and of the AVX2 backend. |
| S11 | `neon.h 712` `_Xint8x16_t::reverse` | `vrbitq_u8` reverses bits within each byte, not the byte order. |
| S12 | `neon.h ~650` `_Xint8x16_t::cmp_` | `ret = eq; ret ^= eq` is always 0. |
| S13 | `neon.h 752` `_Xint8x16_t::scatter` | loop stops at 8 instead of 16. |
| S14 | `neon.h 1638, 2339, 3026, 3745` 256-bit `andnot` | computes NAND `~(a&b)`, while `_Xint8x16_t::andnot` computes `~a&b`. |
| S15 | `neon.h 1420` | signed `set`/`set1` (`int8x32_t`, …) don't compile, because `u8tom128` only takes unsigned pointers. |
| S16 | `simd.h 5317-5400` `_uint16x8_t/_uint32x4_t/_uint64x2_t::operator=` | writes into a local and never modifies `*this`. |
| S17 | `neon.h 1748, ~2445, ~3131, ~3862` constexpr `srli` | mask `(1<<s)-1` should be `~0>>s`. The compile-time result differs from the runtime one. |
| S18 | `generic.h 751, 761` `TxN_t::move` | `bool << i` / `uint32_t << 32` is UB for more than 32 lanes. |
| S19 | `generic.h 695, 712` `gather`/`scatter` | index multiplied by `sizeof(T)` on a `T*` (scaled twice), which reads out of bounds. |
| S20 | `generic.h ~420-436` `andnot_` | computes NAND; the SIMD branches pass a scalar `in2[k]`. |
| S21 | `swar.h` `operator[]` | `T(1)<<nbits` is UB for u32/u64. |
| *likely* | `neon.h 1522/1547` (GCC branch) | `vldrq_p128(ptr128)` is missing `+ i`. |
| *likely* | `avx2.h 185/725/1230/1691` | wrong `__v` vector types (e.g. `__v16qu` for 16-bit lanes). |
| *likely* | `avx2.h Xint8x32_t::rol/ror` | identical bodies; 16-bit shifts leak across bytes. `slli` pre-mask `(1<<s)-1` is wrong. |

---

## 3. Core containers — repros: `repros/containers/`

### 3.1 `container/fq_packed_vector.h`
- `:508` `swap`: `set(i, get(j))`, but `set` takes `(data, index)`, so the arguments are reversed. `5140362514`.swap(0,3) gives `0140332514`.
- `:1126` `right_shift`: loops `j--` instead of `j++`. `0123456012`>>2 gives `2123456000`.
- `:786/805/828` `add`: `get(i)+get(i)` in `DataType` overflows for 8-bit q. q=251: 250+250 gives 244 (expected 249).
- `:907` `mul`/`scalar`: uint16×uint16 overflows in `int`. q=65521: 65520² gives 50401 (expected 1).
- `:749` `mod256_T` calls `neg_T`. q=7: mod(5) gives 2.
- q=3 `neg(l,u)` `:1467`: uses `bits_per_limb` where it should use `numbers_per_limb`. n=100 leaves 51 coordinates wrong.
- q=3 `neg<kl,ku>` `:1525`: the middle loop skips a limb, and `__data[lh]` is accessed out of bounds when `ku%32==0`.
- q=3 `add` `:1760`: the SIMD loop guard and stride are wrong (`numbers_per_*` vs `limbs_per_simd_limb`). For n≥1024 limbs 4..127 are never written; n=1100 gives 684 wrong coordinates.
- q=3 `add_only_weight_partly` `:1785`: when llimb==hlimb the limb is counted twice (`<2,10>` gives 24 instead of 5); fails to compile for l,h≥32.
- q=3 `filter2count_range_T` `:1862`: missing `~` on the lower mask. `filter2count_T<k>` `:1874`: mask is 0 when k%32==0; the `uint8_t` default type is too narrow.
- *likely*: `:1746` `c2` constant built with `<<` instead of `|`; `:318` `accessMask` missing the `* bits_per_number` factor. `fq_packed_vector_v2.h` looks unfinished (empty `set`).

### 3.2 `container/binary_packed_vector.h`
- `:254/258/262` `set_bit`/`flip_bit`/`clear_bit`: `T(1)<<pos` without `% RADIX`. UBSan reports "shift exponent 70".
- `:399, :498, :523` `random(l,u)` and `is_zero(l,u)` (runtime and template): the middle loop stops at `upper_limb-1` and writes `__data[lower_limb]` instead of `__data[i]`. Accesses `__data[upper_limb]` out of bounds when `u==n` and `n%64==0`. `random(0,200)` gives popcounts 22/0/0/5; `is_zero(0,192)` returns true with bit 150 set. **Likely root of the tree/subsetsum test failures.**
- `:282/315` `zero(l,u)`/`one(l,u)`: out of bounds when `u==n` and `n%64==0`.
- `:933` (and `:1179` `mul`): `v3[i]=v1[i]^v2[i]` uses the **bit** `operator[]` instead of limb access. `add<0,256>` gives 46 wrong bits.
- `:903` `add<kl,ku,norm>` returns `bool(weight)` instead of `weight >= norm`.
- `:1009` `add_weight(FqPackedVector v3, …)` takes v3 by value, so the result is lost.
- `:1264` `slr` clears `[s,n)` instead of `[n-s,n)`.
- `:1212` `scalar`: missing return on the zero branch.

### 3.3 `container/fq_vector.h`, `kAry_type.h`, `element.h`
- `fq_vector.h:149` `minus_one` stores `T(-1)` instead of `q-1` (gives 255 for q=7).
- `fq_vector.h:559` `rol` drops wrapped elements. `rol(01234560,3)` gives `00001234`.
- `fq_vector.h:94` (and `:72` constexpr) `hash`: computes `1ull<<64` when `(h-l)*qbits==64`. q=4: `hash(0,32)` returns 0.
- `kAry_type.h:202` `addmul`: `% q` applied only to the product. q=7: 5+3·4 gives 10.
- `kAry_type.h:978/983` `add_T`/`sub_T`: not widened to `T2`. q=2³²−5 gives the wrong result.
- `element.h:259, :311` `Element::sub` (runtime and template) calls **`Value::add`**, so value = e1+e2 while label = e1−e2.
- *likely* (non-arith mode): `kAry_type` `random(l,u)` uses `==` instead of `=`; `mul` uses `^`; `rol_T`/`ror1_T` don't return a value.

---

## 4. Sorting & search — repros: `repros/sort_search/`

- `sort/sorting_network/sorting_network.h:41,48` (non-x86 `int32/uint32_MINMAX`): `b = a > b ? tmp : b` compares against the already-overwritten `a`, so the max is lost. `MINMAX(5,3)` gives (3,3), which means **all sorting networks are broken on ARM**. `sortingnetwork_sort_{i,u}32x8` also call `MINMAX(x1,x0)`, giving descending order.
- `sorting_network.h:59` `sort_minmax_branchless`: the `(long)d` mask zero-extends narrower unsigned types.
- `sort/common.h:16-17` `hoare_partition`: the pivot index is stored in `T` and the swap value in `int`, which destroys values. Doesn't compile for `double`.
- `sort/float_radixsort.h:121` `RadixSort::Sort(const uint32_t*)`: pass 3 reuses pass 2's offsets. Wrong for every n≥3.
- `sort/robinhoodsort.h:164`: `while (aux[--sz]==s)` reads `aux[-1]`. `:94,116`: range and position are truncated to u32, which crashes for 64-bit keys. `rhsort32(n=0)` reads `x[0]`.
- `sort/vv_radixsort.h:56,74`: the `static` buffer is freed after every call, so the second call is a **use-after-free**. Its size is fixed by the first call, and the `realloc` size is `len/8`.
- `search/binary.h:38…605`: every `n<=1` early return yields 0 ("found"). `bsearch({5},3)` gives 0 instead of 1; `standard_binary_search({5},3)` gives 0 instead of -1.
- `binary.h:217,247` `bsearch_approx`: returns an index relative to `f+k`, and `bsearch_leq` assumes a descending array.
- `binary.h:719` `lower_bound_monobound`: `{10,20},10` gives 1. `:671` `upper_bound_monobound` is wrong on absent and last keys.
- `binary.h:796` `tripletapped_binary_search`: count is wrong, so present keys are missed.
- `binary.h:397` `Khuong_bin_search`: reads `low[len_list]` (out of bounds) and misses present keys.
- `search/linear.h:49` `upper_bound_linear_search`: `while(--count)` never checks `first`. `:92` `lower_bound_linear_search` computes upper_bound.
- `search/interpolation.h:44-75` (`interpolation_search`): **infinite loop** for absent keys under NDEBUG; divides by 0 for n=1 or all-equal input. `:338` returns `begin` for an absent key; `:248,311` cast inf/NaN to an integer.
- *likely*: `interpolation.h:212` `new_iter` is never advanced; the monobound iterator searches call `advance(top,-1)` on an empty range.
- Passed: StaticSort/StaticTimSort for N=1..64, vergesort, gfx::timsort, rhsort for ≤32-bit types, float RadixSort, `branchless_lower_bound`.

---

## 5. Algorithm / math / combination — repros: `repros/algorithm_math/`

**math**
- `gcd.h:105-113` `gcd_binary` (the default `gcd`): 32-bit `__builtin_ctz` and an `int` diff for any T. `gcd<u64>(2^33,3·2^33)` gives 2^32; `gcd<int>(-4,6)` hangs. `gcd_recursive_v0(5,0)` gives 0.
- `bc.h:22` `bc()`: the intermediate overflows for n≥63 even where the result fits in u64 (68 wrong values).
- `mod.h` `fastmod<negative d>` gives garbage; `fastdiv<1u>` gives 0.
- `ceil.h:27` `cceil(3e9)` gives 3000000001 (int32 cast). `round(2.7)` gives 2. `log.h` `ceil_log2(UINT64_MAX)` gives 1. `next_prime(0|1)` returns 0/1.

**algorithm**
- `equal.h:38`: returns `memcmp(...)`, so the result is **inverted**. The parallel version reuses `first2` for every chunk.
- `mismatch.h:46`: inverted mask test, so a mismatch is missed unless every lane differs. `find.h:57`/`mismatch.h:53`: `ffs<T>` truncates the u8 mask.
- `count.h:126`: `popcount(int)` sign-extends (see S4).
- `min.h:46` `min_simd_uXX` starts its accumulator at 0, so it always returns 0. The `min`/`max` scalar path mixes value and index (`max({1,2,3,4,0})` gives 3). `max_simd` starts at 0, which is wrong for signed types.
- `argmax.h:141` `argmax_simd_bl32`: `gt` arguments reversed. The bl32 variants read `a[0..30]` when n<32.
- `prefixsum.h:318-322` `inclusive_scan(first,last,d,op)`: off-by-one, so the last element is never written. The init overload computes `op(x0,init)`.
- `reduce.h:42,66`: an empty range returns `T{}` instead of init; the parallel version adds init once per chunk (+1).
- `random_index.h:33`: hangs (missing return; the size check ignores `min_entry`).
- `shift.h:52`: a negative shift does an overlapping forward move (`12345` gives `12121`). `:118/199` `shift_right` returns the wrong index.
- `rotate.h`: helpers are hard-coded to `int*`; passes `n*sizeof(int)` to the element-count `memcpy`, which overruns. `:109/133` `rotl<u8>` is unmasked and UB for k=0. `:619` is ambiguous with std.
- `histogram.h`: 32 bins, counters never zeroed.

**combination**
- `chase.h:201` `changelist`: `resize(listsize==0)` followed by writes, so it **writes past the end of the heap buffer** (also hits `tests/list/enumeration/fq.cpp`).
- `chase.h:404-436` `enumerate3`: `<7,3>` gives 42 changes instead of 34, with out-of-range indices.
- `colex.h:42,53`: `~0UL >> (BITS-k)` is wrong for u32. `colex<u32,10,3>` yields 2 vectors instead of 120.
- `shifts.h:72`: hangs for `<5,3>`. `lexicographic.h:98`: no p==4 branch (0 of 210). `lexicographic.h:319` `negidx2lexrev`: hangs for k=4.
- *likely*: `revolving_door` k=0 segfaults; parallel argmin/argmax treat chunk-relative indices as absolute.

---

## 6. Data structures, lists, hashes, memory — repros: `repros/datastructures_hash_memory/`

**memory**
- `memory/memcpy.h:93`: the tail count uses the modified `t`. With the destination misaligned, up to 17 tail bytes are dropped (1052 (offset,n) cases fail vs `std::memcpy`).
- `memory/memset.h:149`: the half-store goes to `end-N` instead of `end-N/2`. `fill<u16>` n=16 leaves elements 8..15 unset. `:73` (GCC jump table) `M02` casts to `uint8_t`.
- `memory/memcmp.h ~224-244` (AVX2): the 8/4/2/1-byte tails return "different" when the bytes are equal (checked under Rosetta).

**hashes**
- `hash/adler32.h:61-62`: `uint16_t a,b` wrap before the mod (n=37 gives 0x213a11d2, zlib gives 0x214911d2). The AVX2 path `:125-167` has no final reduction and no NMAX chunking. *likely*: the AVX-512 path at `:211` advances by `i*32` instead of `i*64`.
- `hash/komihash.h:77-78`: `KOMIHASH_EC32/64` always byte-swap, so every reference vector differs.
- `compression/leb128.h:76-83` (used by `SimpleCompressedHashMap`): `max_shift` is too small and the shift is done on `int`. u16 300 gives 44; u64 2^40 gives 256.
- Passed: CityHash, xxh3 (vs 0.8.3), crc32, fnv1/fnv1a.

**hashmaps**
- `hashmap/simple.h:288-289` `clear(tid)`: byte ranges are truncated, so the last bucket is never cleared.
- `hashmap/simple2.h`:
  - `:102` multithreaded `FAA` is never clamped, so a bucket overflows (load 20 vs capacity 7).
  - `:256` `load()` calls `load(i)`.
  - `:169/181` `HashFkt` is undeclared, so `find()` doesn't compile.
  - `:173` returns `index*nrbuckets` instead of `index*bucketsize`.
- `hashmap/simple_compressed.h:131`: `decompress` drops the last element. `:97-98`: the full check never fires, and an overflowing bucket wipes its neighbor.
- Passed: hopscotch_map (200k-op differential vs `std::unordered_map`).

**lists** (`list/`)
- `list/common.h:365-370` `MetaListT::copy`: passes a byte count to the element-count `memcpy`, which **overflows the heap buffer**. `s`/`c` are wrong; `set_threads` zeroes the load just copied.
- `list/common.h:522`: lists constructed with `init_data=false` give `start_pos(tid)==0` for every tid (`Parallel_List_IndexElement_T`, `Parallel_List_T`).
- `list/common.h:479` (same in `list/simple.h:56`): `size(last tid)` returns the whole list. `:772` `random(m,tid)` does nothing. `:440` `is_sorted` skips the last element.
- *(inspection)* `list/parallel.h:211` `sort<Hash>` end==start; `parallel_index.h:121` adds an element index as a byte offset.

**other containers**
- `container/stack.h`: `push()` returns the capacity instead of the size; `capacity()` returns the size; `grow()` bumps `s_` without reallocating, so it **overflows the heap** when `growSize != 0`. The pop loop in `tests/container/stack.cpp` is also wrong.
- `container/linkedlist/linkedlist.h` (FreeList): `:153` iterating {1,2,3} yields 0 1 2; `remove` never decrements `__size`; `:161` `insert(0)` dereferences null; `:166` the int sentinel `-1` crashes on `insert(5)`.
- `container/linkedlist/const_linkedlist.h:157-165` `clear()`: frees nodes but leaves `head` pointing at them, giving a **use-after-free and double free**.
- `container/queue/spsc_fixed_queue.h:19-22`: `begin()`/`end()` use unmasked indices, so it segfaults after wrap-around. *likely*: `volatile` indices without acquire/release are not a valid SPSC queue on ARM.
- `container/vector_queue.h`: `:36` `back()` reads the wrong slot; `:56-61` the full check is off by one and overwrites the front.
- *likely*: `hash/crc.h:208` NEON `crc32` is a non-inline definition in a header (ODR violation).

**compression**
- `compression/bwt.h`: `bwt_inplace` always returns 0, and the test passes that as `n` to `bwt_reverse`, which then writes `rev[-1]` (**heap buffer underflow**). The algorithm requires NUL-terminated, `$`-terminated text (it uses `strlen`), but the test data starts with 0. `rank()` compares `uint8_t` against a signed `char`. The functions are non-inline definitions in a header (ODR violation).
- `compression_deflate` (16384 vs 16385, return -14) and `compression_lzss` (32424 vs 32768) fail at runtime; not root-caused.

---

## Suggested fix order

1. B1–B6 (build blockers), which takes the patched build from 23 to 169 targets.
2. NEON mask strides and the signed mask type (S1–S4), which should fix `algorithm_count` and likely `simd_simd`/`nn_simd*`.
3. `binary_packed_vector` range loops (§3.2), the likely root of the tree/subsetsum failures.
4. Sorting-network `MINMAX` on ARM; `memcpy`/`memset` tails; `Element::sub`; `equal`.
5. Memory-safety items: `vv_radixsort`, `const_linkedlist::clear`, `MetaListT::copy`, `Stack::grow`, `chase::changelist`, `bwt_reverse`, `spsc` iteration.

---

## How to rerun

Each `repros/<area>/` directory holds the standalone `.cpp` files plus the `build*.sh` scripts the agents used. The scripts contain
absolute paths to a session scratchpad that no longer exists, so use this command instead (from the repo root, after
`git apply bugreport/build-fixes.patch`):

```bash
clang++ -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fexperimental-library \
  -fsanitize=undefined -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG \
  '-Dulong=unsigned long' -Isrc -Ibuild/_deps/reflect-cpp-src/include \
  bugreport/repros/<area>/<file>.cpp -o /tmp/repro && /tmp/repro
# out-of-bounds detection without ASan on macOS:
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib /tmp/repro
```

Some repros take an argument selecting the sub-test (e.g. `repro_sort.cpp 1..5`, `repro_search.cpp 1..7`) or a `-D` flag
(`mr.cpp -DT_MR/-DT_CRT/-DT_TS/-DT_PR`). See the top of each file.

Full test suite with the patch:

```bash
git apply bugreport/build-fixes.patch
cmake -B build-fixed -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
  -DCMAKE_CXX_FLAGS='-Dulong="unsigned long" -fexperimental-library'
make -C build-fixed -k -j8; (cd build-fixed && ctest -j8 --timeout 300 --output-on-failure)
```
