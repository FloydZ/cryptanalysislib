# cryptanalysislib bug audit (master 5dffd79, SUPERSEDED: see BUGS.md for branch dev)

Audit of `master` at `5dffd79`, done on 2026-10-07. Nothing in `src/` or `tests/` was changed. Every finding is documented here only.

## Method

- **Toolchain:** clang 21.1.8, C++23, x86-64. The CPU supports AVX2 and AVX-512 (F/BW/VL/VBMI/BITALG/VPOPCNTDQ).
- **Two builds were used:**
  - `build/` is the stock configuration, with AVX-512 detected and enabled.
  - `build2/` is the same configuration with AVX-512 disabled through a compiler wrapper, so that far more targets compile.
- **Verification:** most findings were checked with a standalone reproducer built with AddressSanitizer and UBSan, plus ThreadSanitizer for the threading code. Reproducers compare against a naive reference: `std::sort`, `std::lower_bound`, zlib, hashlib, `__int128`, naive GF(q) elimination, or brute force.
- **Labels:**
  - **[C]** means confirmed by running code or by a compile probe.
  - **[R]** means confirmed by reading only. This applies mainly to NEON code, which cannot run on this machine, and to code inside headers that do not compile.
  - **[P]** means plausible but not verified.
- **Line numbers** refer to commit `5dffd79`.

## Test-suite status

| Build | Targets built | Compile errors | Tests run | Tests failing at runtime |
|---|---|---|---|---|
| `build/` (AVX-512 on) | 37 | 1302 | 36 | 2 |
| `build2/` (AVX-512 off) | 123 | 637 | 97 | 11 |

Failing tests in `build2/` and their root causes:

| Test | Root cause | Section |
|---|---|---|
| `algorithm_count` | `popcount` sign-extends the `int` returned by `uint8x32_t::operator==` | 3.4 #1 |
| `algorithm_equal` | `memcmp` tail branches are inverted. The test also expects the inverted meaning. | 2 #3 |
| `algorithm_zip` (both builds, ~45% of runs) | aligned 32-byte store into a 2-byte-aligned `uint16_t*` | 3.4 #2 |
| `compression_bwt` | `bwt_inplace` returns 0, `rank()` uses `strlen`, and the test has no end marker | 3.7 #1 |
| `compression_lzss` | the encoder stops consuming input early and can write 4 bytes past the end | 3.7 #2 |
| `container_ctrie` (SEGFAULT) | a failed CAS overwrites the expected-node variable, then that variable is used as an array | 3.2 #14 |
| `container_stack` | `push` returns capacity instead of size. The test's `pop` expectation is also wrong. | 3.2 #16 |
| `simd_generic` | the `_Xint32x4_t` union is 64 bytes, so `TxN_t` leaves lanes 8 to 15 uninitialised | 3.1 #2 |
| `simd_simd` | the `uint8x16_t` rotate uses 16-bit shifts with a wrong mask | 3.1 #1 |
| `sort_hunter` (timeout) | the test is the SorterHunter `main` with `for(;;)` and no exit | 3.5 #7 |
| `sort_sorting_algorithms` (abort) | `HeapSort` is defined but not registered in `REGISTER_TYPED_TEST_SUITE_P` | 3.5 #8 |
| `thread_work_contract` (timeout) | `work_contract_group::stop()` never notifies the waiters, so `join` deadlocks | 3.8 #10 |
| `bench_thread_join` (`build/` only) | `main` returns the global counter `value`, so the exit code is non-zero | 3.8 #12 |

Several existing tests cannot catch the bugs they target.
- `tests/crypto/aes.cpp` only prints its output.
- The `sse42_crc32` test is gated on `USE_PCLMULDQD`, which CMake never defines.
- `tests/sort/parallel.cpp` checks only the first 128 elements.
- The FQ hash test sits behind `popcount(5)==1`, which is always false.
- The math tests for `miller_rabin`, `crt`, `lucas` and the others are empty, so the uncompilable templates are never instantiated.

---

## 1. Build blockers

These issues stop most of the library from compiling. Each one hides every runtime bug behind it.

1. **AVX-512 popcount builtins**, `src/simd/avx512.h:696, 1545, 2175, 2828` [C].
   - These lines call `__builtin_ia32_vpopcnt{b,w,d,q}_512`. Clang 19 and later no longer provide these builtins, and GCC never did.
   - Every translation unit that includes `simd.h` on an AVX-512 host fails to compile. This accounts for about 700 of the 1302 errors in `build/`.
   - Only the line 696 call has a feature guard.
   - The portable replacement is `_mm512_popcnt_epi{8,16,32,64}`.
2. **`CUSTOM_PAGE_SIZE` is undeclared**, `src/alloc/alloc.h:70` [C]. This causes 271 errors.
   - Commit `5dffd79` turned the global definition into a `const` local to `check_huge_page()`.
   - These files still use the name at namespace or class scope:
     - `matrix/binary_matrix.h:180`
     - `nn/nn.h:300-301`
     - `list/{simple,simple_limb,parallel,parallel_index}.h`
     - `sort.h`
   - As a result `element.h`, `list.h`, `matrix.h`, `nn.h`, and every test that includes them fail to compile.
3. **Leaked `expect` macro**, `src/compression/lzf.h:128/131` [C].
   - The file defines the function-like macro `expect(expr,value)` and never undefines it.
   - This breaks the one-argument `expect(...)` calls in `reflection/reflection.h:974-982`, which accounts for 90 errors.
   - `lzf.h` also leaks the macros `HLOG/HSIZE/MAX_*/FRST/NEXT/IDX`, redefines `inline`, and has no include guard.
4. **Leaked `using namespace cryptanalysislib;` at global scope** [C].
   - Locations: `simd/generic.h:13`, `matrix/binary_matrix.h:23`, `list/enumeration/random.h:13`, `algorithm/random_index.h:11`, `rsa.h:9`, `nn/limb.h:12`.
   - Unqualified `memcpy`/`memcmp` calls in library and user code then silently resolve to `cryptanalysislib::memcpy<T>` and `cryptanalysislib::memcmp`.
     - `cryptanalysislib::memcpy<T>` takes an element count, not a byte count.
     - `cryptanalysislib::memcmp` has the inverted result described in section 2, item 3.
   - This directly causes stack and heap overflows in `combination/chase.h:611`, `sort.h:966/997/1228/1246`, `robinhoodsort.h:36`, `alloc/cache.h:93`, and `list/common.h:360-371`.
5. **ODR violations: non-`inline` function definitions in headers** [C].
   - Including any of the following headers from two translation units fails to link with "multiple definition":
     - All 256 specialisations in `simd/{u32,u64,sse,neon}_ternary.h`.
     - `simd/avx2.h:4738, 4802`.
     - `simd/avx512.h:3270, 3338`.
     - `simd/avx512_intrinsics.h:1896`.
     - `simd/avx2_intrinsics.h:136`.
     - `simd/neon.h:1307`.
     - `math/mod.h:86, 96, 103, 108`.
     - `hash/crc.h:268`.
     - The compression headers `bwt.h`, `lzss.h`, `lz77.h`, `lzf.h`, `lzmat.h`, `deflate.h` and `inflate.h`.
     - `thread/mythread.h`, which alone produces 21 link errors.
     - `size.h:21`, `access_profiler.h`, `cpucycles.h:23`, and `mq/avx_16x16{,_v2}.h`.
   - `simd/avx2_ternary.h` and `lzss.h` also lack include guards.
6. **Submodules are not checked out** [C]. `cmake` fails without `deps/cmake_optimize_for_architecture`. During this audit only that submodule was initialised. `deps/b63` is still missing, which removes all `b63_*` benchmarks.
7. **`USE_HOST_AVX512*=OFF` has no effect** [C]. These cache options are force-overwritten during configure, so a user cannot disable AVX-512 to work around blocker 1.

### Other compile failures (one line each)

- **`atomic/latch.h:20,27,41`:** uses libstdc++ internals (`__atomic_impl::fetch_sub`, `__detail::__platform_wait_t`) and an unqualified `memory_order`. This also breaks `atomic/atomic.h`.
- **`algorithm/`:**
  - `accumulate.h:73` passes an iterator where a pointer is expected.
  - `accumulate.h:106` assigns to `const init`. This also breaks `exclusive_scan.h` and `transform.h`.
  - `find.h:380` resolves to the namespace `cryptanalysislib::search` instead of the function. This also breaks `all_of.h`.
  - `argmax.h:186` and `argmin.h:354` pass iterators to pointer overloads.
  - `min.h:100` uses an unqualified `internal::min_simd_uXX`. The scalar path at lines 103-109 also mixes up value and index.
  - `substr_search.h:343` binds an rvalue to a non-const reference.
  - `transpose.h:793` opens an `#ifdef USE_AVX2` that is never closed.
  - `histogram.h:1039` calls `enqueue`, which the scheduler does not have.
- **`combination/`:**
  - `revolving_door.h:141-142` calls `cryptanalysislib::max/min(a,b)`, but only range overloads exist. This also breaks `mq/fes.h`.
  - `paren.h:14,96,132` uses undeclared helpers.
  - Unported fxtbook helpers break `min_change.h`, `sequency.h`, `grey.h:132`, `revbin_upd.h:58`, `grsnegative.h:19` and `lex.h:133,170`.
  - Classes with only private members: `fibrep.h`, `grsnegative.h`, `lex.h`, `grey.h`, and the constructor at `lexicographic.h:19`.
- **`container/`:**
  - `avl_tree.h:387-388` has template parameters that do not match the class.
  - `deque.h` has errors at line 46 (missing `typename`), lines 46/53 (static allocator calls), lines 106-108 (assignment to a const reference) and line 158 (undeclared `swap2`, `n`, `f`).
  - `ringbuffer.h:34` calls `allocator(...)` where `allocate` is meant.
  - `heap.h:160,172,210` calls `heapify` with the wrong arity.
  - `Heap2` uses undefined names.
  - `vector.h:14` calls the zero-argument `allocate()`.
  - Headers that do not compile when instantiated: `priorityqueue.h`, `queue.h`, `kd_tree.h`, `cartesian_tree.h`, `triple.h:292`, `array.h:80-83`, `segment_tree.h`, `bplus_tree.h`, `aa_tree.h`, `rb_tree.h`, `fq_packed_vector_v2.h`.
- **`crypto/`:**
  - `sha1.h:185-188` calls `preprocess_message` and `big_endian_to_host`, which were never ported. The round function itself is correct.
  - `sha3/common.h:55` uses `static_assert` on a runtime span size. The `cthash` helpers it needs are missing, and `sha3.h` is empty. The Keccak core is correct.
  - `compile_time_sha*.hpp`, `md5.h` and `padding.h` depend on types that are not in the repo.
- **`loop_fusion/runtime/looper.hpp:20,24`:** initialises `rng`, but the member is named `rang`.
- **`reflection/reflection.h:310-311`:** `enum_cases` casts `[-1,1024)` into unscoped enums. This is UB and not a constant expression under clang 21.
- **`graph/digraph.h:421`:** uses `digraph_paths` before declaring it, and does not include `<iomanip>`.
- **`thread/work_contract.h:3-9`:** missing standard includes.
- **`thread/annotated_mutex.h:66,79`:** has a wrong friend name and a wrong `unique_lock` type.
- **`compression/leb128.h:65,104`:** has type errors.
- **`math/`:**
  - `miller_rabin.h:17` and `tonelli_shanks.h:21` use an undeclared `mod_pow`.
  - `lucas.h:11` and `crt.h:34` use an undeclared `mod`.
  - `primitive_root.h:15` uses `.pb`.
  - `eea.h` `#error`s when included directly by its own test.
- **`matrix/matrix.h:712`:** Fq m4ri with `packed=false` calls a `clear()` that does not exist.
- **`simd/swar.h:13,45`:** cannot compile. `tests/simd/swar.cpp` uses `swar::` although the include is commented out.
- **`simd/neon.h`:** does not compile under `USE_ARM`, so the whole ARM path is dead. See section 3.1.
- **`list/enumeration/ternary.h`:** dead code that uses `#import` and undefined names.
- **Test files out of sync with the API:**
  - `tests/container/heap.cpp` and `ringbuffer.cpp` call methods that do not exist.
  - `tests/algorithm/{min,max,argmin,argmax,search}.cpp` use removed names.
  - `tests/math/avg.cpp` is out of sync as well.
  - `tests/crypto/sha1.cpp` has a non-constant `static_assert`.

---

## 2. Cross-cutting runtime bugs

Many other modules depend on these, so they cause the widest damage.

1. **`cryptanalysislib::memcpy` loses tail bytes**, `memory/memcpy.h:93` [C].
   - When the destination is misaligned, the tail length uses `t` after the alignment code has already changed it.
   - Example: with `dst+1` and n=80, 17 bytes are not copied.
   - An exhaustive sweep found 259,776 failing `(src offset, dst offset, n)` triples.
   - `ips4o::sort` uses this path, so it loses elements for every n > 2048. Its output is not a permutation of the input.
2. **`cryptanalysislib::memcpy<T>` counts elements, not bytes.** Several call sites pass byte counts, which over-copies by a factor of `sizeof(T)` (blocker 4).
3. **`cryptanalysislib::memcmp` returns inverted results**, `memory/memcmp.h:40-62` [C].
   - The 8-, 4-, 2- and 1-byte tail branches return 1, meaning "different", when the bytes are **equal**.
   - Verified: two identical 8-byte buffers give 1, and two different ones give 0.
   - `algorithm/equal.h:38` returns this value directly, so `equal()` gives the opposite answer from `std::equal`.
4. **`memset<uint16_t>` misses elements**, `memory/memset.h:148-149` [C]. For n ≤ 32 bytes, elements 8 to 15 are never set. Separately, the gcc-only `memset<uint8_t>` M02 case at `:73` zeroes the second byte.
5. **`popcount<T>` sign-extends**, `algorithm/bits/popcount.h:25` [C]. For example, `popcount<int8_t>(-1)` returns 64.
6. **`clz<u32/u16/u8>(1)` returns 63**, `algorithm/bits/clz.h:20` [C].
   - It uses `clzl` regardless of the type width.
   - As a result, `bsr<u32>(1)` returns 4294967264.
   - The 128-bit branch is inverted at lines 24-27.
   - `ffs<u128>` is missing `+64` at `ffs.h:24`.
7. **`math::log` / `log2` / `exp` never terminate**, `math/log.h:28-29`, `math/exp.h:376` [C].
   - The loops stop only when the change falls below `DBL_EPSILON` in absolute terms, which can never happen for many inputs.
   - Examples: `log(50)`, `log(200)` and `log(999)` hang, `exp(x)` hangs for x ≥ ~100, and `entropy.h` `HH(0.8)` hangs.
   - `log(0)` recurses until the stack overflows.
   - The integral `log2` truncates: `log2<int>(8)` returns 2.
8. **The binomial coefficient `bc(n,k)` overflows**, `math/bc.h:30,32` [C]. Results are wrong for n=63 and n=64, for example `bc(63,29)` and `bc(64,32)`.
9. **`gcd()` is wrong for many inputs**, `math/gcd.h:105-113` [C].
   - It uses a 32-bit `ctz`, an `int` difference, and calls `ctz(0)`.
   - 28.9% of random u64 pairs give the wrong result.
10. **Random number helpers**, `random.h` [C].
    - Signed `rng<T>(limit)` and `rng<T>(l,h)` return out-of-range values in about 45% of draws (`:355-368`).
    - `random_device(seed)` ignores the seed, and seed 0 crashes with SIGFPE (`:533-549`).
    - `xorshf96_seed()` never seeds, because its `fread` check is always true (`:70`).
    - `rng_weighted<u64>` uses `1u << pos` for positions up to 63, which is undefined (`:376-392`).
    - The RNG state is `static` per translation unit, so seeding in one TU does not affect another.
    - pcg64 returns 0 until it is seeded.

---

## 3. Findings by area

### 3.1 SIMD (`src/simd`)

**Failing tests**

1. [C] **`avx2.h:509, 522`:** `_uint8x16_t::ror`/`rol` use 16-bit shifts with the mask `(1<<(8-n))-1`, and `ror` is a copy of `rol`. `rol(0x81,1)` returns 1 and `ror(1,1)` returns 0. This causes `test_simd.h:190`.
2. [C] **`avx2.h:1247`:** the `_Xint32x4_t` union declares `T16 d[32]` where `T32 d[4]` is intended.
   - This makes `sizeof(_uint32x4_t)` 64 and `sizeof(uint32x8_t)` 128 (`avx2.h:3404`).
   - `TxN_t<uint32_t,N>` (`generic.h:144`) therefore never writes `d[8..15]`. This causes `generic.cpp:85` and the `TxN_tuint32_*` failures.

**`avx2.h`, 128-bit types**

3. [C] **`:476` `u8x16::slli`:** the mask is wrong. `slli(2,1)` returns 0.
4. [C] **`:494-495, 1025, 1487, 1936` `srli`:** a `constexpr` local depends on a runtime parameter, so any call fails to compile.
5. [C] **`:185`:** `int8x16`'s `V` is `__v16hi`, so every arithmetic operation fails to compile.
6. [C] **`:725` `_Xint16x8_t::V = __v16qu`:** all arithmetic runs per byte. `add(255,1)` returns 0, and `gt`, `popcnt` and `gather` are also wrong.
7. [C] **`:1124` `all_equal`:** returns a vector from a `bool` function.
8. [C] **`:1230, 1691`:** `_Xint32x4_t` and `_Xint64x2_t` use the 256-bit `V` on `__m128i`, so they do not compile.
9. [C] **Scatter loops hard-coded to 8 iterations**, at `:658, 1164, 1624, 2073, 2729, 3344`. Some lanes are never written, and others are read out of bounds.

**`avx2.h`, 256-bit types**

10. [C] **`:2933-2941` `uint16x16_t::store<false>`:** performs an aligned store and segfaults on misaligned pointers.
11. [C] **`:2992` `andnot`:** computes `~(a&b)`.
12. [C] **`:3051` `u16 div` and `:4313` `u64 div`:** use `mulhrs` and are wrong. `div(7,2)` returns 4, and `div(100000,10)` returns 62430.
13. [C] **`:3062` `u16 slli(x,0)`:** returns 0.
14. [C] **`:3213` `u16 eq`:** uses `_pdep_u32` where `_pext_u32` is needed. `eq(1,1)` returns 0x5555.
15. [C] **`:3238, 3873, 4484` `cmp`:** returns a byte-granular mask, while `gt`, `lt` and `move` return per-lane masks.
16. [C] **`:3663` `u32 mulhi`:** is identical to `mullo`.
17. [C] **`:3985` `u32 pack` and `:4006` `cvtepu8`:** go through `__v8hi` and produce wrong results.
18. [C] **`:4019/4029` `u32 min/max`:** use signed comparisons.
19. [C] **`:4647` `u64 max`:** compares the lanes as doubles, and the gcc branch computes `min` instead. `:4621` `min` has the same cast.
20. [C] **`:4591` `u64 gather`:** ignores `scale`.
21. [C] **`set_bit` uses `1u << pos`**, which is UB for pos ≥ 32: `avx2.h:1740, 4137` and `avx512.h:2449`.
22. [C] **`:4689` `avx2_load_f32x8`:** converts the integer `0x7F800000` to a float value instead of reinterpreting its bits as +inf.
23. [P] **`:3889, 4496`:** the popcount fast path is guarded by `USE_AVX512`, which is never defined, so it is dead code.

**`avx512.h`** (tested with a shim that works around blocker 1)

24. [C] **`:907, 2224` `permute`:** the arguments to `_mm512_permutexvar_*` are swapped.
25. [C] **`:1023/1041` `u8x64 min/max`:** use 64-bit lanes.
26. [C] **min/max width or signedness errors:** `:1674` (u16 min uses `epi32`), `:1684`, `:2344/2354` and `:2986/2996` use signed comparisons on unsigned types.
27. [C] **`:957, 1648, 2933` `reduce_min/max`:** use `epu32` for 8-, 16- and 64-bit lanes.
28. [C] **`:708, 1556, 2839` `lzcnt`:** computes the trailing-zero count instead.
29. [C] **`:558, 569` 8-bit `ror`/`rol`:** use 16-bit shifts.
30. [C] **`:1400, 2683` `srli`:** shift arithmetically. **`:2370/2671`:** `u64 V = __v64qu`, so `slli` shifts per byte.
31. [C] **`cmp` uses the wrong `V` lane width:** `:1065, 1700, 2370`.
32. [C] **Broken `div`:** u32 and u64 `div` (~`:2004`, ~`:2656`) have empty bodies, and u16 `div` (~`:1386`) uses `mulhrs`.
33. [C] **`:950` and siblings `test()`:** pass the struct instead of `.v512`, so they do not compile.
34. [C] **`simd.h:121-198` `Mask<>`:** the operators either fail to compile (const methods assigning to members) or are no-ops (results discarded).
35. [C] **`avx512_intrinsics.h:79, 118`:** `(int)LONG_MIN` evaluates to 0, so the saturating 32-bit add/sub are wrong. **`:36`:** `setr_epi16` takes `char` parameters.

**Generic fallback** (`simd.h` without AVX2, and `generic.h`)

36. [C] **`slli`/`srli` bodies are swapped** in all four 128-bit types: `simd.h:567/580, 1134/1147, 1689/1702, 2225/2238`.
37. [C] **`move()` always returns 0:** `simd.h:792, 1360, 1890, 2450`.
38. [C] **`simd.h:3429` `u16x16 ror`:** is identical to `rol`.
39. [C] **`simd.h:2992` `eq_`:** returns 1 per lane instead of 0xFF.
40. [C] **`generic.h:404-440` `andnot_`:** computes NAND, and its SIMD path does not compile.
41. [C] **`generic.h:600` `cmp_`:** the scalar path means `==` but the SIMD path means `!=`.
42. [C] **`generic.h:751, 761` `move`:** shifts by ≥ 32, which is UB.
43. [C] **`generic.h:696, 713` `gather`/`scatter`:** scale the index twice.
44. [C] **`generic.h:726` `permute`:** uses scatter semantics with the arguments swapped, which causes a stack overflow.
45. [R] **`generic.h:446, 452, 583, 589`:** `mul`, `mulhi`, `ror` and `rol` return an uninitialised value. `:803/807` `operator*` does not compile.
46. [C] **`bits/bits.h:20-32`:** the `pdep` fallback is wrong. `pdep(0b11,0b1010)` returns 0.
47. [C] **`float/avx2.h:28, 36`:** uint32 is converted as signed and rounded to nearest, while the generic path floors. **`float/simd.h:67`:** `u64→f64` goes through `float`.

**NEON** (`neon.h`, checked by cross-compiling and with an emulation shim)

48. [C] **The file does not compile:**
    - `lt`/`lt_` are declared twice at `:3278, 3296`.
    - `rng()` is unqualified at `:1369, 2113, 2819, 3553`.
    - `vmulq_n_u8` does not exist (`:469-471`).
    - The converting constructors are undefined, which gives a link error (`:1072, 1087, 1214, 1229`).
    - The signed types cannot be used (`:186/191`, `:813/818`, and the `set`/`setr`/`load` overloads).
49. [C] **Wrong results** (checked through the shim):
    - `u8 srli` shifts left (`:504-506`), and `64x4 srli` has the same problem (`:3862`).
    - The constexpr `srli` masks are wrong (`:1748, 2444, 3132, 3852`).
    - `andnot` computes NAND (`:1643, 2344, 3031, 3749`).
    - `cmp_` is always 0 (`:671`).
    - `reverse` reverses bits instead of bytes (`:712`).
    - `scatter` writes only 8 of 16 lanes (`:752`).
    - `le_` uses `lt` (`:2585, 2619`).
    - `cmp` uses a half-shift of 16 for every width (`:2688, 3406, 4136`).
    - The `move`/`eq` shifts are wrong (`:3412, 4083, 4086, 4142`).
    - `set`/`setr` lane order is reversed compared with AVX2 (`:2118`–`:3581`).
    - `popcnt` does not mask the high half (`:3457, 4209`).
    - `mullo(S, uint8_t)` truncates its scalar (`:2401, 3089`).
50. **Note:** the `*_ternary.h` files are vpternlog truth tables. All 256 functions are correct in every backend. The only problem with them is the ODR issue in blocker 5.

### 3.2 Containers (`src/container`)

**Binary packed vector** (`binary_packed_vector.h`)

1. [C] **`:1194-1213` `scalar()`:** the `else` branch has no `return`. At -O0 it traps.
2. [C] **`:1149-1190` `mul()`:** does not compile. It is also a copy-paste of `add` that wipes its own result.
3. [C] **`:1009` `add_weight(FqPackedVector v3, …)`:** takes `v3` by value, so the caller's vector never receives the result.
4. [C] **`:384-401` `random(l,u)`:**
   - Within a single limb it falls through.
   - `upper_limb-1` underflows, so the loop runs about 4.29e9 times.
   - It writes to the wrong limb.
   - The middle limbs are never randomised.
5. [C] **`:452` `random_with_weight(w,m,offset)`:** ignores `offset`.
6. [C] **`:498, 523` `is_zero(l,u)`:** skips limb `upper_limb-1`.
7. [C] **`:291, 324, 496, 521`:** `zero`, `one` and `is_zero` access `__data[limbs()]` when `k_upper==n` and `n%64==0`.
8. [C] **`:253-263` `set_bit`/`flip_bit`/`clear_bit`:** shift without `% RADIX`, which is UB for positions ≥ W and wrong for T narrower than 64 bits.
9. [C] **`:1260-1270` `slr()`:** zeroes the wrong range.
10. [C] **`:1686` `hash(l,h)` for T ≠ u64:** the mask is computed in 64 bits and then truncated, so nothing is masked.
11. [C] **`:368` `random()`:** hard-codes `< 64`, so for T=u32 limb 1 is never randomised.
12. [C] **`:1553` `reference::data()`:** returns `bool()`.

**ctrie** (`ctrie.h`)

13. [C] **`:1880` `insert(key,value)`:** hashes `value`, while lookup hashes the key. After `insert(1,2)`, `lookup(1)` returns 0.
14. [C] **`:1826, 1833`:** a failed CAS overwrites `cur_` with another thread's node, and the retry then treats it as an array. This is the cause of the `container_ctrie` segfault.
15. [C] **`:1317-1324`:** the equal-hash branch calls `createLNode(sn1_)` twice, calls `delete` on memory that came from `malloc`, and then uses the freed node.
16. [C] **`remove` (`:1912, 1930, 1937`):** the CAS always fails, but `remove` still returns non-null and decrements the count. **`:627/634` `incrementCount`:** decrements on failure. **`:655`:** no destructor.

**Stack, queues, lists**

17. [C] **`stack.h:76`:** `push` returns capacity, and `capacity()` at `:58` returns the size.
    - `grow()` at `:124-129` does not reallocate, which causes a heap overflow at `:73`.
    - Separately, `tests/container/stack.cpp:26` expects the wrong value from `pop`.
18. [C] **`vector_queue.h:327-357` `push`:** reports success on a full queue and overwrites slot 0. **`:307` `back()`:** reads out of bounds.
19. [C] **`queue/spsc_fixed_queue.h:150-153`:** `begin`/`end` are unmasked, which causes a heap overflow after wrap-around. **`:262-263`:** the indices are `volatile` instead of atomic, and TSan reports a race.
20. [C] **`linkedlist/linkedlist.h`:**
    - `begin`/`end` are off by one (`:153`).
    - With signed T the sentinels compare in the wrong order, leading to a null dereference (`:161`, `:108-117`).
    - Nodes leak (`:181, 201, 211`).
    - The "lock-free" list races on a shared, non-atomic cursor (`:83, 104-142`; TSan).

**Hashmaps** (`hashmap/`)

21. [C] **`simple_compressed.h`:**
    - The capacity check can never fire, so inserts write past the bucket (`:94-116`).
    - `decompress` drops the last element (`:132`).
    - A descending insert wraps the delta, and the leb128 stream desyncs (`:106` together with `leb128.h:76`).
22. [C] **`simple2.h:196-239`:** re-hashes a value that is already a bucket index. **`:98-104`:** the FAA counter wraps.
23. [C] **`simple.h:286-290` `clear(tid)`:** never clears the tail buckets.
24. [C] **`hopscotch_hash.h:980-984` `at()`:** for a missing key it returns `*nullptr`.

**Fq packed vector** (`fq_packed_vector.h`)

25. [C] **`:507` `swap`:** passes the `set` arguments in the wrong order, which overflows the stack for q=255.
26. [C] **`:812, 835, 858` `add` and `:907, 940` `mul`/`scalar`:** overflow before the `% q` reduction for q ≥ 128.
27. [C] **`:520, 530` `neg`:** with a uint32 DataType, computes `(2^32-x) mod q`.
28. [C] **`:241` `hash`:** the 64-bit window shifts by 64, which is UB.
29. [C] **`:1121` `right_shift`:** the unsigned `j--` loop wraps.
30. [C] **`:739` `mod256_T`:** calls `neg_T`.
31. [C] **`:283` `accessMask`:** computes the wrong shift.
32. [C] **`:330, 1206`:** the signed variant promotes a negative value to u64 before `% q`.
33. [C] **The q=3 specialisation:**
    - The SIMD `add` loop skips limbs (`:1756`).
    - `neg(l,u)` is wrong (`:1471-1494`).
    - `neg<kl,ku>` reads out of bounds (`:1517`).
    - `add_only_weight_partly` is wrong (`:1784`).
    - `filter2count_T` masks are wrong (`:1862`).
    - `filter2count_range_T` uses the wrong mask (`:1846`).
    - `mod_T_withoutcorrection` uses `<<` where `|` is meant (`:1590`).
    - `filter2_mod3` calls an undeclared function (`:1913`).

**Fq vector** (`fq_vector.h`)

34. [C] **`:557` `rol`:** drops the wrapped part.
35. [C] **`:1327` q=4 `mul_T`:** all lanes collapse into lane 0.
36. [C] **`:72, 94` `hash`:** computes `1ull<<64`.
37. [C] **Overflow before `% q`:**
    - mul at `:394, 418, 473, 858`.
    - add at `:344, 437, 652–692`.
38. [C] **`:1802` q=5 `add<kl,ku,norm>`:** ignores its range.
39. [C] **`:185` `random_with_weight`:** never produces q-1.
40. [C] **Operations that do not compile for T ≠ uint8:** `:511, 569, 611, 709`.

**FqElement** (`kAry_type.h`)

41. [C] **`:457` `operator-=(T)`:** adds instead of subtracting.
42. [C] **`:210` `addmul`:** is missing the final `% q`.
43. [C] **`:527-537` `operator=`:** negative values are not normalised.
44. [C] **`:1050, 1077` 256-bit `sub`/`mul`:** wrap before the reduction.
45. [C] **`:917-929` `popcnt(l,u)`:** is a copy of `neg`.
46. [C] **`:956, 1011` rotations:** do not compile or have no effect.
47. [R] **`:182` `random(l,u)`:** uses `==` where an assignment is meant.

**Other containers**

48. [C] **`critbit.h:293` `remove`:** treats a prefix of a key as a match. `remove("a")` deletes `"aa"`.
49. [C] **`bk_tree.h:200-269`:** a fake zero root is returned from lookups, and inserting a real 0 is dropped.
50. [C] **`imap.h:440, 481-526`:** `assign()` never grows the storage, which causes a heap overflow.
51. [C] **`trie.h:619`:** `ch-'a'` is not range-checked. **`dancing_links.h`:** leaks memory.
52. [P] **`binary_indexed_tree.h:455`:** `update(0,v)` loops forever.
53. [P] **`sparse_table.h:525`:** calls `clz(0)`.

### 3.3 Matrix (`src/matrix`)

1. [C] **`binary_matrix.h:2378-2379` `_mzd_transpose_notsmall`:** uses the class's `nrows`/`ncols` instead of the recursion's parameters. Transposing a 1100×1300 matrix overflows the heap.
2. [C] **`binary_matrix.h:1182-1184` `matrix_echelonize_partial`:** discards the pivots it found when a block is short. `gaus()` and `m4ri()` then return far fewer pivots than exist, for example 3 instead of 4 and 60 instead of 63. `matrix.h:872` has the same problem for Fq m4ri.
3. [C] **`matrix.h:573-585` Fq `gaus`:** only accepts pivots equal to 1 or q-1, so it stops early for q ≥ 5. **`:567`:** `m = ncols-1` makes the 6×6 identity report rank 5.
4. [C] **Fq m4ri for q ≥ 5:** the output is not systematic in 26 of 30 runs. The suspected cause is in `sub_gaus` at `:826-847`, which scales by the wrong factor.
5. [C] **`binary_matrix.h:433, 447` `is_equal`:** is missing `+ j`.
6. [C] **`binary_matrix.h:953` and `matrix.h:1050` `swap`:** call `set(…, i1, i2)` where `(i1, j1)` is meant.
7. [C] **The binary `FqMatrix` implicit `operator=`:** copies raw owning pointers, which causes a use-after-free.
8. [C] **`binary_matrix.h:732-757` `sub_matrix`:** has UB shifts, uses the wrong row index, and produces wrong output.
9. [P] **`matrix.h:213` `copy_sub`:** passes the `set` arguments in the wrong order. **`binary_matrix.h:1216` `fix_gaus`:** can pivot on the syndrome column. **`:532-551` `row_xor`:** does not compile. **`matrix.h:1314`:** uses `&&` where `==` is meant.

### 3.4 Math and algorithms

1. [C] **`count` over u8 returns 196 instead of 100.** `simd.h:5102` `operator==` returns `int` -1, which then hits the popcount sign-extension bug in section 2, item 5.
2. [C] **`algorithm/zip.h:85-95` and `:166`:** the `__m256i*` overloads store with `*out = …`, which the compiler emits as `vmovdqa`. The wrapper passes `uint16_t*` pointers that are only 2-byte aligned. `test_algorithm_zip` segfaults in about 45% of runs depending on ASLR, and lldb places the fault at `zip.h:93`. `zip_u16/u32/u64` follow the same pattern.
3. [C] **`prefixsum.h:309-318`:** the output is shifted by one, so `{1,2,3,4,5}` becomes `3 6 10 15 ?`. `inclusive_scan.h:28` inherits this. The AVX2 variants read `v[-1]` for small n (`:127, 137, 165, 195, 231`).
4. [C] **`random_index.h:33-38`:** a missing `return` means `generate_random_indices(d,5,3)` hangs.
5. [C] **`mq/fes.h:318`:** a hard-coded `popcount > 10 → continue` drops every solution of weight above 10. [P] At `:648` a debug early-return limits `enum_16x16_w` to 2 steps.
6. [C] **`histogram.h:993-1003`:** `histogram<uint8_t>` overwrites counts where other types accumulate. [P] The parallel path uses 32 bins instead of 256 and drops the remainder.
7. [C] **`transpose.h:97-129` `transpose8`:** does not mask its outputs to 8 bits.
8. [C] **`bits/block.h:13`:** uses a 32-bit popcount on u64. **`bits/gather.h:16, 38`:** `pdep` and `pext` are swapped. **`bits/pdep.h:169`:** shifts by 32.
9. [C] **`math/abs.h:251` `abs_branchless<u32>(5,3)`:** returns 4294967294.
10. [C] **`math/round.h:313`:** `round(2.7)` returns 2.
11. [C] **`math/ceil.h:27`:** `cceil(3e9)` returns 0.
12. [C] **`math/mod.h:233` `fastmod<negative d>`:** is wrong in 85.8% of cases.
13. [C] **`math/bigint.h:257-262` `operator<`:** compares from the least-significant limb first. **`:242-250` `operator==`:** compares only `min(N,M)` limbs.
14. [C] **`math/bc.h:383` `reverse_biject<u32>`:** underflows the shift amount.
15. [C] **`math/log.h:101`:** `ceil_log2(2^64-1)` returns 1.
16. [C] **`math/gcd.h:261`:** `gcd_recursive_v0(12,0)` returns 0.
17. [C] **`math/prime.h:129-150`:** `next_prime(0)` returns 0 and `next_prime(1)` returns 1.
18. [P] **`math/crt.h:453`:** calls `EEA` with the argument roles swapped.

### 3.5 Sorting and search

1. [C] **`sorting_network/avx2.h:1913-2027` `sortingnetwork_small_{u32,i32}`:** produce garbage output.
   - The code assumes 16 lanes per vector, but `__m256i` holds 8.
   - The temporary buffer is uninitialised.
   - The sorted tail is never written back.
   - 1071 of 1100 test inputs fail. [R] `avx512.h:2401-2444` has the same tail bug.
2. [C] **`avx2.h:1882, 1900`:** `sort_i32x40` and `sort_i32x56` use the unsigned comparator.
3. [C] **`avx2.h:1547` `u8x224` and `:633` `u16x16_odd_even`:** never sort.
4. [C] **`sorting_network.h:9-19`:** the asm `int32_MINMAX` sorts in descending order.
5. [C] **`sorting_network.h:37-49`:** the scalar `MINMAX` destroys data, turning (5,3) into (3,3). Without `USE_AVX`, `djb int32_sort` uses it for n ≤ 8.
6. [C] **Other sorts:**
   - `robinhoodsort.h:36` copies `sizeof(T)²·n` bytes, and `:94/116` truncate values.
   - `float_radixsort.h:114-137` uses a stale `mOffset`, which causes a heap overflow, and `:285` frees `new[]` memory with plain `delete`.
   - `radixsort.h:37` allocates bytes where it needs `size_t`s.
   - `vv_radixsort.h:59-77` causes a use-after-free and double free on the second call.
   - `heap/is_heap.h:52` reads out of bounds.
   - `common.h:13-35` uses an `int` swap temporary that truncates values.
   - `sort.h:404-411` `find_next_empty_slot` overwrites live entries.
   - `sort/sort.h` gates the djb include on the wrong macro.
   - These sort headers do not compile: `selectionsort.h`, `quicksort.h`, `heapsort.h`, `bit_sort.h`, `divsufsort.h` (it includes itself), and `ska_sort_copy`.
7. [C] **`tests/sort/hunter.cpp:65, 125`:** `for(;;)` has no exit, which is why the test times out. Lines `:31, 35-37` also use `sizeof` where an element count is meant.
8. [C] **`tests/sort/sorting_algorithms.cpp:118`:** `HeapSort` is missing from the test registration.
9. [C] **`search/binary.h`, n==1 and n==0 cases** (`:38, 65, 92, 121, 150, 179, 212, 266, 296, 327, 359, 386, 453, 505, 535, 573, 605`): for n==1 these return 0 without comparing. The `standard`, `boundless` and `doubletapped` variants also return 0 instead of −1 for n==0.
10. [C] **`binary.h` wrong results:**
    - `bsearch_leq` uses an inverted predicate (`:89-107, 175-193`).
    - `bsearch_approx` does not add the offset back (`:208-250`).
    - `Khuong_bin_search` reads `low[len_list]`, which overflows the heap (`:397`).
    - `upper_bound_monobound` is wrong for 9954 queries (`:643-671`).
    - `lower_bound_monobound` misses the first occurrence (`:692-728`).
    - `tripletapped` is wrong (`:781-811`).
    - `search::binary_search` does not check equality, so it reports false matches (`:1093-1141`).
11. [C] **`search/linear.h:34-56` `upper_bound_linear_search`:** is off by one.
12. [C] **`search/interpolation.h`:**
    - `:33-91` reads out of bounds for absent keys and converts NaN to an integer.
    - `:117` divides by zero, causing SIGFPE.
    - `:122/138` return `first-1`.
    - `:248/311` convert NaN to an integer.
    - `:438` dispatches to binary search instead of interpolation search.

### 3.6 Lists, enumeration, tree, NN

1. [C] **`list/list.h:692-709` `binary_search`:** returns the `lower_bound` position without an equality check.
   - `search_level` therefore reports false matches.
   - `search_boundaries` returns `(210,211)` instead of `(load,load)`.
   - A tree join produced 594 results where a naive join found 7.
2. [C] **`list.h:455-469` `find()`:** the loop bound uses the out-parameter, so it always counts 1.
3. [C] **`list/common.h:378-389` `is_correct()`:** returns true as soon as any single element is correct.
4. [C] **`common.h:360-371` `copy()`:** uses the wrong offset, has its two branches swapped, and passes a byte count to an element-count copy, which overflows the heap.
5. [C] **`element.h:252-260, 304-312` `sub`:** subtracts the labels but adds the values, so `label ≠ H·value` for q > 2.
6. [C] **`list/parallel_full.h:141-193` `sort_level`:** for cross-limb ranges the upper limb is shifted the wrong direction, and a shift by 64 occurs.
7. [C] **`parallel.h:148-158` `random()`:** segfaults when `no_values=true`. **`:184-203` `sort()`:** permutes the labels but not the values.
8. [C] **More list bugs:**
   - `parallel_index.h:115` `zero()` uses element counts as byte offsets.
   - `list.h:833-870` `add_and_append(tid)` ignores `start_pos`.
   - `common.h:768` `random(m,tid)` never runs its loop.
   - `common.h:475` `size(tid)` is wrong for the last thread.
   - `common.h:400, 437` `is_sorted` defaults to a bit count given in bytes, and `:434-470` skips the last pair.
   - `list.h:141` `end()` reads out of bounds.
   - `list.h:213` `sort()` moves the zero padding to the front.
   - `list.h:805` `append` and `common.h:734` `erase` do not update `__size`.
   - [R] `list.h:23-25` ignores the `ListConfig` template parameter.
9. [C] **`combination/chase.h:611`:** `memcpy` resolves to the element-count overload (blocker 4), which overflows the stack.
10. [C] **`chase.h:201`:** calls `resize(listsize)` where `resize(size)` is meant, so the change list is empty and indexed out of bounds.
11. [C] **`chase.h:404-439` `enumerate3`:** produces wrong positions and wrong counts for p=3.
12. [C] **`list/enumeration/fq.h:402, 424`:** `recalculate_label` drops the syndrome, so 109 of 112 labels are wrong.
13. [C] **SinglePartialSingle enumerators** (`fq.h:581`, `binary.h:676`):
    - `LIST_SIZE` uses `bc-1`, so the last block is skipped.
    - For `noreps_w ≥ 3` the reset is wrong.
14. [C] **`nn/nn.h` tail handling:**
    - The tails at `:809, 895, 980, 1029, 1140, 1255, 2077` are skipped or read out of bounds.
    - `bruteforce_96` (`:624, 708-730`) is wrong.
    - `simd_64_uxv` and `256_64_4x4` (`:1795, 2686`) read out of bounds.
    - `generate_special_instance` (`:280-293`) allocates twice.
    - `:597-602` uses a debug-only variable inside an `assert`.
15. [C] **`tree/d1.h:23, 109`:** a zero target skips the sort. **`d2_stream.h:108`:** the weight filter is inverted. [R] **`:357`:** uses the wrong index.

### 3.7 Compression, hashing, crypto

1. [C] **`compression/bwt.h:87`:** `bwt_inplace` returns 0 instead of n, so `bwt_reverse` writes `rev[-1]`.
   - `:16` uses `strlen` as the length.
   - `:12` compares signed with unsigned bytes.
   - `:45-46` read out of bounds.
2. [C] **`compression/lzss.h:148/160`:**
   - The encoder stops consuming input when the output buffer reaches `src_length`.
   - It can then write up to 4 bytes past the buffer at `:201`.
   - `DataCompare` (`:107-110`) returns one more than the matched length.
   - 9- and 16-byte inputs make it hang (`:212-222`).
   - Inputs of n ≤ 8 never roundtrip.
3. [C] **`leb128.h`:**
   - Decoding is wrong for most values (`:77-83`).
   - Negative values encode wrongly (`:28`).
   - `skip` has no effect (`:110`).
4. [C] **Other compressors:**
   - `lz77.h:44` has an unbounded match, which reads and writes out of bounds.
   - `lzmat.h` over-reads by 3 bytes (`:247–471`) and fails spuriously at exact capacity (`:52/237`).
   - `lzf.h:185` over-reads by 1 byte.
   - `smaz2.h` hangs or overflows the heap (`:208-252`).
   - `constexpr_huffman.h:89` uses a negative index for non-ASCII input.
   - `deflate.h:192/448` calls `clz(0)`.
5. [C] **`crypto/aes.h:93–185`:** every round function takes the state by value. `aes_encrypt` therefore returns the plaintext unchanged on the NIST SP 800-38A F.1.1 test vector.
   - `:296` makes `aes_encrypt<256>` impossible to instantiate.
   - `:305` copies only 16 key bytes for AES-192.
6. [C] **`hash/adler32.h:61-62`:** the accumulators are 16-bit and wrap before the modulo, so the result is wrong from about 23 bytes on. This propagates to `deflate.h:307` and `inflate.h:572`, which cannot interoperate with zlib. **`:44, 211`:** the AVX-512 variant does a misaligned 64-byte store and uses the wrong stride.
7. [C] **`hash/crc.h:156-197` `sse42_crc32`:** ignores the trailing `len%16` bytes. **`:89-92`:** reads out of bounds when len < 64.
8. [C] **`hash/komihash.h:77-78`:** byte-swaps unconditionally, so 297 of 301 lengths differ from upstream on little-endian.
9. [C] **`hash/simple.h:465-478` `extract<>`:** does not mask the high limb. **`:80-94` `Hash<…,q>` for q not a power of two:** ignores `lprime` and reads only half of the digits.
10. **Checked and correct:** xxh3, cityhash, fnv1/fnv1a, scalar crc32, crc32c, the SHA-1 round function, and Keccak.

### 3.8 Allocators, threads, atomics, misc

1. [C] **`alloc/alloc.h:240` `StackAllocator::deallocate`:** zeroes the wrong region, which overflows the heap.
2. [C] **`alloc.h:612–660` `STDAllocatorWrapper`:** treats an element count as a byte count.
3. [C] **`alloc.h:705` `AlignmentMallocator`:** the same count-versus-bytes error. `histogram.h:1034` triggers it.
4. [C] **`alloc.h:305-314` `FreeListAllocator::deallocateAll`:** leaves a dangling root, which leads to a use-after-free.
5. [C] **`alloc.h:563` `FreeListPageMallocator`:** never reuses freed pages. **`:510` `owns()`:** returns true for any pointer at or above 4096.
6. [C] **`alloc/gc_simple.h:133`:** frees with `free()` memory that came from `new`.
7. [C] **`atomic/futex.h:45, 59`:** the CAS arguments are swapped.
   - The mutex is broken: one test counted 799286 of 800000 increments.
   - This breaks `mythread` and the `DEBUG_PRINTF` locking.
   - `:91-93` treats `EWOULDBLOCK` as "acquired".
8. [C] **`atomic/pipe.h:83, 129`:** the CAS arguments are reversed, so readers spin forever. **`:46-54`:** members are uninitialised.
9. [C] **`thread/steal.h`:**
   - `wait_for_tasks()` returns while a task is still running (`:117-122, 331-337`).
   - It moves from lvalue arguments (`:166, 193`).
   - `pause()` does nothing.
   - With 0 threads, submitted tasks are dropped.
10. [C] **`thread/work_contract.h:397-405`:** `stop()` never calls `notify_all`. This deadlocks `join` and is why `thread_work_contract` times out.
11. [C] **`thread/simple.h:98-108`:** `wait_for_tasks` can hang. **`:135`:** `clear_tasks` does not notify. **`thread/execution.h:259`:** a `noexcept` function rethrows, which calls `std::terminate`.
12. [C] **`bench/thread/join.cpp:297`:** `main` returns the global counter, so the exit code is non-zero.
13. [C] **`loop_fusion`:**
    - `compiletime/basic_looper_merge.hpp:124` tests `is_last` where `is_first` is meant, so a loop body runs twice.
    - `runtime/looper_union.hpp:37-41` assumes the loopers are sorted, which also runs a body twice.
14. [C] **`combination/` sequence generators:**
    - `shifts.h:72` skips combinations: it yields 10 of the 20 for 6C3.
    - `colex.h:42` and `algorithm/bits/periodic.h:11` use masks that are wrong for narrow T.
    - `necklace.h:51` reads out of bounds, and `:106` hangs.
    - `fibonacci_gray.h:66` shifts by 32.
    - `fibrep_subset_lexrev.h:34` gets stuck at 0.
    - `reverse_gray_code.h:41` is not a correct inverse.
    - `lexicographic.h:90, 167, 319` has no p==4 case and hangs in two places.
    - `subset_gray.h:19` computes `highest_one` incorrectly.
15. [C] **Misc:**
    - `helper.h:155` `translate_level` reads out of bounds.
    - `permutation/permutation.h:17` is copyable while owning a raw pointer, which causes a double free.
    - `print/print.h:50-60` `print_binary` has four separate bugs.
    - `traits.h:106` keeps the first closure's captures in a static, so later captures are ignored.
    - `access_profiler.h:215, 305` wraps the type index at 256 and hangs on real segfaults.
    - [R] `graph/digraph.h:40, 52, 172, 237` writes through null pointers, frees the wrong count, and wraps an index.

---

## 4. Suggested order of work

1. Fix blockers 1 and 2. They hide most of the test suite.
2. Fix `memcmp`, `memcpy`, `popcount` and `clz` from section 2, and remove the global `using namespace` directives. Many modules depend on these primitives.
3. Re-run the full test suite in both configurations, then work through the list, search, matrix and enumeration bugs. They directly affect the correctness of the ISD and subset-sum algorithms.
4. Add regression tests along the lines of the reproducers: differential tests against the standard library, zlib and naive references, using odd sizes such as 1, 63, 65, 127 and 129, and run them under the sanitizers.
