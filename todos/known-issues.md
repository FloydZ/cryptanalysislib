# Known issues (hand-maintained, not generated)

Open items found during the 2026-10 CI repair that are not written as TODO
comments in the source. `gen_todos.py` does not touch this file.

## Flaky / seed-dependent tests

Most tests call `rng_seed(time(NULL))`, so CI draws a new instance every run
and back-to-back local reruns often share a seed. To reproduce, copy the
test and replace the seed with
`rng_seed(getenv("SEED") ? strtoull(getenv("SEED"), nullptr, 10) : time(NULL));`,
then scan seeds. The full binary and `--gtest_filter` runs use different RNG
streams.

- [ ] `tests/nn/simd_bruteforce.cpp:172`, `bruteforce_simd_256`: misses the
      planted solution for some seeds (~4 in 1500 full-binary seeds, together
      with the next two items). Compiler-independent.
- [ ] `tests/nn/simd_bruteforce.cpp:247`, `bruteforce_simd_256_64_4x4`: same.
- [ ] `tests/nn/simd_bruteforce.cpp:265`, `bruteforce_simd_256_64_4x4_rearrange`:
      same.
      (The ux4/ux8 kernels had the same false-negative pattern: only the
      lowest candidate lane was checked. Fixed in `5.`; check these kernels
      for the same pattern.)
- [ ] `tests/algorithm/subsetsum/*`: 16 TESTs are commented out with
      `// not correct right now` (generic 1, join2lists 5, join4lists 6,
      join_stream 1, parameters 3). They fail always or flakily.

## Disabled, unfinished ARM/NEON code

These only compile when `USE_NEON` is set, which happens only on macOS arm64.

- [ ] `src/sort/sorting_network/neon.h`: unfinished port (`layers` and
      `simd_mergesorted_2` are undefined, functions are non-inline in a
      header); its include is commented out in `sorting_network/common.h`.
- [ ] `src/algorithm/transpose.h`: NEON `matrix_transpose_u8_32x32` is
      `#if 0` (`vec256_t` is undefined, no callers).
- [ ] `src/simd/neon.h`: the `_Xint32x4_t`/`_Xint64x2_t` `aligned_load` and
      `unaligned_load` declare `_Xint8x16_t out;` and only compile through
      `-flax-vector-conversions`.
- New NEON code: load/store with `vld1q_u8`/`vst1q_u8` + `vreinterpretq_*`,
  never through `poly128_t *` or other non-char pointers (strict aliasing;
  this broke the gcc-14 ARM nn tests).

## CI / build

- [ ] `matrix_fq` takes ~259 s on the macOS Intel runner (600 s ctest
      timeout). `matrix_simple` was cut down in Debug (`size_step = 9`)
      for the same reason.
- [ ] Only the clang **Release** build is warning-free. The Debug build
      (`-Wall -Wextra`) and gcc warnings have not been cleaned up.
- [ ] macOS (Intel and ARM) and ARM Linux fixes were never built locally;
      they are verified only by CI.
- [x] The earlier gcc-only `test_dispatch` / `algorithm_prefixsum` CI
      failures both came from the gcc job building without `USE_AVX2` (see
      below). `test_dispatch`: `cpucycles()` returned -1 without AVX2, fixed
      in round 1 (`rdtsc` on any x86_64); it passes on the generic backend at
      509f406. `algorithm_prefixsum`: generic `aligned_store` aliasing,
      fixed by patch 12.
- [ ] x86 gcc-13 Release CI (509f406) built **without `USE_AVX2`**
      (prefixsum ran only 3 tests), so it used the generic `simd.h` backend.
      gcc's local configure detects AVX2/AVX-512 fine. Suspect: the submodule's
      `host_optimizations.cmake` maps AMD family 25 only for models
      0/1/8/16/33/64/65/80/97/112; anything else (e.g. EPYC Genoa, model 17)
      stays `generic` = SSE2. Needs the CI configure lines (`detected cpu
      family`/model, "No idea what AMD model") to confirm.
- [ ] The generic `simd.h` backend is used by any x86 build without AVX2,
      but only the clang noavx job tests it; gcc has no generic-backend job.

- [ ] `should_par` (`src/algorithm/algorithm.h`) returns
      `ceil(size / min_size_per_thread)`, i.e. **1** for sizes in
      `[min, 2*min)`: every parallel algorithm then pays the thread pool
      overhead with a single worker (measured: histogram 3x slower than
      sequential at size == min). Returning 0 for `nt < 2` would fix all
      algorithms at once (found in TODO patch 042).

- [ ] `src/algorithm/regex.h`: quantifiers after a group and branches inside
      a group are broken: `x(ab)*y` does not match `xy` and matches `xababy`
      with length 7; `(a|b)c` does not match `bc`. The matcher applies the
      quantifier to the `GROUPEND` token (`int i = ... ? group_start : 0` is
      dead code). Needs a matcher redesign (found in TODO patch 076).
- [ ] `src/algorithm/find.h`: the unsigned SIMD path of `find` uses
      `&(*first)` for any forward iterator (wrong for non-contiguous
      containers), and the `assume_sorted` branch discards the
      `binary_search` result.

- [ ] `src/algorithm/set.h`: the public `set_difference` has an empty
      `if constexpr (... contiguous unsigned ...) {}` "use SIMD version"
      branch, so `internal::set_difference_uXX_simd` is never used
      (found in TODO patch 097).
- [ ] `src/algorithm/sat/cdcl.h`: `parse` does not check `fopen` and calls
      `exit(0)` on a parse error; the solver never frees its memory.

## Smaller leftovers

- [ ] `src/thread/performance.h`: fixed socket path
      `/tmp/cryptanalysislib_scheduler.socket`, so parallel test runs on one
      machine collide; the stale socket file is never removed after `delete`.
- [ ] `src/thread/mythread.h`: threads created with `clone()` share TLS
      (no `CLONE_SETTLS`).
- [ ] `src/crypto/sha1.h`: `_hex_bytes` still `throw`s on a non-hex digit
      (only reachable at compile time; the library is otherwise noexcept).
