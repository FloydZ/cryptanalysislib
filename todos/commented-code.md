# Commented-out code paths

Heuristic scan (expect a few prose-comment false positives): `//`/`/* */`/`#` comment runs that look like code (doc blocks `/** */`, `///` skipped), `#if 0`/`#if false` regions, the `#else` branch of `#if 1`, and `if (0)`/`if constexpr (false)`. Runs that are commented-out TESTs are in disabled-tests.md instead.

645 blocks (2493 lines) in 179 files.

| area | blocks | lines |
|---|---|---|
| (root) | 4 | 9 |
| bench | 1 | 1 |
| bench/algorithm | 2 | 2 |
| bench/enumeration | 2 | 2 |
| bench/list | 1 | 1 |
| bench/loop_fusion | 1 | 8 |
| bench/matrix | 2 | 2 |
| bench/mem | 1 | 20 |
| bench/subsetsum | 6 | 7 |
| bench/thread | 13 | 42 |
| bench/tree | 2 | 2 |
| cryptanalysislib | 43 | 59 |
| src (top level) | 27 | 107 |
| src/algorithm | 50 | 478 |
| src/alloc | 9 | 24 |
| src/atomic | 15 | 239 |
| src/combination | 13 | 59 |
| src/compression | 18 | 135 |
| src/container | 74 | 236 |
| src/crypto | 10 | 39 |
| src/graph | 2 | 2 |
| src/hash | 1 | 9 |
| src/list | 18 | 51 |
| src/math | 21 | 122 |
| src/matrix | 10 | 29 |
| src/memory | 2 | 2 |
| src/nn | 11 | 33 |
| src/simd | 12 | 47 |
| src/sort | 41 | 53 |
| src/thread | 14 | 31 |
| src/tree | 5 | 9 |
| tests | 8 | 13 |
| tests/algorithm | 65 | 213 |
| tests/alloc | 1 | 2 |
| tests/binary | 11 | 22 |
| tests/compression | 1 | 2 |
| tests/container | 47 | 214 |
| tests/crypto | 1 | 1 |
| tests/cuda | 16 | 38 |
| tests/hash | 1 | 1 |
| tests/list | 19 | 23 |
| tests/loop_fusion | 2 | 5 |
| tests/math | 2 | 4 |
| tests/matrix | 10 | 14 |
| tests/nn | 9 | 32 |
| tests/simd | 4 | 4 |
| tests/sort | 4 | 5 |
| tests/thread | 13 | 40 |

## (root) (4)

| location | lines | kind | first line |
|---|---|---|---|
| `CMakeLists.txt:38` | 1 | comment | `import needed` |
| `CMakeLists.txt:60` | 1 | comment | `check_cxx_compiler_flag(-latomic LINKER_SUPPORTS_ATOMIC)` |
| `CMakeLists.txt:205` | 5 | comment | `if (COMPILER_SUPPORTS_CONCEPTS_DIAGNOSTICS_DEPTH)` |
| `CMakeLists.txt:298` | 2 | comment | `set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}     -stdlib=libstdc++")` |

## bench (1)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/bench_config.h:26` | 1 | comment | `using kAryMatrix        = fplll::ZZ_mat<kAryType>;` |

## bench/algorithm (2)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/algorithm/argmax.cpp:25` | 1 | comment | `const size_t t = cryptanalysislib::argmax<T>(data.data(), state.range(0));` |
| `bench/algorithm/histogram.cpp:90` | 1 | comment | `histogram(data.begin(), data.end());` |

## bench/enumeration (2)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/enumeration/chase.cpp:23` | 1 | comment | `memcpy(w1, w2, element_limbs * 8);` |
| `bench/enumeration/chaseg.cpp:22` | 1 | comment | `memcpy(w1, w2, element_limbs * 8);` |

## bench/list (1)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/list/sort.cpp:15` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |

## bench/loop_fusion (1)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/loop_fusion/simple.cpp:14` | 8 | comment | `a[0] = e[1] + f[2];` |

## bench/matrix (2)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/matrix/fq_gaus.cpp:75` | 1 | comment | `B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:in` |
| `bench/matrix/fq_permute.cpp:47` | 1 | comment | `B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:in` |

## bench/mem (1)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/mem/malloc_free.cpp:12` | 20 | comment | `void parallelMalloc(int parallelism, int mallocCnt = 10000000) {` |

## bench/subsetsum (6)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/subsetsum/d1_tree_multithreading.cpp:50` | 1 | comment | `hm->info();` |
| `bench/subsetsum/d1_tree_multithreading.cpp:69` | 1 | comment | `std::cout << c << std::endl;` |
| `bench/subsetsum/d1_tree_multithreading.cpp:91` | 1 | comment | `hm->info();` |
| `bench/subsetsum/d1_tree_multithreading.cpp:111` | 1 | comment | `std::cout << c << std::endl;` |
| `bench/subsetsum/rho.cpp:17` | 2 | comment | `constexpr static SSS instance{.n=n, .q=q, .bp=PARAM_n1_3, .l1=PARAM_l1, .l2=PARAM_l2, .walk_len=128, .flavour_q=1021,};` |
| `bench/subsetsum/tree.cpp:45` | 1 | comment | `using Enumerator = BinaryRandomEnumerator<List, n/2, PARAM_n1_3>;` |

## bench/thread (13)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/thread/join.cpp:34` | 1 | comment | `mythread_yield();` |
| `bench/thread/join.cpp:56` | 1 | comment | `auto func = []{};` |
| `bench/thread/join.cpp:75` | 1 | comment | `auto func = []{};` |
| `bench/thread/join.cpp:77` | 4 | comment | `for ([[maybe_unused]] auto _ : state) {` |
| `bench/thread/join.cpp:94` | 1 | comment | `auto func = []{};` |
| `bench/thread/join.cpp:96` | 5 | comment | `for ([[maybe_unused]] auto _ : state) {` |
| `bench/thread/join.cpp:114` | 1 | comment | `auto func = []{};` |
| `bench/thread/join.cpp:116` | 4 | comment | `for ([[maybe_unused]] auto _ : state) {` |
| `bench/thread/join.cpp:135` | 1 | comment | `auto func = []{ return 1; };` |
| `bench/thread/join.cpp:137` | 4 | comment | `for ([[maybe_unused]] auto _ : state) {` |
| `bench/thread/sum_tree.cpp:9` | 1 | comment | `using Task = HeartbeatScheduler<>::Task;` |
| `bench/thread/sum_tree.cpp:54` | 9 | comment | `inline int64_t heartbeatsum(Task *t, Node *node) noexcept {` |
| `bench/thread/sum_tree.cpp:74` | 9 | comment | `struct SchedulerSum {` |

## bench/tree (2)

| location | lines | kind | first line |
|---|---|---|---|
| `bench/tree/join2lists.cpp:16` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |
| `bench/tree/join2lists2.cpp:18` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |

## cryptanalysislib (43)

| location | lines | kind | first line |
|---|---|---|---|
| `cryptanalysislib/analyser/json_analyzer.py:77` | 1 | comment | `j1 = json_Analyser([lines1])` |
| `cryptanalysislib/builder.py:54` | 1 | comment | `self.__create_build_env()` |
| `cryptanalysislib/builder.py:68` | 1 | comment | `logging.debug(cmd)` |
| `cryptanalysislib/builder.py:87` | 1 | comment | `logging.debug(cmd)` |
| `cryptanalysislib/builder.py:100` | 1 | comment | `logging.debug(self.__build_output)` |
| `cryptanalysislib/builder.py:130` | 1 | comment | `logging.debug(cmd)` |
| `cryptanalysislib/builder.py:143` | 1 | comment | `logging.debug(self.__run_output)` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:44` | 2 | comment | `print('vpcmpeqw %ymm0, %ymm15, %ymm15'.format())` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:49` | 1 | comment | `print('vpmovmskb %ymm15, %r11d')` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:53` | 1 | comment | `print('test %r11d, %r11d')` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:56` | 2 | comment | `print('jne ._report_solution_{0}'.format(i))` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:69` | 1 | comment | `xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq[b], dst=Fl[a])` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:72` | 1 | comment | `xor1 = "vpxor {offset}(%rdi), {dst}, {dst}".format(offset=32*b, dst=Fl[a])` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:75` | 1 | comment | `xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq_memref, dst=Fl[a])` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:77` | 1 | comment | `xor2 = "vpxor {src}, %ymm0, %ymm0".format(src=Fl[a])` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:83` | 1 | comment | `xor1a = "vmovdqa {offset}(%rsi), %ymm14".format(offset=32*a) # load Fl[a]` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:87` | 1 | comment | `xor1b = "vpxor {offset}(%rdi), %ymm14, %ymm14".format(offset=32*b)` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:90` | 1 | comment | `xor1b = "vpxor {src}, %ymm14, %ymm14".format(src=Fq_memref)` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:92` | 1 | comment | `xor1c = "vmovdqa %ymm14, {offset}(%rsi)".format(offset=32*a) # store Fl[a]` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:94` | 1 | comment | `xor2 = "vpxor %ymm14, %ymm0, %ymm0"` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:115` | 1 | comment | `print("vmovdqa {offset}(%rsi), {reg}   ## {reg} = Fl[{i}]".format(offset=i*32, reg=reg, i=i))` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:119` | 1 | comment | `print("vmovdqa {offset}(%rdi), {reg}   ## {reg} = Fq[{idx}]".format(offset=x*32, reg=reg, idx=x))` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:134` | 1 | comment | `Fq_memref = "{offset}(%rdi, %rdx)".format(offset=32*alpha)` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:157` | 1 | comment | `print("vmovdqa {reg}, {offset:2d}(%rsi)     #Fl[{i}] <-- {reg}".format(offset=i*32, reg=reg, i=i))` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:164` | 1 | comment | `print("vmovdqa (%rsi, %rcx), %ymm14")                  # load Fl[beta]` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:166` | 1 | comment | `print("vpxor (%rdi, %r8), %ymm14, %ymm14")             # xor Fq[gamma]` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:168` | 1 | comment | `print("vmovdqa %ymm14, (%rsi, %rcx)")     # store Fl[beta]` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:170` | 1 | comment | `print("vpxor %ymm14, %ymm0, %ymm0")` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:174` | 1 | comment | `print("vmovdqa %ymm0, (%rsi)     #Fl[0] <-- %ymm0")` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:177` | 1 | comment | `print('ret')` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:188` | 1 | comment | `print('._report_solution_{i}:          # GrayCode(i + {i}) is a solution'.format(i=i))` |
| `cryptanalysislib/generator/algorithm/mq/avx2_codegen.py:190` | 6 | comment | `print('vpxor %ymm15, %ymm15, %ymm15    # reset %ymm15 to zero')` |
| `cryptanalysislib/main.py:126` | 2 | comment | `pprint.pprint(param)` |
| `cryptanalysislib/main.py:134` | 1 | comment | `metaOptimizer = MetaOptimizer(problemOptimizer, args.algo_param)` |
| `cryptanalysislib/optimizers/helper.py:10` | 2 | comment | `if k/n not in ZZ:` |
| `cryptanalysislib/optimizers/nn.py:46` | 1 | comment | `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` |
| `cryptanalysislib/optimizers/nn.py:203` | 1 | comment | `print(lam_, gamma_, delta_star, gamma_star, lam)` |
| `cryptanalysislib/optimizers/nn.py:387` | 6 | comment | `print(calc_p(32, delta, True))` |
| `cryptanalysislib/optimizers/nn.py:397` | 3 | comment | `n = 86` |
| `cryptanalysislib/optimizers/subsetsum/rho.py:103` | 1 | comment | `multiH((1-g)*n/2,[(1-g)*n*w/2])` |
| `cryptanalysislib/optimizers/subsetsum/rho.py:108` | 1 | comment | `multiH((1-g)*n/4,[(1-g)*n*w/4])` |
| `cryptanalysislib/optimizers/subsetsum/rho.py:125` | 1 | comment | `multiH((1-g)*n/8,[(1-g)*n*w/8])` |
| `cryptanalysislib/parsers/parse_google_benchmark.py:14` | 1 | comment | `data = pd.read_csv(args.file, usecols=["name", args.metric])` |

## src (top level) (27)

| location | lines | kind | first line |
|---|---|---|---|
| `src/access_profiler.h:206` | 7 | comment | `FILE* out = fopen("_access_profile.out", "w+");` |
| `src/element.h:97` | 7 | comment | `requires requires(const typename Container::LimbType a,` |
| `src/element.h:112` | 1 | comment | `{ Container::length } -> std::convertible_to<uint32_t>;` |
| `src/random.h:37` | 4 | comment | `uint64_t* const Seed2 ) noexcept {` |
| `src/random.h:42` | 3 | comment | `kh_m128( s1, s2, &s1, &s2 );` |
| `src/random.h:46` | 2 | comment | `Seed2 = s2;` |
| `src/random.h:49` | 2 | comment | `return( s1 );` |
| `src/sort.h:303` | 2 | comment | `((b1 != b2) && (size_t<15)) ? true : false;` |
| `src/sort.h:536` | 1 | comment | `static_assert(b1 < b2);` |
| `src/sort.h:764` | 1 | comment | `__buckets[bucketOffset].third = et(L.data_label(i));` |
| `src/sort.h:1287` | 1 | comment | `std::fill(buckets_load.begin(), buckets_load.end(), 0);` |
| `src/sort.h:1291` | 1 | comment | `std::fill(__buckets.begin(), __buckets.end(), InternalPair(-1, -1));` |
| `src/sort.h:1416` | 1 | comment | `constexpr ArgumentLimbType mask = b0 == 0 ? rmask2 : rmask2&lmask1;` |
| `src/sort.h:1452` | 13 | comment | `if (nrt != 1) {` |
| `src/sort.h:1466` | 10 | comment | `for (uint64_t j = 0; j < nri; ++j) {` |
| `src/tree.h:2150` | 3 | comment | `std::cout << target << std::endl;` |
| `src/tree.h:2170` | 3 | comment | `std::cout << "\|L1\|: " << std::log2(L1.load() * 1.0) << ", " << L1.load() << std::endl;` |
| `src/tree.h:2215` | 2 | comment | `std::cout << L1 << std::endl << std::endl;` |
| `src/tree.h:2393` | 1 | comment | `using D = typename LabelType::DataType;` |
| `src/tree.h:2401` | 6 | comment | `using HML0 = SimpleHashMap<D, size_t,       simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;` |
| `src/tree.h:2411` | 2 | comment | `join2lists_on_iT_v2<k_lower1, k_upper1>(iL, L1, L2, iT, false);` |
| `src/tree.h:2414` | 2 | comment | `if (iL.load() == 0) { return; }` |
| `src/tree.h:2417` | 5 | comment | `iL.template sort_level<k_lower1, k_upper2>();` |
| `src/tree.h:2657` | 2 | comment | `std::cout << L1;` |
| `src/tree.h:3276` | 4 | comment | `ElementType::add(tmpe1, hmL1[k10], L1[k]);` |
| `src/tree.h:3340` | 20 | comment | `std::vector<Label> iTs(1u<<(depth-1u));` |
| `src/warp_scan.cuh:56` | 1 | #else of #if 1 | `return warp_scan_inc_up_add_shuffled(v,width) - v;` |

## src/algorithm (50)

| location | lines | kind | first line |
|---|---|---|---|
| `src/algorithm/argmin.h:174` | 16 | comment | `#ifdef USE_AVX2` |
| `src/algorithm/argmin.h:191` | 8 | comment | `size_t i = 0;` |
| `src/algorithm/argmin.h:200` | 8 | comment | `__m256i mask = _mm256_cmpgt_epi32(p, y);` |
| `src/algorithm/argmin.h:209` | 3 | comment | `p = _mm256_set1_epi32(min);` |
| `src/algorithm/argmin.h:213` | 6 | comment | `for (; i < n; i++) {` |
| `src/algorithm/argmin.h:220` | 2 | comment | `return idx;` |
| `src/algorithm/argmin.h:223` | 28 | comment | `template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>` |
| `src/algorithm/argmin.h:252` | 3 | comment | `p = _mm256_set1_epi32(min);` |
| `src/algorithm/argmin.h:256` | 6 | comment | `for (; i < n; i++) {` |
| `src/algorithm/argmin.h:263` | 2 | comment | `return idx;` |
| `src/algorithm/argmin.h:266` | 24 | comment | `template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>` |
| `src/algorithm/argmin.h:291` | 9 | comment | `y1 = _mm256_min_epi32(y1, y2);` |
| `src/algorithm/argmin.h:301` | 3 | comment | `p = _mm256_set1_epi32(min);` |
| `src/algorithm/argmin.h:305` | 6 | comment | `size_t idx2 = idx+31;` |
| `src/algorithm/argmin.h:312` | 5 | comment | `for (; i < n; i++) {` |
| `src/algorithm/argmin.h:318` | 2 | comment | `return idx2;` |
| `src/algorithm/argmin.h:330` | 4 | comment | `if (n < 128) {` |
| `src/algorithm/argmin.h:335` | 3 | comment | `return argmin_avx2_i32_bl32<config>(a, n);` |
| `src/algorithm/bits/zip.h:118` | 1 | comment | `x = (x \| (x<<1))  & 0x5555555555555555UL;` |
| `src/algorithm/bits/zip.h:128` | 1 | comment | `x = (x \| (x>>1))  & 0x3333333333333333UL;` |
| `src/algorithm/histogram.h:52` | 4 | comment | `const __m512i d1 = _mm512_i32gather_epi32(t2, cnt, 4);` |
| `src/algorithm/histogram.h:406` | 1 | comment | `__mmask64 bit7 = _mm512_movepi8_mask(data);` |
| `src/algorithm/histogram.h:914` | 2 | comment | `_mm256_i32scatter_epi32(C, newv, 4);` |
| `src/algorithm/int2weight.h:83` | 1 | comment | `weights ^= 1 << (wn - 1);` |
| `src/algorithm/int2weight.h:90` | 1 | comment | `weights ^= 1ull << (wn - 1);` |
| `src/algorithm/mq/fes.h:125` | 1 | comment | `extern struct solution_t * feslite_avx2_asm_enum(const void * Fq, void * Fl, uint64_t alpha, uint64_t beta, uint64_t gam` |
| `src/algorithm/mq/fes.h:313` | 5 | comment | `if (!valid) {` |
| `src/algorithm/mq/fes.h:497` | 10 | comment | `{` |
| `src/algorithm/mq/fes.h:563` | 1 | comment | `int npositive = 0;` |
| `src/algorithm/mq/fes.h:578` | 1 | comment | `if (n > 16) {` |
| `src/algorithm/mq/fes.h:582` | 1 | comment | `}` |
| `src/algorithm/mq/fes.h:584` | 1 | comment | `printf("FOUND %d positive for %ld iterations\n", context.size[0], iterations);` |
| `src/algorithm/prefixsum.h:34` | 1 | comment | `return x;` |
| `src/algorithm/regex.h:882` | 2 | comment | `if (group_end && i == group_end)` |
| `src/algorithm/rsa.h:19` | 1 | comment | `using L = TypeTemplate<config.N>;` |
| `src/algorithm/sat/cdcl.h:254` | 1 | comment | `printf("c restarting after %i conflicts (%i %i) %i\n", S->res, S->fast, S->slow, S->nLemmas > S->maxLemmas);` |
| `src/algorithm/shift.h:190` | 3 | comment | `dst[i] = src[i + n] for i in [b, e), front to back;` |
| `src/algorithm/shift.h:201` | 2 | comment | `dst[i + n] = src[i] for i in [b, e), back to front;` |
| `src/algorithm/subsetsum.h:258` | 1 | comment | `using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, instance.bp>;` |
| `src/algorithm/subsetsum.h:271` | 1 | comment | `constexpr static size_t factor = 2;` |
| `src/algorithm/subsetsum.h:362` | 2 | comment | `std::cout << tree_target << std::endl;` |
| `src/algorithm/subsetsum.h:380` | 6 | comment | `std::cout << "iters:" << iters << std::endl;` |
| `src/algorithm/subsetsum.h:408` | 3 | comment | `if ((iters % instance.print_iterations) == 0) {` |
| `src/algorithm/subsetsum.h:424` | 2 | comment | `std::cout << x2 << std::endl;` |
| `src/algorithm/subsetsum.h:431` | 5 | comment | `if (!x2.label.template is_equal<0, 26>(y2.label)) {` |
| `src/algorithm/subsetsum.h:442` | 11 | comment | `Label ss;` |
| `src/algorithm/subsetsum.h:474` | 9 | comment | `std::cout << x2 << ", x" << std::endl;` |
| `src/algorithm/subsetsum.h:665` | 1 | comment | `constexpr static size_t factor = 2;` |
| `src/algorithm/transpose.h:423` | 247 | #if 0 | `/// Transposes a 32x32 matrix of bytes using ARM NEON instructions` |
| `src/algorithm/zip.h:13` | 11 | comment | `#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)` |

## src/alloc (9)

| location | lines | kind | first line |
|---|---|---|---|
| `src/alloc/alloc.h:170` | 1 | comment | `constexpr static size_t alignment = 1;` |
| `src/alloc/alloc.h:512` | 1 | comment | `std::free(b.ptr);` |
| `src/alloc/alloc.h:684` | 1 | comment | `}` |
| `src/alloc/alloc.h:707` | 5 | comment | `template<class TT, class... Args>` |
| `src/alloc/alloc.h:714` | 5 | comment | `template<class TT>` |
| `src/alloc/alloc.h:722` | 4 | comment | `static constexpr inline size_type max_size(const Alloc &a) noexcept {` |
| `src/alloc/gc_simple.h:61` | 1 | comment | `constexpr ~Allocation() noexcept { }` |
| `src/alloc/gc_simple.h:474` | 3 | comment | `if (alloc->dtor) {` |
| `src/alloc/gc_simple.h:554` | 3 | comment | `if (chunk->dtor) {` |

## src/atomic (15)

| location | lines | kind | first line |
|---|---|---|---|
| `src/atomic/atomic_primitives.h:371` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:375` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:379` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:383` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:387` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:391` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:395` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:399` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:403` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:407` | 3 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:411` | 10 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:422` | 10 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:433` | 11 | comment | `template<typename _Tp>` |
| `src/atomic/atomic_primitives.h:445` | 11 | comment | `template<typename _Tp>` |
| `src/atomic/semaphore.h:90` | 167 | #if 0 | `// #include "coro/semaphore.hpp"` |

## src/combination (13)

| location | lines | kind | first line |
|---|---|---|---|
| `src/combination/chase.h:620` | 4 | comment | `for (size_t i = 1; i <= t; ++i) {` |
| `src/combination/grey.h:74` | 10 | comment | `T h=1, r=0;` |
| `src/combination/grey.h:85` | 4 | comment | `T r = BITS_PER_LONG;` |
| `src/combination/lexicographic.h:352` | 7 | comment | `if ( 0==x )  return 0;` |
| `src/combination/lexicographic.h:360` | 12 | comment | `T r = 0;` |
| `src/combination/lexicographic.h:373` | 10 | comment | `T r = 0;` |
| `src/combination/min_change.h:142` | 2 | comment | `=^=  (by Doug Moore)` |
| `src/combination/necklace.h:88` | 1 | #if 0 | `j_ = highest_zero_idx( a_ ^ (~mm_)  );` |
| `src/combination/necklace.h:98` | 1 | comment | `a_ = bit_copy_periodic(a_, j_);  a_ &= mm_;` |
| `src/combination/paren.h:105` | 1 | comment | `if ( s<0 )  {"Invalid word"}` |
| `src/combination/sequency.h:46` | 2 | comment | `return inverse_gray_code( first_comb(k) );` |
| `src/combination/sequency.h:49` | 3 | comment | `T s = ((1UL<<k) - 1);` |
| `src/combination/shifts.h:85` | 2 | #else of #if 1 | `// Alternative: attach block using bit-scan operations` |

## src/compression (18)

| location | lines | kind | first line |
|---|---|---|---|
| `src/compression/NLZM.cpp:792` | 1 | comment | `uint32 remh = 1; for (int i = 0; i < BLOCK_SIZE; i++) { remh *= addh; }` |
| `src/compression/NLZM.cpp:1134` | 1 | comment | `const static int CMD_RolzDict = 3;` |
| `src/compression/NLZM.cpp:1196` | 1 | comment | `cdf_init(m.dist_lo);` |
| `src/compression/NLZM.cpp:1231` | 5 | comment | `if (add_bits > 4) {` |
| `src/compression/NLZM.cpp:1237` | 1 | comment | `}` |
| `src/compression/NLZM.cpp:1333` | 3 | comment | `dv_add &= 0xF;` |
| `src/compression/NLZM.cpp:1406` | 3 | comment | `uint32 dv_lo = frame.ReadCDF(m.dist_lo);` |
| `src/compression/NLZM.cpp:1650` | 4 | comment | `void RolzDict::Init() {` |
| `src/compression/NLZM.cpp:1655` | 4 | comment | `void RolzDict::Add(int ctx, uint32 pos, uint16 len) {` |
| `src/compression/NLZM.cpp:1660` | 1 | comment | `RolzEntry &e = cells[ctx][head[ctx]++ & TABLE_MASK];` |
| `src/compression/NLZM.cpp:1662` | 3 | comment | `e.pos = pos;` |
| `src/compression/NLZM.cpp:1666` | 3 | comment | `RolzEntryMatch RolzDict::Lookup(int ctx, uint32 pos) {` |
| `src/compression/NLZM.cpp:1670` | 2 | comment | `uint32 lo = 0;` |
| `src/compression/NLZM.cpp:1673` | 15 | comment | `while (lo < hi) {` |
| `src/compression/NLZM.cpp:1689` | 2 | comment | `return m;` |
| `src/compression/horizontalpacking32.c:31` | 34 | comment | `static void simdhunpack1(const uint8_t *  in,uint32_t *  out) {` |
| `src/compression/horizontalpacking32.c:171` | 48 | comment | `static  void simdhunpack4(const uint8_t *  in,uint32_t *  out) {` |
| `src/compression/lzf.h:119` | 4 | #if 0 | `/* original lzv-like hash function, much worse and thus slower */` |

## src/container (74)

| location | lines | kind | first line |
|---|---|---|---|
| `src/container/array.h:345` | 4 | comment | `template<typename T, size_t N>` |
| `src/container/array.h:352` | 8 | comment | `template<typename E, typename ...Es>` |
| `src/container/array.h:362` | 4 | comment | `template<typename T, size_t N>` |
| `src/container/array.h:367` | 4 | comment | `template<typename T, size_t N>` |
| `src/container/array.h:373` | 5 | comment | `template<typename T, size_t N, size_t M>` |
| `src/container/array.h:392` | 5 | comment | `template<typename F, typename T, size_t N>` |
| `src/container/array.h:399` | 5 | comment | `template<typename P, typename T, size_t N>` |
| `src/container/array.h:406` | 7 | comment | `namespace detail {` |
| `src/container/array.h:414` | 5 | comment | `template<typename T, size_t N>` |
| `src/container/binary_packed_vector.h:61` | 3 | comment | `using M = FqPackedVectorMeta<_n, 2, T, true>;` |
| `src/container/binary_packed_vector.h:1557` | 1 | comment | `For b[i] = __x;` |
| `src/container/binary_packed_vector.h:1566` | 1 | comment | `For b[i] = b[__j];` |
| `src/container/binary_packed_vector.h:1579` | 1 | comment | `For __x = b[i];` |
| `src/container/binary_packed_vector.h:1584` | 1 | comment | `For b[i].flip();` |
| `src/container/binary_packed_vector.h:1654` | 1 | comment | `T data(uint64_t index) { assert(index < length); return get_bit_shifted(index); }` |
| `src/container/binary_packed_vector.h:1698` | 4 | comment | `using S = TxN_t<T, limbs()>;` |
| `src/container/binary_packed_vector.h:1723` | 4 | comment | `constexpr uint16_t bytes = (length+7)/8;` |
| `src/container/critbit.h:12` | 2 | comment | `struct critbit_key;` |
| `src/container/critbit.h:64` | 2 | comment | `assert(t->node_free != NULL);` |
| `src/container/critbit.h:240` | 1 | comment | `return critbit_ref_get_key(ref);` |
| `src/container/critbit.h:257` | 1 | comment | `const uint8_t *const ubytes = (const uint8_t *const)&key;` |
| `src/container/critbit.h:286` | 1 | comment | `pkey = (const uint8_t *)(critbit_ref_get_key(p));` |
| `src/container/ctrie.h:362` | 3 | comment | `using SCacheAllocator = CacheAllocator<SNode>;` |
| `src/container/ctrie.h:552` | 2 | comment | `auto *n = sll.allocate();` |
| `src/container/ctrie.h:735` | 3 | comment | `cache_size = 1024;` |
| `src/container/ctrie.h:941` | 2 | comment | `auto *xn = new (std::align_val_t(alignment)) XNode {parent, parentPos, current, hash, level};` |
| `src/container/ctrie.h:1347` | 1 | comment | `auto *tmp_node = (ANode *) accessNode(node);` |
| `src/container/ctrie.h:1706` | 3 | comment | `ANode *told = (ANode *) accessNode(old);` |
| `src/container/ctrie.h:1930` | 3 | comment | `ANode *told = (ANode *) accessNode(old);` |
| `src/container/deque.h:191` | 1 | comment | `x_ = ReAlloc<Type>(x_, ns, s_);` |
| `src/container/fq_packed_vector.h:164` | 1 | comment | `return Hash<uint64_t, 0, n, q>::hash((uint64_t *)ptr());` |
| `src/container/fq_packed_vector.h:1136` | 1 | comment | `For b[i] = __x;` |
| `src/container/fq_packed_vector.h:1144` | 1 | comment | `For b[i] = b[__j];` |
| `src/container/fq_packed_vector.h:1151` | 1 | comment | `For __x = b[i];` |
| `src/container/fq_packed_vector.h:1158` | 6 | comment | `[[nodiscard]] constexpr operator kAry_Type_T<q>() noexcept {` |
| `src/container/fq_packed_vector.h:1884` | 12 | comment | `template<const uint32_t n,` |
| `src/container/fq_packed_vector.h:1937` | 2 | comment | `constexpr TT m = TT(1) << (I::bits_per_number - 1u);` |
| `src/container/fq_packed_vector_v2.h:99` | 1 | comment | `return Hash<uint64_t, 0, n, q>::hash((uint64_t *)ptr());` |
| `src/container/fq_vector.h:102` | 4 | comment | `using S_ = TxN_t<T, limbs()>;` |
| `src/container/fq_vector.h:1203` | 1 | comment | `using M::__data;` |
| `src/container/fq_vector.h:1246` | 1 | comment | `using M::__data;` |
| `src/container/fq_vector.h:1938` | 2 | comment | `a = ((a >> 4) & mask_f) + (a & mask_f);` |
| `src/container/hashmap/hopscotch_hash.h:541` | 1 | comment | `"The map exceeds its maximum size.");` |
| `src/container/hashmap/hopscotch_hash.h:579` | 1 | comment | `"The map exceeds its maximum size.");` |
| `src/container/hashmap/hopscotch_hash.h:1208` | 3 | comment | `#ifndef TSL_HH_NO_EXCEPTIONS` |
| `src/container/hashmap/hopscotch_hash.h:1239` | 7 | comment | `const bool use_stored_hash =` |
| `src/container/hashmap/hopscotch_hash.h:1247` | 4 | comment | `const std::size_t hash =` |
| `src/container/hashmap/hopscotch_hash.h:1258` | 3 | comment | `throw;` |
| `src/container/hashmap/lock_free_hopscotch.h:45` | 1 | comment | `typedef BrownOriginal<Allocator, MemReclaimer, std::size_t(3)> KCAS;` |
| `src/container/hashmap/lock_free_hopscotch.h:157` | 1 | comment | `m_table[current_bucket].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:200` | 1 | comment | `m_table[current_bucket].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:216` | 1 | comment | `const VersionState vw = cur_bucket.vw.load();` |
| `src/container/hashmap/lock_free_hopscotch.h:222` | 1 | comment | `print_state(vw.state);` |
| `src/container/hashmap/lock_free_hopscotch.h:235` | 2 | comment | `m_table[i].timestamp.store(std::uintptr_t(0),` |
| `src/container/hashmap/lock_free_hopscotch.h:254` | 1 | comment | `m_table[hash].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:279` | 1 | comment | `m_table[hash].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:296` | 1 | comment | `m_table[hash].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:323` | 1 | comment | `m_table[hash].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/lock_free_hopscotch.h:433` | 56 | comment | `Move back keys (optional)` |
| `src/container/hashmap/lock_free_hopscotch.h:504` | 1 | comment | `m_table[hash].timestamp.load(std::memory_order_relaxed);` |
| `src/container/hashmap/simple.h:341` | 3 | comment | `alignas(1024) data_type __internal_hashmap_array[total_size];` |
| `src/container/imap.h:384` | 1 | comment | `newtree = (imap_node_t *)IMAP_ALIGNED_ALLOC(sizeof(imap_node_t), newsize);` |
| `src/container/imap.h:470` | 1 | comment | `IMAP_ALIGNED_FREE(tree);` |
| `src/container/kAry_type.h:57` | 1 | comment | `using T = TypeTemplate<q>;` |
| `src/container/kAry_type.h:498` | 4 | comment | `FqElement &operator=(T const obj) {` |
| `src/container/kAry_type.h:503` | 4 | comment | `FqElement &operator=(unsigned int const obj) {` |
| `src/container/kAry_type.h:508` | 4 | comment | `FqElement &operator=(unsigned long obj) {` |
| `src/container/kAry_type.h:596` | 1 | comment | `TOOD was deactivated due to some bug in rho static_assert(upper <= bits);` |
| `src/container/priorityqueue.h:22` | 4 | #else of #if 1 | `// next() is the one with the biggest key` |
| `src/container/priorityqueue.h:154` | 1 | comment | `heapify(m);` |
| `src/container/ringbuffer.h:35` | 1 | comment | `x_ = new Type[s_];` |
| `src/container/ringbuffer.h:42` | 1 | comment | `delete [] x_;` |
| `src/container/stack.h:38` | 1 | comment | `x_ = (Type *)std::malloc( s_ * sizeof(Type) );` |
| `src/container/stack.h:44` | 1 | comment | `std::free( x_ );` |

## src/crypto (10)

| location | lines | kind | first line |
|---|---|---|---|
| `src/crypto/aes.h:351` | 7 | comment | `template<const uint8_t version>` |
| `src/crypto/aes.h:359` | 4 | comment | `for (uint8_t i = 0; i < 16; i++) {` |
| `src/crypto/aes.h:364` | 2 | comment | `expand_key(key_schedule, version);` |
| `src/crypto/aes.h:367` | 4 | comment | `for (uint8_t i = 0; i < 16; i++) {` |
| `src/crypto/aes.h:372` | 2 | comment | `add_round_key(state, round_key);` |
| `src/crypto/aes.h:375` | 4 | comment | `for (uint8_t i = 1; i < rounds; i++) {` |
| `src/crypto/aes.h:380` | 3 | comment | `for (uint8_t j = 0; j < 16; j++) {` |
| `src/crypto/aes.h:384` | 4 | comment | `add_round_key(state, round_key);` |
| `src/crypto/aes.h:389` | 4 | comment | `for (uint8_t i = 0; i < 16; i++) {` |
| `src/crypto/aes.h:394` | 5 | comment | `inv_shift_rows(state);` |

## src/graph (2)

| location | lines | kind | first line |
|---|---|---|---|
| `src/graph/digraph.h:1126` | 1 | comment | `if ( 0==fct )  { "dead end: this is a U-turn"; }` |
| `src/graph/digraph.h:1189` | 1 | comment | `if ( 0==fct )  { "dead end: this is a U-turn"; }` |

## src/hash (1)

| location | lines | kind | first line |
|---|---|---|---|
| `src/hash/crc.h:343` | 9 | comment | `using hardware crc instructions to generate lut` |

## src/list (18)

| location | lines | kind | first line |
|---|---|---|---|
| `src/list/enumeration/ternary.h:151` | 1 | comment | `std::cout << L1.data_value(ctr) << " \| " << L2.data_value(ctr) << " \| " << L3.data_value(ctr) << " \| " << L4.data_value(` |
| `src/list/enumeration/ternary.h:163` | 1 | comment | `std::cout << CL.first << ":" << CL.second << "\n";` |
| `src/list/enumeration/ternary.h:202` | 1 | comment | `std::cout << data << " \| " << data2 << ", " << cl.first << ":" << cl.second << "\n";` |
| `src/list/enumeration/ternary.h:267` | 1 | comment | `std::cout << data << " \| " << data2 << ", " << cl.first << ":" << cl.second << "\n";` |
| `src/list/enumeration/ternary.h:284` | 1 | comment | `constexpr uint64_t sbc1 = sum_bc(LeftRightSplitLen, config.nr1);` |
| `src/list/enumeration/ternary.h:411` | 2 | comment | `std::cout << data << " \| " << data2 << ", " << cl.first << ":" << cl.second << "\n";` |
| `src/list/enumeration/ternary.h:420` | 6 | comment | `uint64_t clctr1 = 0, i_clctr1 = 0, clctr2 = 0, i_clctr2 = 0;` |
| `src/list/enumeration/ternary.h:427` | 3 | comment | `if (RepsLessPartLen != 0) {` |
| `src/list/enumeration/ternary.h:431` | 1 | comment | `std::cout << ", " << clctr1 << ", " << i_clctr1 << ", " << clctr2 << ", " << i_clctr2 << "\n";` |
| `src/list/enumeration/ternary.h:433` | 4 | comment | `i_clctr2 += 1;` |
| `src/list/enumeration/ternary.h:438` | 6 | comment | `if (i_clctr1 == ccL1[clctr1].size()) {` |
| `src/list/enumeration/ternary.h:445` | 5 | comment | `if (clctr2 == config.nr2) {` |
| `src/list/enumeration/ternary.h:470` | 4 | comment | `const uint64_t val = start + in*len;` |
| `src/list/enumeration/ternary.h:691` | 5 | comment | `Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i]]);` |
| `src/list/enumeration/ternary.h:707` | 5 | comment | `Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i] + LeftRightSplit + RepsLessPartLen * 0]);` |
| `src/list/enumeration/ternary.h:726` | 3 | comment | `if constexpr(config.threads != 1) {` |
| `src/list/enumeration/ternary.h:980` | 1 | comment | `std::cout << L1.data_label(i-1) << "->" << L1.data_label(i) << ", " << cL1[i].first << ":" << cL1[i].second << ", " << i` |
| `src/list/list.h:153` | 1 | comment | `using TMP = decltype(e.label.hash());` |

## src/math (21)

| location | lines | kind | first line |
|---|---|---|---|
| `src/math/bc.h:153` | 9 | comment | `float version` |
| `src/math/bc.h:232` | 1 | comment | `a -= ((wn-1u) *(wn-2u)) >> 1u;` |
| `src/math/bc.h:316` | 1 | comment | `a -= ((wn-1u) *(wn-2u)) >> 1u;` |
| `src/math/bc.h:349` | 1 | comment | `a -= ((wn-1u) *(wn-2u)) >> 1u;` |
| `src/math/bigint/bigint.h:301` | 5 | comment | `template<typename T, size_t M, size_t N>` |
| `src/math/bigint/bigint.h:315` | 4 | comment | `template<size_t padding_limbs = 0U, size_t M, size_t N, typename T>` |
| `src/math/bigint/bigint.h:320` | 12 | comment | `big_int<M + N + padding_limbs, T> w{};` |
| `src/math/bigint/bigint.h:333` | 5 | comment | `template<typename Q, typename R>` |
| `src/math/bigint/bigint.h:346` | 6 | comment | `template<size_t M, size_t N, typename T>` |
| `src/math/bigint/bigint.h:365` | 4 | comment | `using TT = typename dbl_bitlen<T>::type;` |
| `src/math/bigint/bigint.h:373` | 1 | comment | `big_int<M, T> q{};` |
| `src/math/bigint/bigint.h:375` | 9 | comment | `if (tight_N == 1) {// short division` |
| `src/math/bigint/bigint.h:385` | 7 | comment | `uint8_t k = 0;` |
| `src/math/bigint/bigint.h:393` | 6 | comment | `for (int j = M - tight_N; j >= 0; --j) {` |
| `src/math/bigint/bigint.h:400` | 26 | comment | `auto b = static_cast<TT>(1) << std::numeric_limits<T>::digits;` |
| `src/math/bigint/bigint.h:456` | 5 | comment | `template<typename T, size_t N1, size_t N2>` |
| `src/math/bigint/bigint.h:462` | 5 | comment | `template<typename T, size_t N1, size_t N2>` |
| `src/math/bigint/bigint.h:468` | 5 | comment | `template<typename T, size_t N1, size_t N2>` |
| `src/math/bigint/bigint.h:480` | 5 | comment | `template<typename T, size_t N1, size_t N2>` |
| `src/math/log.h:132` | 2 | comment | `if constexpr (std::is_constant_evaluated()) {` |
| `src/math/log.h:135` | 3 | comment | `} else {` |

## src/matrix (10)

| location | lines | kind | first line |
|---|---|---|---|
| `src/matrix/binary_matrix.h:1308` | 2 | comment | `continue;` |
| `src/matrix/binary_matrix.h:1320` | 1 | comment | `RowT::add(row(j), row(j), row(i));` |
| `src/matrix/binary_matrix.h:1359` | 1 | comment | `RowT::add(row(j), row(j), row(i + c));` |
| `src/matrix/binary_matrix.h:2285` | 17 | #else of #if 1 | `// The same as the above, but without using _mzd_copy_transpose_64x64_2.` |
| `src/matrix/matrix.h:80` | 3 | comment | `MatrixType::transpose(c, c);` |
| `src/matrix/matrix.h:120` | 1 | comment | `typedef __RowType RowType;` |
| `src/matrix/matrix.h:1349` | 1 | comment | `using DataType = typename LabelType::DataType;` |
| `src/matrix/matrix.h:1414` | 1 | comment | `std::cout << std::setw(3) << j << ": ";` |
| `src/matrix/matrix.h:1432` | 1 | comment | `std::cout << std::setw(3) << j << ": ";` |
| `src/matrix/matrix.h:1442` | 1 | comment | `std::cout << std::setw(3) << i << ": ";` |

## src/memory (2)

| location | lines | kind | first line |
|---|---|---|---|
| `src/memory/memset.h:108` | 1 | comment | `case 16 < bytes <= 32;` |
| `src/memory/memset.h:144` | 1 | comment | `case 16 < bytes <= 32;` |

## src/nn (11)

| location | lines | kind | first line |
|---|---|---|---|
| `src/nn/nn.h:220` | 1 | comment | `choose e2;` |
| `src/nn/nn.h:364` | 1 | comment | `std::cout << solutions_nr << "\n";` |
| `src/nn/nn.h:1341` | 1 | comment | `assert(uint64_t(wt) < (1ull << 32ull));` |
| `src/nn/nn.h:1456` | 23 | comment | `#pragma unroll 4` |
| `src/nn/nn.h:1800` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:1838` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:2005` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:2017` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:2028` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:2039` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |
| `src/nn/nn.h:2091` | 1 | comment | `std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";` |

## src/simd (12)

| location | lines | kind | first line |
|---|---|---|---|
| `src/simd/avx2.h:140` | 21 | comment | `constexpr static void m256tou16(uint16_t t[16],` |
| `src/simd/avx2.h:4448` | 9 | comment | `if (std::is_constant_evaluated()) {` |
| `src/simd/avx2_transpose.h:54` | 1 | comment | `CHECK_MULT_EIGHT(start);` |
| `src/simd/avx512.h:533` | 2 | comment | `out.v512 = _mm512_slli_epi16(in1.v512, in2);` |
| `src/simd/avx512.h:548` | 2 | comment | `const S mask = S::set1((1u << ((8u - in2) & 7u)) - 1u);` |
| `src/simd/avx512.h:551` | 1 | comment | `out = S::and_(out, mask);` |
| `src/simd/avx512_intrinsics.h:1536` | 2 | comment | `count = _mm_or_si128(count, _mm_set_epi32(0x38302820, 0x18100800, 0x38302820, 0x18100800));` |
| `src/simd/bits/avx2.h:31` | 1 | comment | `mask \|= mask<<1 \| mask<<2 \| ... \| mask<<7;` |
| `src/simd/bits/avx2.h:61` | 1 | comment | `mask \|= mask<<1 \| mask<<2 \| ... \| mask<<7;` |
| `src/simd/bits/bits.h:35` | 2 | comment | `using _pdep_u32_ = pdep<uint32_t>;` |
| `src/simd/simd.h:5824` | 4 | comment | `{ S::set(ll) } -> std::convertible_to<S>;` |
| `src/simd/simd.h:5866` | 1 | comment | `{ S::mask(u32) } -> std::convertible_to<S>;` |

## src/sort (41)

| location | lines | kind | first line |
|---|---|---|---|
| `src/sort/bit_sort.h:143` | 1 | comment | `simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosRight]);` |
| `src/sort/bit_sort.h:145` | 2 | comment | `std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(` |
| `src/sort/bit_sort.h:150` | 1 | comment | `simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosLeft]);` |
| `src/sort/bit_sort.h:152` | 2 | comment | `std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(` |
| `src/sort/bit_sort.h:169` | 1 | comment | `restMask = simd::kshiftr(simd::knot(simd::Mask<_numElemsPerVec>(0)), _numElemsPerVec - numElemsRest);` |
| `src/sort/bit_sort.h:171` | 1 | comment | `keyVecRest = simd::maskz_loadu<_numElemsPerVec * sizeof(K)>(restMask, &keys[readPosLeft]);` |
| `src/sort/bit_sort.h:173` | 1 | comment | `payloadVecRest = std::make_tuple(simd::maskz_loadu<_numElemsPerVec * sizeof(Ps)>(restMask, &payloads[readPosLeft])...);` |
| `src/sort/bit_sort.h:194` | 1 | comment | `simd::kand(sortMaskLeftRest, restMask);` |
| `src/sort/bit_sort.h:196` | 1 | comment | `simd::kand(sortMaskRightRest, restMask);` |
| `src/sort/bit_sort.h:198` | 1 | comment | `simd::kpopcnt(sortMaskLeftRest);` |
| `src/sort/bit_sort.h:221` | 1 | comment | `simd::test_bit(keyVec, bitNo);` |
| `src/sort/bit_sort.h:223` | 1 | comment | `simd::knot(sortMaskRight);` |
| `src/sort/bit_sort.h:227` | 1 | comment | `simd::test_bit(keyVec, bitNo);` |
| `src/sort/bit_sort.h:229` | 1 | comment | `simd::knot(sortMaskLeft);` |
| `src/sort/divsufsort.h:1397` | 1 | comment | `trbudget_init(&budget, tr_ilg(n) * 3 / 4, n);` |
| `src/sort/djb_sort.h:643` | 1 | comment | `everything is still masked with _mm256_set_epi32(0,-1,0,-1,0,-1,0,-1);` |
| `src/sort/ips4o.h:198` | 1 | comment | `memcpy(storage_, src, Cfg::kBlockSizeInBytes);` |
| `src/sort/ips4o.h:204` | 1 | comment | `memcpy(block.storage_, storage_, Cfg::kBlockSizeInBytes);` |
| `src/sort/ips4o.h:210` | 1 | comment | `memcpy(dest, storage_, Cfg::kBlockSizeInBytes);` |
| `src/sort/ips4o.h:269` | 1 | comment | `memcpy(dest, bucket, Cfg::kBlockSizeInBytes);` |
| `src/sort/ips4o.h:600` | 1 | comment | `HWY_ABORT("add case for log_buckets %zu\n", log_buckets);` |
| `src/sort/ips4o.h:615` | 2 | comment | `b[i] = 2 * b[i] + comp_(splitter(b[i]),` |
| `src/sort/ips4o.h:692` | 2 | comment | `b[i] = 2 * b[i] + !comp_(begin[i], sortedSplitter(b[i] -` |
| `src/sort/ips4o.h:821` | 3 | comment | `left-most)` |
| `src/sort/ips4o.h:843` | 3 | comment | `auto write_ptr = begin_ + std::max(my_first_empty_block, bucket_start);` |
| `src/sort/ips4o.h:985` | 2 | comment | `memcpy(dst, src, remaining * sizeof(value_type));` |
| `src/sort/ips4o.h:1026` | 1 | comment | `memcpy(dst, src, remaining * sizeof(value_type));` |
| `src/sort/merge_sort.h:68` | 1 | comment | `if (tmp==nullptr)  t = new Type[n];` |
| `src/sort/merge_sort.h:71` | 1 | comment | `if (tmp==nullptr)  delete [] t;` |
| `src/sort/merge_sort.h:112` | 1 | comment | `if (tmp==nullptr)  t = new Type[n];` |
| `src/sort/merge_sort.h:115` | 1 | comment | `if (tmp==nullptr)  delete [] t;` |
| `src/sort/radixsort.h:41` | 1 | comment | `ALLOCA(uint64_t, cv, nb);` |
| `src/sort/radixsort.h:43` | 1 | comment | `size_t cv[nb] = {0};` |
| `src/sort/sorting_network/avx2.h:849` | 1 | comment | `CMPXCH_SHUFFLE2(a, b, perm4, sel4);` |
| `src/sort/sorting_network/avx2.h:1892` | 1 | comment | `avx2_sortingnetwork_small(f, float, __m512);` |
| `src/sort/sorting_network/avx512.h:494` | 2 | comment | `inline void CoreExchangeSort2V(__m512i& input, __m512i& input2,` |
| `src/sort/sorting_network/avx512.h:2465` | 1 | comment | `avx512_sortingnetwork_small(f, float, __m512);` |
| `src/sort/sorting_network/hunter.h:174` | 1 | comment | `Pair_t InitialNetwork[0];` |
| `src/sort/sorting_network/hunter.h:654` | 1 | comment | `RandGen_t &rndgen(0);` |
| `src/sort/sorting_network/hunter.h:855` | 2 | comment | `assert(failvector<PARWORDSIZE);` |
| `src/sort/sorting_network/sorting_network.h:56` | 2 | comment | `Set {a, b} := {min(a, b), max(a,b)}` |

## src/thread (14)

| location | lines | kind | first line |
|---|---|---|---|
| `src/thread/execution.h:27` | 2 | comment | `using pool_type_ = HeartbeatScheduler<>;` |
| `src/thread/mythread.h:336` | 1 | comment | `DEBUG_PRINTF("I am idle\n");` |
| `src/thread/mythread.h:404` | 3 | comment | `futex_init(&main_tcb->sched_futex, 1);` |
| `src/thread/mythread.h:451` | 2 | comment | `futex_init(&gfutex, 1);` |
| `src/thread/mythread.h:496` | 2 | comment | `futex_init(&new_node->sched_futex, 0);` |
| `src/thread/old/heartbeat_scheduler.h:335` | 2 | comment | `return std::invoke(func, &t, arg...);` |
| `src/thread/old/heartbeat_scheduler.h:351` | 1 | comment | `backgroundWorker();` |
| `src/thread/old/heartbeat_scheduler.h:473` | 1 | comment | `return std::future(call<ReturnType, Function, Args...>(f, args...));` |
| `src/thread/old/heartbeat_scheduler.h:552` | 1 | comment | `TODO best_worker->shared_job->reset();` |
| `src/thread/old/heartbeat_scheduler.h:561` | 2 | comment | `std::lock_guard lock(mutex);` |
| `src/thread/simple.h:247` | 1 | comment | `try {` |
| `src/thread/simple.h:249` | 1 | comment | `} catch (...) { }` |
| `src/thread/steal.h:205` | 8 | comment | `use shared promise here so that we don't break the promise later (until C++23)` |
| `src/thread/thread.h:9` | 4 | comment | `a.submit();` |

## src/tree (5)

| location | lines | kind | first line |
|---|---|---|---|
| `src/tree/d1.h:392` | 1 | comment | `hm.clear();` |
| `src/tree/d1_thread.h:26` | 1 | comment | `using LoadType = typename HML0::load_type;` |
| `src/tree/d2_stream.h:272` | 1 | comment | `out.add_and_append(tmpe1, L2[l], k_lower1, k_upper2, filter);` |
| `src/tree/dissection.h:23` | 1 | comment | `constexpr size_t baselist_size = sum_bc(n/2, n/4);` |
| `src/tree/dissection.h:27` | 5 | comment | `using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;` |

## tests (8)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/test_build_tree.cpp:208` | 1 | comment | `std::cout << out[i];` |
| `tests/test_build_tree.cpp:210` | 1 | comment | `std::cout << out[i];` |
| `tests/test_element.cpp:63` | 3 | comment | `std::cout << l1;` |
| `tests/test_helper.py:15` | 1 | comment | `endif"""` |
| `tests/test_helper.py:17` | 1 | comment | `dict2include("./test.h", {"a": 1})` |
| `tests/test_reflection.cpp:50` | 4 | comment | `constexpr auto t = reflect::to<std::tuple>(f);` |
| `tests/warp_scan.cu:51` | 1 | #else of #if 1 | `return warp_scan_inc_up_add_shuffled(v,width) - v;` |
| `tests/warp_scan.cu:131` | 1 | comment | `uint32_t const vin_h[32] = { 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1 };` |

## tests/algorithm (65)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/algorithm/histogram.cpp:154` | 1 | comment | `EXPECT_EQ(cnt[0], s);` |
| `tests/algorithm/mq/fes.cpp:27` | 2 | comment | `for (uint32_t i = 0; i < 496; i++) { Fq_internal[i] = rng() & mask; }` |
| `tests/algorithm/mq/fes.cpp:122` | 1 | comment | `const uint32_t x = (1<<9);` |
| `tests/algorithm/mq/fes.cpp:124` | 6 | comment | `for (uint32_t x = 0; x < (1u << 10); x++) {` |
| `tests/algorithm/pcs.cpp:55` | 5 | comment | `using T   = uint64_t;` |
| `tests/algorithm/pcs.cpp:61` | 5 | comment | `Label target(0);` |
| `tests/algorithm/pcs.cpp:67` | 2 | comment | `std::vector<uint32_t> sol;` |
| `tests/algorithm/pcs.cpp:86` | 4 | comment | `if (alb == blb) {` |
| `tests/algorithm/pcs.cpp:91` | 3 | comment | `return a2.label == b2.label;` |
| `tests/algorithm/pcs.cpp:95` | 9 | comment | `auto f = [&instance,` |
| `tests/algorithm/pcs.cpp:105` | 13 | comment | `if (in.value() & 1u) {` |
| `tests/algorithm/pcs.cpp:119` | 3 | comment | `assert(ret.value.popcnt() == n4);` |
| `tests/algorithm/pcs.cpp:123` | 7 | comment | `while(true) {` |
| `tests/algorithm/pcs.cpp:131` | 4 | comment | `a.label.random();` |
| `tests/algorithm/pcs.cpp:136` | 2 | comment | `a = f(a); b = f(b);` |
| `tests/algorithm/pcs.cpp:139` | 7 | comment | `std::cout << "solution: ";` |
| `tests/algorithm/pcs.cpp:147` | 12 | comment | `for (uint32_t i = 0; i < n; ++i) {` |
| `tests/algorithm/pcs.cpp:160` | 4 | comment | `Label c1(0);` |
| `tests/algorithm/pcs.cpp:166` | 4 | comment | `Label c2(0); Value v2;` |
| `tests/algorithm/pcs.cpp:171` | 5 | comment | `std::cout << c1 << std::endl;` |
| `tests/algorithm/pcs.cpp:177` | 1 | comment | `}` |
| `tests/algorithm/pcs.cpp:247` | 2 | comment | `const auto el = out[i];` |
| `tests/algorithm/prefixsum.cpp:104` | 1 | comment | `for (uint32_t i = 0; i < s; i++) { d1[i] = fastrandombytes_uint64() % (1u << 8u); }` |
| `tests/algorithm/regex.cpp:135` | 1 | comment | `printf("\n");` |
| `tests/algorithm/search.cpp:61` | 4 | comment | `const auto r1 = cryptanalysislib::search(par_if(true), in.begin(), in.end(), 1);` |
| `tests/algorithm/subsetsum/BCJ.cpp:43` | 1 | comment | `using Value  = S::Value;` |
| `tests/algorithm/subsetsum/HGJ.cpp:25` | 1 | comment | `using Value  = S::Value;` |
| `tests/algorithm/subsetsum/HGJ.cpp:44` | 1 | comment | `using Value  = S::Value;` |
| `tests/algorithm/subsetsum/dissection.cpp:26` | 1 | comment | `using Value      = kAryPackedContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/dissection.cpp:62` | 1 | comment | `std::cout << target << ":" << out[i].label << std::endl;` |
| `tests/algorithm/subsetsum/generic.cpp:24` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/generic.cpp:34` | 1 | comment | `constexpr uint32_t k_lower1=0, k_higher1=n/3, k_higher2=2*n/3, k_higher3=n;` |
| `tests/algorithm/subsetsum/generic.cpp:51` | 3 | comment | `auto nr_iTs = [](const uint32_t i = 0 ){` |
| `tests/algorithm/subsetsum/generic.cpp:85` | 2 | comment | `if (out[i].value.popcnt() != n/2) { continue; }` |
| `tests/algorithm/subsetsum/join2lists.cpp:27` | 1 | comment | `using Value      = kAryPackedContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/join4lists.cpp:25` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/join4lists.cpp:109` | 1 | comment | `constexpr size_t baselist_size = sum_bc(n/4, n/8);` |
| `tests/algorithm/subsetsum/join4lists.cpp:175` | 1 | comment | `constexpr size_t baselist_size = sum_bc(n/4, n/8);` |
| `tests/algorithm/subsetsum/join4lists.cpp:233` | 4 | comment | `std::cout << out[i] << std::endl;` |
| `tests/algorithm/subsetsum/join8lists.cpp:41` | 2 | comment | `constexpr size_t baselist_size = sum_bc(n/2, n/4);` |
| `tests/algorithm/subsetsum/join8lists.cpp:44` | 6 | comment | `std::vector<List> L{12};` |
| `tests/algorithm/subsetsum/join8lists.cpp:51` | 3 | comment | `Label target;` |
| `tests/algorithm/subsetsum/join8lists.cpp:55` | 2 | comment | `Tree t{1, A, 0};` |
| `tests/algorithm/subsetsum/join8lists.cpp:58` | 4 | comment | `uint32_t right=0;` |
| `tests/algorithm/subsetsum/join8lists.cpp:63` | 9 | comment | `Label test_recalc1(0), test_recalc2(0), test_recalc3(0);` |
| `tests/algorithm/subsetsum/join8lists.cpp:73` | 7 | comment | `EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, 0, n));` |
| `tests/algorithm/subsetsum/join8lists.cpp:81` | 2 | comment | `EXPECT_GT(right,0);` |
| `tests/algorithm/subsetsum/join8lists.cpp:180` | 1 | comment | `std::cout << out[i] << std::endl;` |
| `tests/algorithm/subsetsum/join8lists.cpp:252` | 1 | comment | `std::cout << out[i] << std::endl;` |
| `tests/algorithm/subsetsum/join_for.cpp:25` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/join_stream.cpp:25` | 1 | comment | `using Value      = kAryContainer_T<T, n, 2>;` |
| `tests/algorithm/subsetsum/parameters.cpp:49` | 1 | comment | `using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;` |
| `tests/algorithm/subsetsum/parameters.cpp:69` | 7 | comment | `using T    = uint64_t;` |
| `tests/algorithm/subsetsum/parameters.cpp:77` | 2 | comment | `Matrix A; A.random();` |
| `tests/algorithm/subsetsum/parameters.cpp:80` | 1 | comment | `using Enumerator = BinarySinglePartialSingleEnumerator<List, 32, 1, 2, 16>;` |
| `tests/algorithm/subsetsum/parameters.cpp:82` | 5 | comment | `constexpr size_t baselist_size = Enumerator::LIST_SIZE;` |
| `tests/algorithm/subsetsum/parameters.cpp:88` | 4 | comment | `Tree t{1, A, 0};` |
| `tests/algorithm/subsetsum/parameters.cpp:93` | 2 | comment | `EXPECT_GT(out.load(), 0);` |
| `tests/algorithm/subsetsum/parameters.cpp:103` | 7 | comment | `using T    = uint64_t;` |
| `tests/algorithm/subsetsum/parameters.cpp:111` | 5 | comment | `Matrix A; A.random();` |
| `tests/algorithm/subsetsum/parameters.cpp:117` | 2 | comment | `using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, p>;` |
| `tests/algorithm/subsetsum/parameters.cpp:120` | 3 | comment | `Label target;` |
| `tests/algorithm/subsetsum/parameters.cpp:124` | 4 | comment | `Tree t{1, A, 0};` |
| `tests/algorithm/subsetsum/parameters.cpp:129` | 2 | comment | `EXPECT_GT(out.load(), 0);` |
| `tests/algorithm/subsetsum/tmto.cpp:46` | 1 | comment | `using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;` |

## tests/alloc (1)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/alloc/dev.cpp:111` | 2 | comment | `EXPECT_EQ(s.owns(b),  false);` |

## tests/binary (11)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/binary/build_tree.cpp:53` | 1 | comment | `std::cout << out[i].get_label();` |
| `tests/binary/build_tree.cpp:94` | 1 | comment | `std::cout << out[i];` |
| `tests/binary/build_tree.cpp:96` | 1 | comment | `std::cout << out[i];` |
| `tests/binary/build_tree.cpp:138` | 1 | comment | `std::cout << out[i];` |
| `tests/binary/build_tree.cpp:140` | 1 | comment | `std::cout << out[i];` |
| `tests/binary/container.cpp:1042` | 1 | comment | `std ::cout << k_lower << " " << k_higher << "\n";` |
| `tests/binary/container.cpp:1278` | 1 | comment | `using S = decltype(c);` |
| `tests/binary/container.cpp:1280` | 4 | comment | `const auto *c1 = (const uintptr_t *)&b;` |
| `tests/binary/container.cpp:1285` | 6 | comment | `const auto *d = (uint64_t *)b.ptr();` |
| `tests/binary/container.cpp:1292` | 4 | comment | `std::cout << std::hex;` |
| `tests/binary/container_avx.cpp:38` | 1 | comment | `std::cout << j << "\n";` |

## tests/compression (1)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/compression/lz77.cpp:18` | 2 | comment | `const unsigned char text[] = "this aaaaaaaaaaaaaaaaaa is an 283682 ithis is an 283682 aaaaaaaaaaaaaaa input text, alsdkf` |

## tests/container (47)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/container/btree_set.cpp:88` | 2 | comment | `if (realDist(randGen) < 0.003)` |
| `tests/container/btree_set.cpp:124` | 2 | comment | `if (realDist(randGen) < 0.001)` |
| `tests/container/btree_set.cpp:162` | 2 | comment | `if (realDist(randGen) < 1.0 / std::min(std::max(set1.size(), static_cast<size_t>(1)), static_cast<size_t>(1000)))` |
| `tests/container/constNonBlockingLinkedList.cpp:105` | 4 | comment | `for (auto const &i: ll) {` |
| `tests/container/critbit.cpp:95` | 1 | comment | `EXPECT_NE(r1, nullptr);` |
| `tests/container/fq_vector.cpp:20` | 1 | comment | `using K5 = kAryPackedContainer_T<uint64_t, 100, 5>;` |
| `tests/container/hashmap/hopscotch.cpp:20` | 2 | comment | `it.value() += 2;` |
| `tests/container/hashmap/hopscotch.cpp:24` | 1 | comment | `{d, 6} {a, 3} {e, 7} {c, 5}` |
| `tests/container/hashmap/hopscotch.cpp:54` | 1 | comment | `{a, 1} {b, 2}` |
| `tests/container/hashmap/simple.cpp:40` | 1 | comment | `std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";` |
| `tests/container/hashmap/simple.cpp:66` | 1 | comment | `std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";` |
| `tests/container/hashmap/simple2.cpp:33` | 1 | comment | `std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";` |
| `tests/container/imap.cpp:68` | 4 | comment | `static void imap_primitives_test() {` |
| `tests/container/imap.cpp:73` | 5 | comment | `memset(vec32, 0, sizeof vec32);` |
| `tests/container/imap.cpp:79` | 12 | comment | `memset(vec32, 0, sizeof vec32);` |
| `tests/container/imap.cpp:134` | 2 | comment | `tree = 0;` |
| `tests/container/imap.cpp:137` | 28 | comment | `slot = tree.assign(0xA0000056);` |
| `tests/container/imap.cpp:166` | 2 | comment | `tree = 0;` |
| `tests/container/imap.cpp:169` | 37 | comment | `slot = tree.assign(0xA0000056);` |
| `tests/container/imap.cpp:207` | 2 | comment | `tree = 0;` |
| `tests/container/imap.cpp:210` | 46 | comment | `slot = tree.assign(0xA0000056);` |
| `tests/container/imap.cpp:659` | 1 | comment | `uint64_t mark = tree->vec32[imap__tree_mark__];` |
| `tests/container/imap.cpp:1005` | 2 | comment | `EXPECT_NE(0, pair.x);` |
| `tests/container/packed_f3.cpp:56` | 1 | comment | `t.print();` |
| `tests/container/packed_f3.cpp:137` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:142` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:147` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:157` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:162` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:167` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:193` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:199` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:205` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:211` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:217` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:223` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:239` | 2 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:246` | 2 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:253` | 2 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:260` | 2 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:267` | 2 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:274` | 1 | comment | `Row::print(t);` |
| `tests/container/packed_f3.cpp:306` | 1 | comment | `a random limb of 32 numbers in {0, 1, 2}` |
| `tests/container/rb_tree.cpp:6` | 24 | #if 0 | `#include "container/rb_tree.h"` |
| `tests/container/signal_tree.cpp:200` | 2 | comment | `EXPECT_EQ(test_single_threaded<100000000>(), 0);` |
| `tests/container/test_fqpackedvector.h:259` | 2 | comment | `EXPECT_LT(v1, size2);` |
| `tests/container/test_fqpackedvector.h:262` | 1 | comment | `const size_t v3 = b1.hash(l, h);` |

## tests/crypto (1)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/crypto/sha1.cpp:18` | 1 | comment | `using namespace cryptanalysislib;` |

## tests/cuda (16)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/cuda/CMakeLists.txt:6` | 1 | comment | `target_compile_options(${file} PUBLIC ${COMPILE_TEST_FLAGS})` |
| `tests/cuda/CMakeLists.txt:8` | 1 | comment | `add_dependencies(cuda_${file} m4ri_project fplll_project)` |
| `tests/cuda/CMakeLists.txt:17` | 1 | comment | `target_compile_options(${file} PUBLIC ${COMPILE_TEST_FLAGS})` |
| `tests/cuda/CMakeLists.txt:19` | 1 | comment | `add_dependencies(cuda_${file} m4ri_project fplll_project)` |
| `tests/cuda/cuda.cpp:41` | 5 | comment | `dim3 dimGrid(2, 2);` |
| `tests/cuda/cuda.cpp:61` | 1 | comment | `join2lists_cuda<BinaryElement><<<1, 10>>>(dL3, (const BinaryElement *)dL2, (const BinaryElement *)dL3, lta[0], lta[1]);` |
| `tests/cuda/gaus.cu:11` | 4 | comment | `using DecodingValue = Value_T<BinaryContainer<G_k + G_l>>;` |
| `tests/cuda/gaus.cu:53` | 1 | comment | `const rci_t n  = L->nrows;` |
| `tests/cuda/gaus.cu:77` | 1 | comment | `int permuted_until = integrated_gauss_jordan_cuda(rows, L_new->width * 2, L_new->nrows);` |
| `tests/cuda/gaus.cu:98` | 2 | comment | `if(permuted_until != -1)` |
| `tests/cuda/gaus.cu:101` | 1 | comment | `__M4RI_DD_MZD(L_inv);` |
| `tests/cuda/gaus.cu:105` | 13 | comment | `for(int i = 0; i < 64; i++)` |
| `tests/cuda/gaus.cu:152` | 1 | comment | `const rci_t n  = L->nrows;` |
| `tests/cuda/gaus.cu:170` | 2 | comment | `mzd_t* L_inv = gauss_jordan_cuda(L, n);` |
| `tests/cuda/gaus.cu:176` | 1 | comment | `gauss_jordan_cuda_integrated<<<dim3(1),dim3(16,16), 2*size_array>>>(rows, L_new->width * 2, L_new->nrows, 1, n, 0, empty` |
| `tests/cuda/gaus.cu:178` | 2 | comment | `uint32_t permuted_until;` |

## tests/hash (1)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/hash/crc32.cpp:34` | 1 | comment | `std::cout << i << " " << t1 << " " << t2 << std::endl;` |

## tests/list (19)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/list/binary.cpp:73` | 1 | comment | `std::cout << "Not correct weight: " << l[i] << " pos:" << i << "\n";` |
| `tests/list/binary.cpp:109` | 2 | comment | `std::cout << l;` |
| `tests/list/enumeration/binary.cpp:54` | 2 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:76` | 2 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:93` | 1 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:117` | 1 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:123` | 2 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:145` | 1 | comment | `print_binary(x, nn, "");` |
| `tests/list/enumeration/binary.cpp:203` | 1 | comment | `std ::cout << epos1 << ":" << epos2 << " " << pos1 << ":" << pos2 << std::endl;` |
| `tests/list/enumeration/fq.cpp:48` | 1 | comment | `enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);` |
| `tests/list/enumeration/fq.cpp:148` | 1 | comment | `enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 2);` |
| `tests/list/enumeration/fq.cpp:171` | 1 | comment | `enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);` |
| `tests/list/enumeration/fq.cpp:213` | 1 | comment | `enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 2);` |
| `tests/list/enumeration/fq.cpp:240` | 1 | comment | `enumerator.template run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(L1, L2, L3, L4);` |
| `tests/list/enumeration/fq.cpp:343` | 1 | comment | `EXPECT_EQ(L1.data_value(i).popcnt(0, split), mitm_w);` |
| `tests/list/enumeration/fq.cpp:348` | 1 | comment | `EXPECT_EQ(L1.data_value(i).popcnt(split, n), noreps_w);` |
| `tests/list/enumeration/fq.cpp:387` | 1 | comment | `EXPECT_EQ(L1.data_value(i).popcnt(0, split), mitm_w);` |
| `tests/list/enumeration/fq.cpp:392` | 1 | comment | `EXPECT_EQ(L1.data_value(i).popcnt(split, n), noreps_w);` |
| `tests/list/parallel.cpp:57` | 1 | comment | `NOT valid anymore EXPECT_EQ(L.is_correct(m), true);` |

## tests/loop_fusion (2)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/loop_fusion/constexpr.cpp:135` | 2 | comment | `typedef decltype(merged_3)::something_made_up X;` |
| `tests/loop_fusion/runtime.cpp:115` | 3 | comment | `(u1 \| u2).run();` |

## tests/math (2)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/math/math.cpp:65` | 3 | comment | `EXPECT_EQ(std::isnan(HH(1.1)), true);` |
| `tests/math/math.cpp:70` | 1 | comment | `EXPECT_DOUBLE_EQ(HH(0.5), 1.0);` |

## tests/matrix (10)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/matrix/fq.cpp:348` | 2 | comment | `std::cout << rank << std::endl;` |
| `tests/matrix/fq.cpp:427` | 1 | comment | `using MT = FqMatrix<T, __ncols, __nrows, q, true>;` |
| `tests/matrix/fq.cpp:462` | 1 | comment | `using MT = FqMatrix<T, __ncols, __nrows, q, true>;` |
| `tests/matrix/fq.cpp:612` | 2 | comment | `m.print();` |
| `tests/matrix/fq.cpp:626` | 1 | comment | `random_seed(time(0));` |
| `tests/matrix/simple.cpp:292` | 2 | comment | `std::cout << rank << std::endl;` |
| `tests/matrix/simple.cpp:330` | 1 | comment | `using MT = FqMatrix<T, __ncols, __nrows, q, true>;` |
| `tests/matrix/simple.cpp:364` | 1 | comment | `using MT = FqMatrix<T, __ncols, __nrows, q, true>;` |
| `tests/matrix/simple.cpp:431` | 2 | comment | `m.print();` |
| `tests/matrix/simple.cpp:445` | 1 | comment | `random_seed(time(0));` |

## tests/nn (9)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/nn/cuda.cu:60` | 1 | comment | `return bench();` |
| `tests/nn/cuda.cu:70` | 2 | comment | `const uint32_t threads = 1024;` |
| `tests/nn/cuda.cu:86` | 7 | comment | `for (uint32_t i = 0; i < N; i++) {` |
| `tests/nn/parameters.cpp:72` | 2 | comment | `constexpr static NN_Config config2{128, 4, 10, 32, LS, 14, 11, 0, 1024};` |
| `tests/nn/parameters.cpp:94` | 3 | comment | `constexpr size_t LS = 1u << 18u;` |
| `tests/nn/parameters.cpp:99` | 3 | comment | `constexpr static NN_Config config{256, 4, 100, 64, LS, 22, 20, 0, 5000};` |
| `tests/nn/parameters.cpp:103` | 9 | comment | `constexpr static NN_Config config{256, 4, 100, 64, LS, 25, 14, 0, 512};  // 44s` |
| `tests/nn/parameters.cpp:113` | 4 | comment | `constexpr static NN_Config config{256, 4, 5731, 64, LS, 22, 14, 0, 1000};` |
| `tests/nn/parameters.cpp:163` | 1 | comment | `constexpr static NN_Config config2{86, 3, 50, 32, LS, 11, 6, 0, 1024};` |

## tests/simd (4)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/simd/generic.cpp:32` | 1 | comment | `break;` |
| `tests/simd/simd.cpp:223` | 1 | comment | `uint32_t d2[32] = {0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0};` |
| `tests/simd/simd512.cpp:80` | 1 | comment | `break;` |
| `tests/simd/test_simd.h:8` | 1 | comment | `break;` |

## tests/sort (4)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/sort/hunter.cpp:150` | 1 | comment | `nmods += mtRand() % MaxMutations;` |
| `tests/sort/hunter.cpp:189` | 1 | comment | `if ((config.EscapeRate > 0) && ((mtRand() % EscapeRate) == 0)) {` |
| `tests/sort/sorting_algorithms.cpp:38` | 2 | comment | `free(array);` |
| `tests/sort/sorting_network.cpp:351` | 1 | comment | `const uint8_t datas1[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};` |

## tests/thread (13)

| location | lines | kind | first line |
|---|---|---|---|
| `tests/thread/heartbead_scheduler.cpp:6` | 1 | comment | `using namespace cryptanalysislib;` |
| `tests/thread/heartbead_scheduler.cpp:10` | 3 | comment | `HeartbeatScheduler s;` |
| `tests/thread/scheduler.cpp:195` | 1 | comment | `std::atomic_uint_fast64_t count(0);` |
| `tests/thread/scheduler.cpp:197` | 5 | comment | `auto throw_no_return = []() { throw std::logic_error("Error occurred."); };` |
| `tests/thread/scheduler.cpp:203` | 2 | comment | `{` |
| `tests/thread/scheduler.cpp:206` | 5 | comment | `auto throw_future = pool.submit(throw_task, 1);` |
| `tests/thread/scheduler.cpp:212` | 4 | comment | `pool.submit_detach(throw_no_return);` |
| `tests/thread/scheduler.cpp:217` | 2 | comment | `EXPECT_EQ(count.load(), 1);` |
| `tests/thread/scheduler.cpp:281` | 12 | comment | `SUBCASE("with tasks") { total_tasks = 30; }` |
| `tests/thread/scheduler.cpp:294` | 2 | comment | `EXPECT_EQ(counter.load(), total_tasks);` |
| `tests/thread/scheduler.cpp:510` | 1 | comment | `std::cout << cnt_wrp.counter << std::endl;` |
| `tests/thread/scheduler.cpp:541` | 1 | comment | `std::cout << cnt_wrp.counter << std::endl;` |
| `tests/thread/scheduler.cpp:560` | 1 | comment | `std::cout << cnt_wrp.counter << std::endl;` |
