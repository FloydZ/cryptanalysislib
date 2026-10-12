# Applied to the dev checkout (uncommitted)

Session 2026-10-11: one by one, approved by the user. `build/todo-tools/apply.sh` (git apply, or patch(1) with fuzz), `preview.sh` (patch rebased onto the checkout).

| patch | how | verification after applying (gcc 15.3 Release, build/) |
|---|---|---|
| 002-006 | earlier session | - |
| 010 | earlier session | - |
| 103 (WIP) | earlier session | UNVERIFIED |
| 011 | clean | py_compile |
| 014 | clean | py_compile |
| 015 | fuzz (onto 014) | py_compile |
| 016 | clean | py_compile |
| 017 | clean | py_compile |
| 020 | clean | py_compile |
| 021 | clean | py_compile |
| 022 | clean | -O0 before 52:1, after 52:31 56:22 |
| 023 | clean | test_algorithm_argmin 12/12 x3 seeds; dispatch 65536: 1630 ns vs 2437 ns |
| 024 | clean | test target did not compile (tzcnt/lzcnt without -mbmi/-mlzcnt) → 024b |
| 024b | clean (new: algorithm/countr.h, countl.h, tests/algorithm/count_zero.cpp) | count_zero 5/5, pdep 6/6, clang ASan+UBSan clean |
| 025.v2 | clean (025 + countr/countl, covers 026) | pext 6/6 x3 seeds, clang ASan+UBSan clean |
| 027 | clean | test_algorithm_count 12/12 |
| 028 | clean | test_algorithm_exclusive_scan 12/12 x3 |
| 030 | clean | docs only |
| 031.v2 | clean | test_algorithm_find (rebased onto 030) |
| 032 | clean | test_algorithm_find 13/13, test_algorithm_all_of |
| 038.v2 | clean (also drops stale 037 TODO) | test_algorithm_for_each |
| 039.v2 | clean (+ namespace cryptanalysislib) | test_algorithm_generate 3/3 |
| 040 | fuzz (onto 039.v2) | test_algorithm_generate 3/3 |
| 041 | clean | histogram 6/6 (+1 pre-existing DISABLED), sort_sorting_algorithms 33/33, bench builds |
| 042.v2 | clean (rebased onto 041) | histogram 6/6; BM_histogram_large gcc: 16M 4.78 → 24.5 GiB/s |
| 043 | clean | pcs 1/1, 11 subsetsum binaries build+pass (some have 0 active tests, known) |
| 045 | clean | max 28/28; gcc 65536: std 24.5 µs → u8 0.43, u32 2.5, u64 7.1 µs |
| 046.v2 | clean (3-way rebased onto 045) | max 32/32 |
| 047.v2 | clean (rebased) | docs |
| 049 | clean | merge 3/3 x3; gcc 2x1M: std 4.74 ms → 0.97 ms |
| 050.v2 | clean (hand-merged onto 049, < uses SIMD per thread) | merge 5/5 x3; 2x1M: std 5.08 ms, simd 0.97, par 0.19 ms |
| 051.v2 | clean (hand-rebased) | docs; merge 5/5 |
| 053 | clean | min 28/28 |
| 054.v2 | clean (3-way rebased onto 053) | min 32/32 x3; gcc 65536: std 24.6 µs → u8 0.43, u32 2.8, u64 6.1 µs |
| 055.v2 | clean (rebased) | docs |
| 057 | clean | minmax 3/3 x3 |
| 059 | clean | mismatch 8/8 x2 |
| 060 | clean | mq fes test |
| 065 | clean | docs |
| 066.v2 | clean (rebased + reworked: 1 buffer, cryptanalysislib::copy) | partition 3/3 x3; 1M stable_partition 2.13 → 0.63 ms |
| 068.v2 | clean (+ pcs.h TODO→NOTE) | README example compiled+run: collision found |
| 071 | 3-way onto 068 (same content) | pcs 3/3, subsetsum binaries pass |
| 076 | clean | regex test |
| 077.v2 | clean (cryptanalysislib::find/popcount, + bench) | remove 3/3 x3; 4M: 5.72 → 0.90 ms |
| 078.v2 | clean (hand-merged: forward move loop, self-move fix, SIMD per chunk, simd threshold 2^21) | remove 4/4 x3 |
| 079 | clean | replace 8/8; gcc -mavx2 4M x2: u8 15.7→0.08, u32 2.68→0.34, u64 4.09→2.20 ms |
| 080.v2 | clean (+ namespace cryptanalysislib, cryptanalysislib::iter_swap, digraph qualified) | reverse 11/11 x3, digraph 6/6; 1M par 58.5 → 28.9 µs |
| 081.v2 | clean (note: cryptanalysislib::reverse) | docs; reverse 11/11 |
| 083 | clean | rotate 3/3; 1M: std 74 µs, helix/drill 59, trinity 62, griesmills/piston 325 µs |
| 084.v2 | clean (NOTE with measured numbers) | docs |
| 087 | clean | sat_bruteforce 1/1 |
| 088.v2 | clean (+ namespace cryptanalysislib) | sat 2/2, mq_fes 1/1 |
| 089 | superseded by 088 | - |
| 090.v2 | clean (rebased) | sat 3/3 |
| 095.v2 | clean (+ parse returns NOMEM on alloc failure) | cdcl 2/2 |
| 096 | clean | search 12/12 |
| 097.v2 | clean (own binary searches, cryptanalysislib::copy) | set 1/1 x3; 2x8M union 38.3 → 10.7 ms |
| 098.v2 | clean (rebased onto 097) | set 5/5; uint8 fails without the mask fix |
| 099.v2 | clean (rebased onto 097+098) | set 9/9 x3 |
| 100 | dup of 099 | - |
| 103 + 103-fix.v2 | WIP was in tree; fixup clean | set_intersection 1/1 x3 (gcc), clang ASan+UBSan clean; fixed galloping infinite loop, UB pointer arith, lzcnt |

## Round 2 (session 2026-10-12)

| patch | how | verification after applying (gcc Release, build/) |
|---|---|---|
| 104 | clean | shift 4/4 x3 |
| 105 | clean | sort 2/2 x3 |
| 109 | clean | docs only |
| 111 | clean | swap 7/7 |
| 112.v2 | clean (user rework: TODO deleted, AVX2 kernels -> internal::zip_uXX_avx2 looping over n, new internal::zip_uXX_avx512, generic zip dispatches) | zip 4/4 (build/), scratch build/z112: avx512 4/4, nobw 4/4, avx2 3/3, scalar 2/2; mutation (wrong lane perm) fails |
| 113 | clean | docs only |
| 114 | clean | alloc cache 6/6, dev 11/11, gc 7/7 |
| 115 | clean | semaphore 2/2 x2, atomic 4/4, atomic_primitives 3/3, futex 1/1, pipe 1/1 |
| 116.v3 | clean (user: remove the CMOVE macros entirely + TEST(cmov, simple); v1/v2 fix dropped) | atomic 3/3, atomic_primitives 3/3 |
| 117 | clean (on top of 116.v3) | docs only |
| 118 | clean | atomic_primitives 4/4 |
| 119 | src: git apply put the wait/notify_one hunks at the wrong (identical) TODOs after the 118 offset -> notify_one(volatile) before its target = infinite recursion (test hung); fixed placement by hand. test: merged after 118 test (both kept) | atomic_primitives 5/5 x3 |
| 123 | clean | list_enumeration_binary 10/10 |
| 124 | chase.h offset +3 (from 123, correct place) | list_enumeration_binary 10/10 |
| 125.v2 | chase.h offset -2 (correct place); v2 = without the NOTE (user) | chase 2/2 |
| 126.v2 | clean; v2 = history NOTE replaced by the fxtbook attribution | min_change 2/2 |
| 127.v2 | clean; v2 = without the "before gotos" comment | revolving_door 2/2, mq_fes 1/1 (+1 pre-existing disabled) |
| 128 | clean (CRLF kept) | not built by cmake; verified standalone when prepared |
| 129.v2 | clean; v2 = without the tests/bench pointer NOTE | bwt 3/3, bench_compression_bwt runs |
| 130.v2 | clean; v2 = history/pointer NOTEs removed, std::memcpy -> __builtin_bit_cast | compress_float 2/2 (build/, scalar); scratch build/z130: 2/2 with -mavx512bf16 and 2/2 without |
| 131 | clean, comment-only (applied without asking) | docs checked against the code |
| 133.v2 | clean; v2 = TODO just removed (no pointer NOTE) | lzss 4/4, bench_compression_lzss builds |
| 134.v2 | clean; v2 = without the "before, unfinished copy" NOTE (also covers 135-137) | aa_tree 3/3 |
| 138.v2 | clean; v2 = TODO removed without pointer NOTE, history comment in less_r removed | array 1/1 (21 static_asserts) |
| 139 | clean | avl_tree 13/13 |
| 140.v2 | clean; v2 = one-line pointer instead of the history NOTE | header compiles (build/z140/check.cpp, included twice); found the 118 debug-build ambiguity |
| 144.v2 | clean; v2 = `_bzhi_u64` inside `if !consteval` (also covers 145) | ctest 8/8 (container_binary_packed_vector, binary_container/32/64/128/256/tree/list); scratch build/z144: bmi2, no-bmi2, debug 1/1 each |
| 146.v2 | clean; v2 = hunk headers rebased on 144, test rewritten readably | ctest 9/9 binary suites incl. new ranges; scratch build/z146: avx512 1/1, avx2 1/1 |
| 147 | offset +4 (from 144, correct place), comment-only (applied without asking) | - |
| 149 | offset +5 (144 + 147, correct place) | rotate 1/1, debug syntax check OK |
| 153 | clean (also covers 154) | bk_tree 2/2, avl_tree 13/13 |
| 155.v2 | clean; v2 = without the "was called BKTreeConfig" NOTE (also covers 157, 158) | btree_set 5/5 |
| 159.v2 | clean; v2 = without the history/pointer NOTE (also covers 160-164) | cartesian_tree 2/2; bench_b63_container_cartesian_tree: 256K insert 103 ms (std::set 37), contains 39 ms |
| 165.v2 | clean; v2 = history phrases/NOTE removed (also covers 169-171, 173, 174) | critbit 2/2 |
| 176 | clean | ctrie 9/9 |
| 182 | clean | dancing_links 6/6 |
| 183.v2 | clean, comment-only (applied without asking); v2 = "former commented out versions" phrase removed | - |
| 186 | clean | ctest container_hashmap_* 5/5 (incl. new growth_policy), debug syntax check OK; ska_flat (only user) commits via next_size_over (size >= 2, so shift <= 63) |
| 188 | clean | hopscotch 2/2 |
| 194.v2 | clean; v2 = history NOTE + test pointer removed (also covers 195) | simple_compressed 2/2 |
| 197 | clean | hopscotch 2/2 (includes ska_flat.h) |
| 199.v2 | first hunk rebased on 197, rest offset -17 (= 197) (covers 196, 200-203) | ska_flat 1/1, hopscotch 2/2, debug syntax check OK |
| 205.v2 | clean; v2 = NEON NOTE without "cannot be verified here" (covers 207-209) | imap 12/12 |
| 210.v2 | clean; v2 = history NOTE -> "lowest bit first" (covers 213) | kArytype 127/127 |
| 214.v2 | clean; v2 = history NOTE removed | kd_tree 2/2, debug syntax check OK |
| 215.v2 | clean; v2 = TODO removed without pointer comment | left_right_array 1/1 |
| 216 | offset -1 (from 215, correct place) | left_right_array 1/1, left_right_array_alloc 1/1 |
| 217 | clean | constNonBlockingLinkedList 5/5 |
| 218.v2 | src clean (v1 src part); test by hand: no counting allocator (user), plain FreeList test with cryptanalysislib::allocator after 217's test | constNonBlockingLinkedList 6/6, linkedlist bench builds |
| 217-rework | user: counting allocator test of 217 replaced by a plain ConstFreeList test with cryptanalysislib::allocator, `<atomic>`/`<memory>` includes removed | constNonBlockingLinkedList 6/6 |
| 139/216-rework | user: counting allocator tests removed: 139's block in tests/container/avl_tree.cpp (file back to HEAD), 216's tests/container/left_right_array_alloc.cpp deleted | avl_tree 12/12 |
| 220.v3 | clean; v3 = Allocator template parameter (default std::allocator = cryptanalysislib::allocator, alloc.h includes queue.h), no malloc/realloc/free, grow copies in FIFO order | queue 1/1, alloc_dev 11/11; scratch build/z220: release + ASan/UBSan 1/1, mutation fails |
| 222.v2 | clean; v2 = history NOTE removed | signal_tree 3/3 (only user of select_tree_size) |
| 225.v2 | clean; v2 = pointer NOTE removed (also covers 227) | adler32 3/3 |
| 228.v2 | clean; v2 = history NOTE removed (covers 229, 230) | cityhash 1/1 |
| 231 | clean (also covers 232) | crc32 3/3, hash simple/cityhash/adler32 build |
| 234 | clean | fnv1a 1/1 |
| 235 | clean (covers 236, 237) | ctest 9/9 (hash_simple, hashmap simple/simple2/compressed, list_enumeration_binary, subsetsum join2/4/8, binary_tree), tree/subsetsum benches build, no nodiscard warnings |
| 238 | clean | hash standard 1/1 (no other includer of hash/standard.h) |
| 239.v3 | clean; v3 = user: implementation replaced by qlibs/jmp v5.0.4 (single jmp.h, namespace cryptanalysislib::jmp, memcpy instead of the unaligned u32 store) + new AArch64 static_branch (nop <-> b imm26, __builtin___clear_cache, "Si" constraint); x86.h/arm.h/riscv.h deleted; tests/jmp built (Linux x86-64/aarch64, gcc: -fno-reorder-blocks-and-partition) (covers 240, 241, 243) | jmp 4/4 in build/ (ctest jmp_test); scratch build/zjmp: gcc -O2/-O3 + clang -O2 4/4, mutation fails; AArch64 UNVERIFIED at runtime: clang --target=aarch64 -O2 -fPIE/-fPIC compiles, disassembly + __jmp records + relocs checked by hand |
| 244 | clean (1 whitespace warning from git) | ctest list_* 10/10 incl. new listview; no other users of view()/Listview |
| 245.v2 | clean; v2 = test rebased after 123's Chase.repeated, history sentences removed | list_enumeration_binary 12/12 |
| 246.v2 | offset +18 (from 245, unique contexts); v2 = history sentences removed from the test comment | list_enumeration_fq 11/11 |
| 247.v2 | clean; v2 = history sentences removed (covers 248, 462) | list_enumeration_fq 13/13 x3, subsetsum_BCJ builds/passes |
| 249.v2 | fq.h offset +6 (247), test +51 (247), placement checked (2 chase_step lambdas in fq.h, hunk at the right one); v2 = history NOTE parts removed | list_enumeration_fq 14/14 x3, debug syntax check OK |
| 118-fix | clean: qualify `__atomic_impl::__is_valid_cmpexch_failure_order` (ADL ambiguity with std::, only without NDEBUG) | debug syntax check OK; atomic_primitives 5/5 release + 5/5 debug (build/z118) |
| 250 | SKIPPED by user ("far too complicated"); v2 prepared (history NOTEs removed), tested in build/z250 only | - |
| 252.v2 | clean; v2 = "Before, this was assert(false)" sentence removed; after apply also the "not hashable: sort_level used ..." NOTE removed by hand | list (scratch build/z252) 51/51 x3 release + debug; build/ test_list_list 51/51 |
| 253.v3 | clean; v3 = history notes removed + (user request) new key-extractor overload `cryptanalysislib::sort(first, last, extract_key)` in algorithm/sort.h (wraps ska_sort), all 6 ska_sort calls in src/ replaced, sort/sort.h includes algorithm/sort.h | scratch build/z253 parallel_full 5/5 x3 release+debug; build/ parallel_full 5/5, list 51/51, list_hashmap_api 1/1, binary_sort 1/1, algorithm_sort 2/2 |
| 254 | clean | scratch build/z254 parallel_index 3/3 x3 release + debug; build/ test_list_parallel_index 3/3 |
| 255.v3 | clean; v3 = "reproduce the old behaviour" NOTE reworded + (user request) `logging::DataLogger`, `cryptanalysislib::data_logger`, `cryptanalysislib::endl`, `LOG_DATA(x)`/`logging::named(key, x)` (json record per line) | build/ log 5/5; clang 21 -Wall -Wextra clean |
| 256.v2 | clean; v2 = NOTE sentence about the removed `used` member dropped | build/ test_math_math 13/13 (scratch z256 release + debug) |
| 257.v2 | offset -1 (256); v2 = both history NOTEs ("Before it took ...", "the old version wrapped ...") removed | build/ test_math_math 14/14 (scratch z257 release + debug) |
| 258 | clean (new tests/math/complex.cpp, cmake re-run) | build/ test_math_complex 2/2, test_math_math 14/14; scratch z258 release + debug x3 |
| 259 | clean, doc-only (applied without asking) | - |
| 260.v2 | clean; v2 = "before this was a stub" NOTE removed | build/ test_math_mod 4/4; scratch z260 avx512 + debug + avx2-only x3 |
| 261 | clean | build/ test_simd_simd 354/354, test_list_list 51/51; scratch z261 avx512/debug/avx2/generic |
| 262 | offset +5 (261) | binary m4ri output identical before/after on 60 seeded matrices (build/z262/check.cpp); build/ test_list_list 51/51, test_matrix_fq 18/18 |
| 263.v2 | clean; v2 = "before only q <= 10" NOTE + "as before" test comment removed | build/ test_matrix_fq 19/19, test_matrix_simple 18/18; scratch z263 release + debug |
| 264.v2 | src offset +22 (263); v2 = test rebased after 263's from_string test (same anchor), "NOTE: was set(...)" removed | build/ test_matrix_fq 20/20; scratch release + debug |
| 265.v2 | src offset +24; v2 = test rebased after 264's copy_sub test | build/ test_matrix_fq 21/21, test_matrix_simple 18/18 |
| 266 | offset +21 | build/ test_matrix_fq 21/21 (incl. 4 transpose tests), test_matrix_simple 18/18, test_list_list 51/51 |
| 267 | offsets +18 / test +58 | build/ test_matrix_fq 22/22; mutation (wrong pivot row in table) caught |
| 268.v2 | src offset +46; v2 = "Before, only 1 and q-1" removed, test rebased after 267's test | build/ test_matrix_fq 23/23. NOTE: new markov_gaus_q test also passes with the OLD matrix.h (does not catch the 1/q-1-only limitation) |
| 269.v2 | src +39 / test +100; v2 = history NOTE + test comment removed, `mulmod_q` helper inlined (no new function) | build/ test_matrix_fq 25/25, test_matrix_simple 18/18; mult_large_q fails on the old matrix.h |
| 271.v2 | src +40, test rebased after 269's augment test; v2 = "was (OUT_COLS && ROWS)" NOTE removed, mulmod lambda inlined | scratch z263 matrix fq 26/26 release + debug; build/ ctest matrix_fq/matrix_simple/list_list/list_enumeration_fq/list_enumeration_binary 9/9 |
| 274.v2 | clean; v2 = "moved to traits.h" / "where they were defined before" NOTEs removed | build/ 15 users built (alloc, containers, list, memory, tree), ctest 13/13 |
| 275 | clean | build/ test_memory_memset 2/2 (memcmp.simd incl. uint64x8_t) |
| 276.v2 | clean (3 space-before-tab warnings, existing indentation style); v2 = history NOTE removed, add/scalar lambdas inlined, test uses cryptanalysislib::min/popcount (covers 277) | scratch build/z276 release avx512 -Wall -Wextra + debug avx2 ASan/UBSan 1/1; build/ (cmake re-run) test_nn_limb 1/1 |
| 278.v2 | clean; v2 = "upper half was silently dropped" / "former __syncwarp() was even wrong" history NOTEs removed (covers 279, 283) | UNVERIFIED (no nvcc/GPU); nn.cuh not part of the CMake build |
| 288.v2 | clean; v2 = "... were removed" NOTE dropped | build/ test_permutation 2/2, test_matrix_fq 26/26, test_matrix_simple 18/18 |
| 289.v2 | clean (1 whitespace warning); v2 = "(before, ... were global)" NOTE reworded, existing `namespace cryptanalysislib {` line removed instead of close+reopen (covers 291, 292) | build/ 11 tests + 5 benches built (search, binary sort/tree, subsetsum, tree, list); ctest 9/9 |
| 290.v2 | clean; v2 = rebased on 289.v2 (NOTE text, -4 lines), test: cryptanalysislib::sort, linear-count reference instead of std::lower_bound | scratch build/z290 release avx512 + debug avx2 ASan/UBSan 1/1; build/ test_search_binary 21/21 |
| 293.v2 | clean; v2 = history NOTE removed, std::distance/std::to_address -> `last - first` / `&*first` (generated from patched copies in build/z293, keeps the trailing space of the removed TODO line) (covers 294, 295) | scratch z293 release avx512 + debug avx2 ASan/UBSan 5/5; build/ ctest search_linear/binary/interpolation + algorithm_find 4/4, list_list + bench search build |
| 296 | clean, doc-only (applied without asking) | - |
| 297.v2 | clean; v2 = inline NOTE shortened, existing history NOTEs ("Was mulhrs...", "was empty ...") removed in the touched div functions (avx2.h ~4393 untouched function keeps its note) (covers 298, 299, 307, 310) | scratch z297 gcc release avx512 9/9 + debug avx2 ASan/UBSan 2/2, clang 21 -fsyntax-only clean; build/ ctest simd_lane_ops/simd_simd/math_mod/algorithm_find 4/4, list_list builds |
| 300.v2 | clean; v2 = history NOTE + duplicated \param/\return removed, test rebased after 297's Div tests, reference countl_zero/countr_zero instead of __builtin_clz/ctz (covers 301) | scratch z300 gcc release avx512 10/10 + debug avx2 ASan/UBSan 3/3, clang -fsyntax-only clean; build/ test_simd_lane_ops 10/10, test_simd_simd 354/354 |
| 302.v2 | clean; v2 = history NOTE removed, test rebased after 300's, `& (uint16_t)(1u << j)` -> `& (one << j)` (gcc 15: scalar int -> vector truncation error) | scratch z302 gcc release avx512 11/11 + debug avx2 ASan/UBSan 4/4, clang -fsyntax-only clean; build/ test_simd_lane_ops 11/11, test_simd_simd 354/354 |
| 303.v2 | offset +74 (3 whitespace warnings, moved code as is); v2 = "moved to/from" NOTEs reworded; moved code verified identical except the 2 TODO lines (covers 304) | build/ ctest algorithm_transpose + simd_* 7/7, list_list builds |
| 305.v2 | offset +15; v2 = "instead of a scalar loop" / "stays a scalar loop" reworded (covers 306) | build/ test_simd_simd512 11/11, test_simd_simd builds |
| 308.v2 | clean; v2 = test rebased after 302's, test comment no longer claims "not accessed"; avx2 hunk placement checked (Xint32x8_t / Xint64x4_t) (covers 309) | scratch z308 release avx512 12/12 + debug avx2 ASan/UBSan 5/5; build/ test_simd_lane_ops 12/12, test_simd_simd512 11/11 |
| 311.v2 | clean; v2 = history NOTE (old histogram computation) removed, test rebased after 305's all_equal test (covers 312) | scratch z311 release + debug ASan/UBSan (avx512) 12/12; build/ test_simd_simd512 12/12, test_simd_simd builds |
| 313 | SKIPPED by user ("this patch is shit and not what I meant"; concept/derived-to-base wrapper machinery for the AVX-512 operators); v2 prepared in build/z313 only | - |
| 314.v2 | clean, doc-only (applied without asking); v2 = contents list corrected against the file (setr, lz/tz incl. _mm_ + gfni, prefix_xor _mm_) | names grep-checked |
| 315.v3 | clean; v3 = user: mask_type = LogTypeTemplate<N> (u8..u64, __uint128_t), static_assert(N <= 128) in move (gt/lt/eq go through it); no lambda/history NOTE; compare test rewritten for all N <= 128 (values 0,1,max-1,max), move test for all N | scratch z315 simd.cpp 354/354 release avx512 + 298/298 debug avx2 ASan/UBSan, N=200 static_assert fires; build/ 29 TxN_t-user targets built, ctest 28/28 |
| 316.v2 | clean; v2 = user: popcount_limb wrapper removed ("wtf is this shit"), the TODO asking for it deleted; only docs remain (covers 317-321) | build/ test_simd_simd 354/354, test_simd_generic 9/9 |
| 322.v2 | clean; v2 = div NOTE shortened (as 297.v2), "before, this was assert(0)" NOTE removed (covers 323-338) | UNVERIFIED (NEON, no aarch64 build); neon.h not compiled on x86 |
| 341 | SKIPPED by user (v2 prepared in build/z341: signed 128 bit bit-cast ctors/operator= as class-template members) | - |
| 342 | offset +70 (covers 343, 344) | build/ test_simd_generic 9/9 (static_assert SIMDAble<TxN_t<uint32_t,8>>), test_simd_simd builds |
| 345.v2 | clean; v2 = ctz/clz NOTE reworded (no comparison with the builtins) (covers 346-353) | build/ simd_swar/simd_simd/binary_container/algorithm_find 4/4, list_list builds |
| 354.v2 | clean; v2 = "was hard coded to 64 bytes" NOTE removed, test rebased in tests/math/math.cpp (covers 355) | build/ ctest sort_bit_sort + math_math 2/2 |
| 356.v2 | clean; v2 = history NOTE (moved/leaked using) removed | build/ 26 sort.h users built (sort/list/algorithm/subsetsum/tree tests + 6 sort benches), ctest 13/13 |
| 357.v2 | clean (new tests/sort/heap.cpp + bench/sort/heap.cpp, cmake re-run); v2 = history NOTEs removed (incl. old one in is_heap.h), scatter lambda inlined, std::push_heap/std::is_heap fallbacks -> push_heap_scalar/is_heap_rnd (covers 358) | scratch z357 release avx512 + debug avx2 ASan/UBSan 2/2; build/ test_sort_heap 2/2, bench_b63_sort_heap builds |
| 359.v2 | clean; v2 = minimal: no aligned_allocator struct / Buffers<Allocator> (allocator TODOs kept), cryptanalysislib::aligned_alloc/aligned_free, kUnroll/tail change dropped (dead in scalar branch), history NOTEs removed, test reference cryptanalysislib::sort (covers 360, 366, 367, 372, 373) | scratch z359 release avx512 + debug avx2 ASan/UBSan 1/1; build/ ctest sort_parallel/sort_sorting_algorithms/algorithm_sort 3/3, list_list + bench sort build |
| 374.v2 | clean; v2 = test rebased after 356's CountingSort test, reference cryptanalysislib::sort | build/ test_sort_sorting_algorithms 35/35, algorithm_sort + list_list build |
| 375 | clean, comment-only (applied without asking) | - |
| 376.v2 | clean; v2 = history NOTEs removed, test reference = counting loop + element compare (sort.h include -> __atomic_impl ambiguity under `using namespace std`; memory/memcmp.h include hijacks the existing unqualified memcmp(bytes) calls) (covers 377) | scratch z376 45/45; build/ test_sort_sorting_network 45/45, bench sortingnetwork builds |
| 378.v2 | clean; v2 = tests rebased after 376, counting reference + stress_u8<288>, NOTE without comparison to the replaced shuffles | scratch z378 47/47; build/ test_sort_sorting_network 47/47 |
| 379.v2 | clean, doc-only (applied without asking); v2 = NOTE describes only the current network (no "Before, ...") | - |
| 382 | clean, doc-only (applied without asking) | - |
| 384 | clean, doc-only (applied without asking) | - |
| 385.v2 | clean; v2 = "before, the policy was ignored" removed, virtual-function NOTE shortened | build/ 27 targets built, ctest algorithm_* 26/26 |
| 393 | clean, comment-only (applied without asking; claim checked: clone_flags without CLONE_SETTLS) | - |
| 409.v2 | clean; v2 = 3 history NOTEs reworded/removed (jthread macro, assert(false), raw JSON); macOS branch unverified | build/ thread_* + algorithm_for_each ctest 5/5, scheduler_performance_manager builds |
| 415 | clean, comment-only (applied without asking) | - |
| 416.v2 | clean; v2 = "former minimum_power_of_two" NOTE removed | build/ test_thread_work_contract 1/1 |
| 417.v2 | offset -3 (416); v2 = TODO + unused declaration removed without NOTE | build/ test_thread_work_contract 1/1 |
| 419.v2 | clean; v2 = 3 history NOTEs reworded (current behaviour only) | build/ 9 targets built (tree, binary_tree, subsetsum, bench tree), ctest 7/7 |
| 420 | clean, doc-only (applied without asking; checked against the ctors) | - |
| 441.v2 | offset +6, doc-only (applied without asking); v2 = "level `level`+1" corrected to `lists[level + 2]` | - |
| 438 | offset +6 | build/ tree + subsetsum targets built (callers use k_l1 = 0), ctest 3/3 |
| 458.v2 | offsets +1/+9; v2 = ctor NOTE without "Before, ..." | build/ test_container_imap 13/13 (incl. assign_val128) |
| 459.v2 | clean; v2 = "re-enabled ..." history NOTE removed (hunk count fixed: +38) | build/ test_container_packed_f3 8/8 |
| 461.v2 | clean; v2 = no byte-accessor lambdas / Get template (core takes `const C *`, 64 byte block copy); old MD5 did not compile (crypto_hash.hpp: Array undefined) | scratch z461 -Wall -Wextra + ASan/UBSan 2/2, clang syntax ok; build/ test_crypto_md5 2/2 |
| 463.v2 | clean; v2 = history NOTE removed, reference cryptanalysislib::sort | build/ test_sort_parallel 13/13 |
| 466 | clean (claims checked: fill via memset, minmax iterator overload, bench_<algo> targets) | build/ ctest algorithm_fill/minmax 2/2 |
| 339 | clean (covers 340) | build/ 15 targets built (simd, list, matrix, binary, containers, algorithm, hash, nn, tree, search), ctest 13/13 |
