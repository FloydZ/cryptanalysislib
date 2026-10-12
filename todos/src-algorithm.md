# TODOs: src-algorithm

90 entries. Format: `- [ ] file:line [MARKER] text`.

## src/algorithm/argmin.h

- [ ] `src/algorithm/argmin.h:162` **TODO** select correct one

## src/algorithm/bits/pdep.h

- [ ] `src/algorithm/bits/pdep.h:10` **TODO** tests and benches

## src/algorithm/bits/pext.h

- [ ] `src/algorithm/bits/pext.h:10` **TODO** tests and benches
- [ ] `src/algorithm/bits/pext.h:113` **TODO** `unsigned int pext32_emu(unsigned int v, unsigned int m) {`

## src/algorithm/count.h

- [ ] `src/algorithm/count.h:110` **TODO** replace with SIMDSelector

## src/algorithm/exclusive_scan.h

- [ ] `src/algorithm/exclusive_scan.h:17` **TODO** add SIMD implementations

## src/algorithm/fill.h

- [ ] `src/algorithm/fill.h:21` **TODO** implement SIMD algorithms

## src/algorithm/find.h

- [ ] `src/algorithm/find.h:13` **TODO** docs for find_first_of and adjacent_find
- [ ] `src/algorithm/find.h:14` **TODO** parallel versions of `find_first_of` and `adjacent_find`
- [ ] `src/algorithm/find.h:18` **TODO** somehow this yields an anonymous unitiialized field element
- [ ] `src/algorithm/find.h:395` **TODO** doc — `template<class InputIt,`
- [ ] `src/algorithm/find.h:409` **TODO** doc — `template<class InputIt,`
- [ ] `src/algorithm/find.h:425` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/find.h:442` **TODO** doc — `template<class ForwardIt,`

## src/algorithm/for_each.h

- [ ] `src/algorithm/for_each.h:7` **TODO** parallel version of for_each_n
- [ ] `src/algorithm/for_each.h:8` **TODO** parallel version of for_each_chunk

## src/algorithm/generate.h

- [ ] `src/algorithm/generate.h:3` **TODO** tests — NOTE: parallel version of generator makes no sense.
- [ ] `src/algorithm/generate.h:6` **TODO** doc — \tparam ForwardIt

## src/algorithm/histogram.h

- [ ] `src/algorithm/histogram.h:13` **TODO** add cryptanalysislib namespace
- [ ] `src/algorithm/histogram.h:14` **TODO** multiple parallel histograms, result in a speedup?

## src/algorithm/int2weight.h

- [ ] `src/algorithm/int2weight.h:11` **TODO** add namespace cryptanalysislib

## src/algorithm/matrix.h

- [ ] `src/algorithm/matrix.h:15` **TODO** `inline double A[MAXN], B[MAXN], C[MAXN], D[MAXN], X[MAXN];`

## src/algorithm/max.h

- [ ] `src/algorithm/max.h:16` **TODO** implement max_element simd version
- [ ] `src/algorithm/max.h:17` **TODO** implement max_element parallel version
- [ ] `src/algorithm/max.h:182` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/max.h:201` **TODO** doc — `template<class ForwardIt,`

## src/algorithm/merge.h

- [ ] `src/algorithm/merge.h:3` **TODO** simd version
- [ ] `src/algorithm/merge.h:4` **TODO** parallel version
- [ ] `src/algorithm/merge.h:8` **TODO** doc — `template<class InputIt1,`
- [ ] `src/algorithm/merge.h:33` **TODO** doc — `template<class InputIt1,`

## src/algorithm/min.h

- [ ] `src/algorithm/min.h:17` **TODO** parallel version of min_element
- [ ] `src/algorithm/min.h:18` **TODO** simd version of min_element
- [ ] `src/algorithm/min.h:175` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/min.h:194` **TODO** doc — `template<class ForwardIt,`

## src/algorithm/minmax.h

- [ ] `src/algorithm/minmax.h:9` **TODO** parallel version implementation
- [ ] `src/algorithm/minmax.h:15` **TODO** make iterator version of this

## src/algorithm/mismatch.h

- [ ] `src/algorithm/mismatch.h:22` **TODO** test — SIMD-optimized mismatch finding for integral arrays

## src/algorithm/mq/fes.h

- [ ] `src/algorithm/mq/fes.h:250` **FIXME** get rid of this multiplication
- [ ] `src/algorithm/mq/fes.h:287` **FIXME** consider getting rid of this. This function is internal !
- [ ] `src/algorithm/mq/fes.h:603` **TODO** to fix the issue with 10 is to greate two more kernels which only enumerate 6 or 7 variables
- [ ] `src/algorithm/mq/fes.h:629` **TODO** iterativer revolving door ansatz
- [ ] `src/algorithm/mq/fes.h:648` **TODO** gamma is not correct, need to proper understand it.

## src/algorithm/partition.h

- [ ] `src/algorithm/partition.h:10` **TODO** doc — TODO parallel versions
- [ ] `src/algorithm/partition.h:11` **TODO** parallel versions
- [ ] `src/algorithm/partition.h:12` **TODO** simd versions

## src/algorithm/pcs.h

- [ ] `src/algorithm/pcs.h:9` **TODO** README and example
- [ ] `src/algorithm/pcs.h:10` **TODO** multithreading and then pcs
- [ ] `src/algorithm/pcs.h:11` **TODO** gospers cycle finding
- [ ] `src/algorithm/pcs.h:12` **TODO** config class
- [ ] `src/algorithm/pcs.h:241` **TODO** probaly also safe the length of the walk within the distinguished point data set
- [ ] `src/algorithm/pcs.h:247` **TODO** flavour — `T v = cryptanalysislib::rng();`
- [ ] `src/algorithm/pcs.h:272` **TODO** `std::vector<std::future<bool>> futures;`
- [ ] `src/algorithm/pcs.h:282` **TODO** reconstruct or whatever

## src/algorithm/regex.h

- [ ] `src/algorithm/regex.h:33` **FIXME** - '(...)'    Group

## src/algorithm/remove.h

- [ ] `src/algorithm/remove.h:5` **TODO** SIMD optimized version
- [ ] `src/algorithm/remove.h:6` **TODO** parallel version

## src/algorithm/replace.h

- [ ] `src/algorithm/replace.h:6` **TODO** implement SIMD optimized version

## src/algorithm/reverse.h

- [ ] `src/algorithm/reverse.h:5` **TODO** simd and parallel version
- [ ] `src/algorithm/reverse.h:7` **TODO** doc — `template<class BidirIt>`
- [ ] `src/algorithm/reverse.h:31` **TODO** doc — `template<class BidirIt,`

## src/algorithm/rotate.h

- [ ] `src/algorithm/rotate.h:182` **TODO** tests and benchmarks
- [ ] `src/algorithm/rotate.h:610` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/rotate.h:638` **TODO** doc — `template<class ForwardIt,`

## src/algorithm/rsa.h

- [ ] `src/algorithm/rsa.h:12` **TODO** generic Fq numbers

## src/algorithm/sat/bruteforce.h

- [ ] `src/algorithm/sat/bruteforce.h:8` **TODO** logging of solutions
- [ ] `src/algorithm/sat/bruteforce.h:9` **TODO** move to ./src/combination/grey.h
- [ ] `src/algorithm/sat/bruteforce.h:28` **TODO** replace with cryptanalysislib::ctz
- [ ] `src/algorithm/sat/bruteforce.h:34` **TODO** doc — `template <typename S,`
- [ ] `src/algorithm/sat/bruteforce.h:50` **TODO** write python script which unrolls everything

## src/algorithm/sat/cdcl.h

- [ ] `src/algorithm/sat/cdcl.h:31` **TODO** class: with allocator
- [ ] `src/algorithm/sat/cdcl.h:32` **TODO** add PR https://github.com/marijnheule/microsat/pulls
- [ ] `src/algorithm/sat/cdcl.h:33` **TODO** class for literal
- [ ] `src/algorithm/sat/cdcl.h:291` **TODO** calloc (aligned)

## src/algorithm/search.h

- [ ] `src/algorithm/search.h:43` **TODO** optimize the code for the case n < 4, 8, 16

## src/algorithm/set.h

- [ ] `src/algorithm/set.h:7` **TODO** parallel versions
- [ ] `src/algorithm/set.h:11` **TODO** tests — SIMD-optimized set_difference for unsigned integer types
- [ ] `src/algorithm/set.h:112` **TODO** tests — SIMD-optimized includes for unsigned integer types
- [ ] `src/algorithm/set.h:185` **TODO** definelty not correct

## src/algorithm/set_intersection.h

- [ ] `src/algorithm/set_intersection.h:11` **TODO** simd version
- [ ] `src/algorithm/set_intersection.h:12` **TODO** parrallel version
- [ ] `src/algorithm/set_intersection.h:102` **TODO** simsimd_intersect_u16_serial(a, b, a_end - a, b_end - b, results);

## src/algorithm/shift.h

- [ ] `src/algorithm/shift.h:7` **TODO** replace ExecPolicy with my parallel policy

## src/algorithm/sort.h

- [ ] `src/algorithm/sort.h:5` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/sort.h:13` **TODO** doc — `template<class ForwardIt,`
- [ ] `src/algorithm/sort.h:33` **TODO** doc — `template<class ForwardIt>`
- [ ] `src/algorithm/sort.h:40` **TODO** doc — `template<class ForwardIt, class Compare>`

## src/algorithm/subsetsum.h

- [ ] `src/algorithm/subsetsum.h:52` **TODO** image and explanation of config parameters
- [ ] `src/algorithm/subsetsum.h:539` **TODO** `constexpr static size_t L1_bucketsize = 100; // factor * (Enumerator::max_list_size >> (instance.l1)`

## src/algorithm/swap.h

- [ ] `src/algorithm/swap.h:46` **TODO** all other functions

## src/algorithm/zip.h

- [ ] `src/algorithm/zip.h:8` **TODO** normal iterator version
