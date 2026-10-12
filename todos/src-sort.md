# TODOs: src-sort

31 entries. Format: `- [ ] file:line [MARKER] text`.

## src/sort/bit_sort.h

- [ ] `src/sort/bit_sort.h:28` **TODO** move somewhere useful
- [ ] `src/sort/bit_sort.h:103` **TODO** use SIMD wrapper

## src/sort/counting_sort.h

- [ ] `src/sort/counting_sort.h:17` **TODO** doc — \param arr

## src/sort/heap/is_heap.h

- [ ] `src/sort/heap/is_heap.h:8` **TODO** - tests and benchmarks for all functions

## src/sort/heap/push_heap.h

- [ ] `src/sort/heap/push_heap.h:6` **TODO** tests and benchmarks

## src/sort/ips4o.h

- [ ] `src/sort/ips4o.h:72` **TODO** allocator + thread template
- [ ] `src/sort/ips4o.h:81` **TODO** move somewhere usefull
- [ ] `src/sort/ips4o.h:87` **TODO** (?): make into template to allow other key types.
- [ ] `src/sort/ips4o.h:104` **TODO** (?): currently AVX-512 specific to avoid Gather
- [ ] `src/sort/ips4o.h:107` **TODO** (?): adapt to base case sorter (exact match size of sorting network)
- [ ] `src/sort/ips4o.h:216` **TODO** template<class Allocator = std::allocator<int>>
- [ ] `src/sort/ips4o.h:247` **TODO** (?): to allow SIMD writes, we should add >=1 vector of padding to the
- [ ] `src/sort/ips4o.h:296` **TODO** replace with allocator
- [ ] `src/sort/ips4o.h:353` **TODO** replace with my prng class
- [ ] `src/sort/ips4o.h:506` **TODO** (?): for verification, we should narrow the scope/visibility of these.
- [ ] `src/sort/ips4o.h:574` **TODO** (?): this is a bottleneck, should be replaced with SIMD writes.
- [ ] `src/sort/ips4o.h:640` **TODO** was 16 for avx 512
- [ ] `src/sort/ips4o.h:714` **TODO** tail mngt — `for (size_t i = 0; i < num; i += kUnroll) {`
- [ ] `src/sort/ips4o.h:1066` **TODO** pass as argument
- [ ] `src/sort/ips4o.h:1105` **TODO** fix this iterator nonsence

## src/sort/quicksort.h

- [ ] `src/sort/quicksort.h:41` **TODO** via config — parameter: threshold for nonrecursive algorithm

## src/sort/ska_sort.h

- [ ] `src/sort/ska_sort.h:207` **TODO** not optimized: exclusive_sum

## src/sort/sorting_network/avx2.h

- [ ] `src/sort/sorting_network/avx2.h:1153` **TODO** this seems to be a bug in the implementation. Currently only needed for `sortingnetwork_mergesort_u8x64`
- [ ] `src/sort/sorting_network/avx2.h:1330` **TODO** somewhere is a bug, for reasons I dont understand this shuffle is missing
- [ ] `src/sort/sorting_network/avx2.h:1555` **TODO** write a descending sorting func

## src/sort/sorting_network/avx512.h

- [ ] `src/sort/sorting_network/avx512.h:153` **TODO** remaining 4 — layers`), i.e. both 16 element halves were sorted descending and

## src/sort/sorting_network/hunter.h

- [ ] `src/sort/sorting_network/hunter.h:3` **TODO** improvements:  https://github.com/HJLebbink/SorterHunter (256 hannels)
- [ ] `src/sort/sorting_network/hunter.h:4` **TODO** add a function to `Pair_t` which checks if its a simd swap pattern
- [ ] `src/sort/sorting_network/hunter.h:6` **TODO** config is contantexpr
- [ ] `src/sort/sorting_network/hunter.h:8` **TODO** find until Level=91 oder so

## src/sort/timsort.h

- [ ] `src/sort/timsort.h:83` **TODO** doc — `template <typename RandomAccessIterator>`
