# TODOs: src-search

7 entries. Format: `- [ ] file:line [MARKER] text`.

## src/search/binary.h

- [ ] `src/search/binary.h:19` **TODO** add namespace, for all the c functions
- [ ] `src/search/binary.h:20` **TODO** add @lemire binary search in parallel in multiple different lists
- [ ] `src/search/binary.h:21` **TODO** add `constexpr` where possible
- [ ] `src/search/binary.h:22` **TODO** add concept to the c functions below to check that `Type` is arithmetic

## src/search/linear.h

- [ ] `src/search/linear.h:14` **TODO** add simd implementation (probably just steal is from `src/algorithm/find.h`)
- [ ] `src/search/linear.h:15` **TODO** add namespace
- [ ] `src/search/linear.h:16` **TODO** add dispatch function as in `src/search/binary.h`
