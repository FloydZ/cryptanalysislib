# TODOs: src-matrix

13 entries. Format: `- [ ] file:line [MARKER] text`.

## src/matrix/binary_matrix.h

- [ ] `src/matrix/binary_matrix.h:51` **TODO** write a SIMD abstraction layer (static global function?) which gives you the max aligment the computer supports.
- [ ] `src/matrix/binary_matrix.h:81` **TODO** rename — \param M input matrix

## src/matrix/matrix.h

- [ ] `src/matrix/matrix.h:158` **TODO** generalize over field
- [ ] `src/matrix/matrix.h:190` **TODO** doc — \tparam nrows_prime
- [ ] `src/matrix/matrix.h:324` **TODO** replace with `RowType` operation
- [ ] `src/matrix/matrix.h:419` **TODO** replace this code with a function call to the function just below
- [ ] `src/matrix/matrix.h:721` **TODO** add a config class to the matrix class, which contains a config
- [ ] `src/matrix/matrix.h:991` **TODO** currently only 1 and q-1 are pivot rows
- [ ] `src/matrix/matrix.h:1151` **TODO** doc — \tparam Tprime
- [ ] `src/matrix/matrix.h:1217` **TODO** doc — \param C
- [ ] `src/matrix/matrix.h:1313` **TODO** optimization if Value is binary
- [ ] `src/matrix/matrix.h:1314` **TODO** more tests — TODO simd
- [ ] `src/matrix/matrix.h:1315` **TODO** simd — `if constexpr ((OUT_COLS == COLS) && (IN_COLS == ROWS)) {`
