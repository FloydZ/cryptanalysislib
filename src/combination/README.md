Enumerator interface:
=====================

All enumerators in this directory implement the concept `EnumeratorAble`
(`combination/enumerator.h`):

```C
enumeration_colex<uint64_t, 10, 3> e;    // at the first element
do {
    use(e.current());                    // the current element
} while (e.next());                      // false at the last element
while (e.previous()) { ... }             // false at the first element
```

`next()`/`previous()` do not change the current element if they return
`false`. Enumerators which cannot go back (`kolakoski_seq`, `bit_necklace`,
`combination_revdoor`, `bit_comb_shifts`, `bit_sl_gray`,
`Combinations_Binary_Chase`, `Combinations_Fq_Chase`) `assert(false)` in
`previous()`. Tests: `tests/combination/enumerator.cpp`.

Chase-Sequence:
===============

Usage:
------

Generate a single element of length `n` and weight `2`:
```C 
uint64_t data[2] = {0};
Combinations_Binary_Chase<uint64_t, 100, 2, 0> cbc;

uint16_t pos1, pos2;
cbc.left_step(data, &pos1, &pos2);
```

If you want to generate the full list:

```C 
std::vector<std::pair<uint16_t, uint16_t>> list(bc(100, 2));
Combinations_Binary_Chase<uint64_t, 100, 2, 0> cbc;
cbc.changeList(list.data());
```

Note this generates only the change list.

Another function to generate such lists are: `next_chase<n, p>(&pos1, &pos2)`;
