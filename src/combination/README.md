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
