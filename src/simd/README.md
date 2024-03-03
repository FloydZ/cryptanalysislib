The `cryptanalysislib` provides a very easy to use and strong Single-Instruction
Multiple-Data abstraction layer. This layer is currently covering all `amd64`,
`arm-neon`, `riscv-v` ISAs.

Usage:
======

The following pictures the main concepts:
```c++
#include <cryptanalysislib/simd/simd.h>
using S = TxN_t<uint32_t, 8>;

int main() {
    S t1 = S::set1(0), 
      t2 = S::setr(0,1,2,3,4,5,6,7);

    const auto t3 =  t1 + t2;
    return t3[4];
}
```

`TxN_t` is a class that wraps `N` times the type `T` in a single container. 
Currently only `unsinged`/`signed` with `8,16,32,64` bits, integers are supported
This container provides all operations like:
- Addition, Subtraction, Multiplication, Division
- Scatter/Table lookups, Permutations
- Memory stores/loads

Additionally for the 3 main architectures `x86`, `arm` and `neon` and their 
vector extensions the following specialisations are provides which wrap:
- `x86`: `__m128i`, `__m256i` and `__m512i`
- `arm-neon`: `uint8x16`, ...

- `T=uint8_t`:  `uint8x16_t`, `uint8x32_t`
- `T=uint16_t`: `uint16x8_t`, `uint16x16_t`
- `T=uint32_t`: `uint32x4_t`, `uint32x8_t`
- `T=uint64_t`: `uint64x2_t`, `uint64x4_t`
 

 TODO:
 =====

 - riscv backeend
 - uint16x8, uint8x16_t in neon, avx
 - cvtepu8 in neon und noch mehr zero extend