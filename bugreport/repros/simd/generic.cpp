#include "simd/simd.h"
#include <cstdio>
#define CHECK(name, got, exp) do { auto _g=(uint64_t)(got); auto _e=(uint64_t)(exp); printf("%-44s got=0x%llx exp=0x%llx %s\n", name, (unsigned long long)_g, (unsigned long long)_e, _g==_e?"OK":"BUG"); } while(0)
int main() {
  // TxN_t::move for N > 32 (scalar tail int shift)
  { using T = TxN_t<uint8_t, 40>; T a = T::set1(0); a.d[35] = 0x80;
    CHECK("TxN_t<u8,40>::move (bit35)", T::move(a), 1ull<<35); }
  // TxN_t::move: simd256_type::move() << 32 (uint32 shift)
  { using T = TxN_t<uint8_t, 64>; T a = T::set1(0); a.d[40] = 0x80;
    CHECK("TxN_t<u8,64>::move (bit40)", T::move(a), 1ull<<40); }
  // TxN_t::gather scales index twice
  { using T = TxN_t<uint32_t, 5>; uint32_t data[64]; for (int i=0;i<64;i++) data[i]=100+i;
    T idx; for (int i=0;i<5;i++) idx.d[i]=i;
    T r = T::gather(data, idx);
    CHECK("TxN_t<u32,5>::gather(data,{..,1,..})[1]", r.d[1], 101); }
  // TxN_t::andnot_
  { using T = TxN_t<uint32_t, 5>; T a = T::set1(0xF0), b = T::set1(0xFF);
    CHECK("TxN_t<u32,5>::andnot_(0xF0,0xFF)", T::andnot_(a,b).d[0], 0x0F); }
  // TxN_t::lt_/gt with uint64 tail (non-simd)
  { using T = TxN_t<uint64_t, 3>; T a = T::set1(1), b = T::set1(2);
    CHECK("TxN_t<u64,3>::lt(1,2)", T::lt(a,b), 0x7); }
  return 0;
}
