#include "container/binary_packed_vector.h"
#include <cstdio>
using B = BinaryVector<256>;
int main() {
  setvbuf(stdout,NULL,_IONBF,0);
  { B v; v.zero(); v.set_bit(70); printf("set_bit(70): bit70=%d (exp 1), popcnt=%u\n", (int)v.get_bit_shifted(70), v.popcnt()); }
  { B v; v.zero(); for (uint32_t i=0;i<256;i++) v.write_bit(i,1); B w = v;
    B o; o.zero(); B::slr(o, v, 3);
    printf("slr(all-ones,3): popcnt=%u (exp 253)\n", o.popcnt()); }
  { B v; v.zero(); v.write_bit(150, 1);
    printf("is_zero(0,192) with bit150 set: %d (exp 0)\n", (int)v.is_zero(0,192));
    printf("is_zero<0,192>() with bit150 set: %d (exp 0)\n", (int)v.is_zero<0,192>()); }
  { // template add over 4 limbs: middle limbs use bit operator[]
    B a,b,c; a.zero(); b.zero(); c.zero();
    for (uint32_t i=0;i<256;i++){ a.write_bit(i, (i*7+3)%5==0); b.write_bit(i, (i*3+1)%4==0);} 
    B::add<0,256>(c,a,b); int bad=0, first=-1;
    for (uint32_t i=0;i<256;i++){ bool e = a.get_bit_shifted(i)^b.get_bit_shifted(i); if (c.get_bit_shifted(i)!=e){bad++; if(first<0) first=i;}}
    printf("add<0,256>: %d wrong bits, first=%d\n", bad, first);
    bool r = B::add<0,256,200>(c,a,b);
    uint32_t w = c.popcnt();
    printf("add<0,256,norm=200>: returned %d, weight=%u (runtime add with norm returns weight>=norm=%d)\n", (int)r, w, (int)(w>=200));
    B d; d.zero(); uint32_t ww = B::add_weight(d, a, b, 0, 256);
    printf("add_weight(v3,v1,v2,0,256): returned %u, but v3.popcnt()=%u (v3 passed by value)\n", ww, d.popcnt());
  }
  { B v; v.zero(); v.random(0, 256); printf("random(0,256): limb popcnts %u %u %u %u (limbs 1..3 stay 0)\n",
      v.popcnt(0,64), v.popcnt(64,128), v.popcnt(128,192), v.popcnt(192,256)); }
}
