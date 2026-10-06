#include "container/binary_packed_vector.h"
#include <cstdio>
int main(int argc, char**argv) {
  setvbuf(stdout,NULL,_IONBF,0);
  int t = argc>1 ? atoi(argv[1]) : 0;
  if (t==0) { using B = BinaryVector<256>; B v; v.zero(); v.random(0, 200);
    printf("random(0,200): limb popcnts %u %u %u %u (exp all ~32 except last ~4)\n",
      v.popcnt(0,64), v.popcnt(64,128), v.popcnt(128,192), v.popcnt(192,256)); }
  if (t==1) { using B = BinaryVector<64>; B v; v.one(); v.zero(0, 64); printf("zero(0,64) n=64 ok, popcnt=%u\n", v.popcnt()); }
  if (t==2) { using B = BinaryVector<128>; B v; v.zero(); v.one(0, 128); printf("one(0,128) n=128 ok, popcnt=%u\n", v.popcnt()); }
  if (t==3) { using B = BinaryVector<128>; B a, c; a.one(); c.one(); B::scalar(c, a, false, 0, 100); printf("scalar(.., 0) popcnt=%u\n", c.popcnt()); }
  if (t==4) { using B = BinaryVector<128>; B v; v.zero(); printf("is_zero(0,128) n=128: %d\n", v.is_zero(0,128)); }
}
