#include "container/fq_packed_vector.h"
#include <cstdio>
int main(){ using V = FqPackedVector<64, 3, uint64_t>; V v; for(uint32_t i=0;i<64;i++) v.set(1,i); v.neg<0,64>(); 
  int bad=0; for(uint32_t i=0;i<64;i++) bad += v.get(i)!=2; printf("neg<0,64> n=64: %d wrong\n", bad); }
