#include <cstdint>
#include "fork.h"
#include "math/math.h"
static unsigned __int128 ref(uint64_t n, uint64_t k){ if(k>n) return 0; unsigned __int128 r=1; if(k>n-k)k=n-k; for(uint64_t i=1;i<=k;i++){ r = r*(n-k+i)/i;} return r;}
int main(){
  int bad=0;
  for (uint64_t n=0;n<=67;n++) for(uint64_t k=0;k<=n+1;k++){ auto r=ref(n,k); if (r>UINT64_MAX) continue; uint64_t v=bc(n,k); if(v!=(uint64_t)r){ if(bad<8) printf("bc(%llu,%llu)=%llu expected %llu\n",(unsigned long long)n,(unsigned long long)k,(unsigned long long)v,(unsigned long long)(uint64_t)r); bad++;}}
  printf("bad=%d\n",bad);
  printf("sum_bc(5,0)=%llu (empty sum, returns max(...,1))\n",(unsigned long long)sum_bc(5,0));
}
