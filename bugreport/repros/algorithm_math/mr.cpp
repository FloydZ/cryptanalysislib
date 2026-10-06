#include <cstdint>
#include <cassert>
#include "math/math.h"
#include "math/eea.h"
#include "math/miller_rabin.h"
#include "math/crt.h"
#include "math/tonelli_shanks.h"
#include "math/primitive_root.h"
using namespace cryptanalysislib;
int main(){
#ifdef T_MR
  printf("%d\n", millerRabin<uint64_t>(561));
#endif
#ifdef T_CRT
  auto r = crt<int64_t>(2,3,3,5, eea<int64_t>, eea<int64_t>);
#endif
#ifdef T_TS
  printf("%lld\n",(long long)tonelli_shanks<int64_t>(2,7));
#endif
#ifdef T_PR
  printf("%lld\n",(long long)primitive_root<int64_t>(7));
#endif
}
