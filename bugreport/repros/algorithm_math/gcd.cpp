#include <cstdint>
#include <numeric>
#include "fork.h"
#include "math/gcd.h"
using namespace cryptanalysislib;
int main(){
  uint64_t a = 1ull<<33, b = 3ull<<33;
  CASE("gcd64", printf("gcd<u64>(2^33,3*2^33)=%llu expected %llu\n",(unsigned long long)gcd<uint64_t>(a,b),(unsigned long long)std::gcd(a,b)));
  CASE("gcd64b", printf("gcd<u64>(2^40+2, 6)=%llu expected %llu\n",(unsigned long long)gcd<uint64_t>((1ull<<40)+2,6),(unsigned long long)std::gcd((1ull<<40)+2,6ull)));
  CASE("gcd64c", printf("gcd<u64>(3e9+3=3000000003, 3)=%llu expected %llu\n",(unsigned long long)gcd<uint64_t>(3000000003ull,6000000006ull),(unsigned long long)std::gcd(3000000003ull,6000000006ull)));
  CASE("gcdu32", printf("gcd<u32>(3e9, 1e9)=%u expected %u\n",gcd<uint32_t>(3000000000u,1000000000u),std::gcd(3000000000u,1000000000u)));
  CASE("gcdneg", printf("gcd<int>(-4,6)=%d expected %d\n",gcd<int>(-4,6),std::gcd(-4,6)));
  CASE("gcd_v0", printf("gcd_recursive_v0(5,0)=%d expected 5\n",internal::gcd_recursive_v0<int>(5,0)));
}
