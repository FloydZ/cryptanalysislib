#include <cstdint>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "math/math.h"
#include "math/gcd.h"
#include "math/eea.h"
using namespace cryptanalysislib;
#include "fork.h"
int main(){
  // gcd
  uint64_t a = 1ull<<33, b = 3ull<<33;
  CASE("L12", printf("gcd(2^33,3*2^33) = %llu expected %llu\n", (unsigned long long)gcd<uint64_t>(a,b), (unsigned long long)std::gcd(a,b)););
  uint64_t c = 6000000000ull, d = 4000000000ull;
  CASE("L14", printf("gcd(6e9,4e9) = %llu expected %llu\n", (unsigned long long)gcd<uint64_t>(c,d), (unsigned long long)std::gcd(c,d)););
  CASE("L15", printf("gcd<int>(-4,6) = %d expected %d\n", gcd<int>(-4,6), std::gcd(-4,6)););
  CASE("L16", printf("gcd_recursive_v0(5,0) = %d expected 5\n", internal::gcd_recursive_v0<int>(5,0)););
  CASE("L17", printf("ceil_log2(UINT64_MAX)=%llu expected 64\n", (unsigned long long)ceil_log2(UINT64_MAX)););
  CASE("L18", printf("cceil(3e9)=%lld expected 3000000000\n", (long long)math::cceil(3e9)););
  CASE("L19", printf("cceil(2.5f)=%d cceil(-2.5f)=%d\n", math::cceil(2.5f), math::cceil(-2.5f)););
  CASE("L20", printf("round(2.7)=%lld expected 3; round(-2.7)=%lld expected -3\n", (long long)math::round(2.7), (long long)math::round(-2.7)););
  CASE("L21", printf("fastmod<-7>(20)=%d expected %d\n", fastmod<-7>(20), 20 % -7););
  CASE("L22", printf("fastdiv<-7>(20)=%d expected %d\n", fastdiv<-7>(20), 20 / -7););
  CASE("L23", printf("fastmod<7>(-20)=%d expected %d\n", fastmod<7>(-20), -20 % 7););
  CASE("L24", printf("fastdiv<7>(-20)=%d expected %d\n", fastdiv<7>(-20), -20 / 7););
  CASE("L25", printf("fastdiv<1u>(5)=%u expected 5\n", fastdiv<1u>(5u)););
  CASE("L26", printf("ipow(3,-1)=%d ipow(2.0,-2)=%f\n", math::ipow(3,-1), math::ipow(2.0,-2)););
  CASE("L27", printf("log(1e6)=%f exp %f; log(0.1)=%f exp %f\n", math::log(1e6), std::log(1e6), math::log(0.1), std::log(0.1)););
  CASE("L28", printf("log2(8)=%f; log<int>(2000)=%d\n", math::log2(8.0), math::log<int>(2000)););
  CASE("L29", printf("sqrt(2)=%f sqrt<int>(10)=%d cbrt(27)=%f\n", math::sqrt(2.0), math::sqrt<int>(10), math::cbrt(27.0)););
  CASE("L30", printf("HH(0.5)=%f HH(0.11)=%f exp %f\n", math::HH(0.5), math::HH(0.11), -0.11*std::log2(0.11)-0.89*std::log2(0.89)););
  CASE("L31", printf("floor(-2.5)=%f floor(-3.0)=%f\n", math::floor(-2.5), math::floor(-3.0)););
  CASE("L32", printf("round_up_pow2(0)=%zu (5)=%zu (1<<40 +1)=%zu\n", math::round_up_to_power_of_two(0), math::round_up_to_power_of_two(5), math::round_up_to_power_of_two((1ull<<40)+1)););
  CASE("L33", printf("next_prime(4)=%zu prev_prime(4)=%zu next_prime(0)=%zu is_prime(25)=%d is_prime(49)=%d\n", next_prime(4), prev_prime(4), next_prime(0), is_prime(25), is_prime(49)););
  CASE("L34", for (uint64_t n : {1ull<<40, 18446744073709551557ull}) printf("is_prime(%llu)=%d\n", (unsigned long long)n, is_prime(n)););
  int64_t x,y; int64_t g = eea<int64_t>(x,y,240,46); printf("eea(240,46)=%lld x=%lld y=%lld check=%lld\n",(long long)g,(long long)x,(long long)y,(long long)(x*240+y*46));
  g = eea<int64_t>(x,y,0,5); printf("eea(0,5)=%lld x=%lld y=%lld\n",(long long)g,(long long)x,(long long)y);
}
