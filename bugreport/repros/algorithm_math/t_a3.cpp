#include "common.h"
#include "algorithm/min.h"
#include "algorithm/max.h"
#include "algorithm/minmax.h"
#include "algorithm/argmin.h"
#include "algorithm/argmax.h"
template<typename T> void run(const char* name){
  for (size_t n : SIZES) for (size_t off=0; off<3; off++) {
    if (!n) continue;
    auto v = rvec<T>(n+off, 1000); for (auto &x : v) x = (T)(x + 10);
    T emin = *std::min_element(v.begin()+off, v.end()), emax = *std::max_element(v.begin()+off, v.end());
    if (n >= 32) {
      T g = cryptanalysislib::internal::min_simd_uXX<T>(v.data()+off, n);
      CHECK(g==emin, "min_simd_uXX<%s> n=%zu off=%zu exp=%lld got=%lld", name, n, off, (long long)emin, (long long)g);
      T h = cryptanalysislib::internal::max_simd_uXX<T>(v.data()+off, n);
      CHECK(h==emax, "max_simd_uXX<%s> n=%zu off=%zu exp=%lld got=%lld", name, n, off, (long long)emax, (long long)h);
    }
  }
}
int main(){
  run<uint8_t>("u8"); run<uint16_t>("u16"); run<uint32_t>("u32"); run<uint64_t>("u64");
  // public min/max scalar path (len < 32)
  CASE("max_small", { std::vector<uint32_t> v{1,2,3,4,0}; auto m = cryptanalysislib::max(v.begin(), v.end()); printf("max({1,2,3,4,0}) = %u expected 4\n", m); });
  CASE("max_big_neg", { std::vector<uint32_t> v(64, 7); auto m = cryptanalysislib::max(v.begin(), v.end()); printf("max(64x7) = %u expected 7\n", m); });
  // argmin / argmax
  for (size_t n : SIZES) {
    if (!n) continue;
    auto v = rvec<uint32_t>(n, 1000);
    size_t e = std::min_element(v.begin(), v.end()) - v.begin();
    size_t g = cryptanalysislib::internal::argmin_simd<uint32x8_t>(v.data(), n);
    CHECK(e==g, "argmin_simd n=%zu exp=%zu got=%zu", n, e, g);
    size_t g16 = cryptanalysislib::internal::argmin_simd_bl16<uint32x8_t>(v.data(), n);
    CHECK(e==g16, "argmin_simd_bl16 n=%zu exp=%zu got=%zu", n, e, g16);
    size_t g32 = cryptanalysislib::internal::argmin_simd_bl32<uint32x8_t>(v.data(), n);
    CHECK(e==g32 || v[g32]==v[e], "argmin_simd_bl32 n=%zu exp=%zu got=%zu", n, e, g32);
    CHECK(v[g32]==v[e] , "argmin_simd_bl32 value n=%zu", n);
    size_t ea = std::max_element(v.begin(), v.end()) - v.begin();
    size_t ga = cryptanalysislib::internal::argmax_simd<uint32x8_t>(v.data(), n);
    CHECK(ea==ga, "argmax_simd n=%zu exp=%zu got=%zu", n, ea, ga);
    size_t ga16 = cryptanalysislib::internal::argmax_simd_bl16<uint32x8_t>(v.data(), n);
    CHECK(ea==ga16, "argmax_simd_bl16 n=%zu exp=%zu got=%zu", n, ea, ga16);
    size_t ga32 = cryptanalysislib::internal::argmax_simd_bl32<uint32x8_t>(v.data(), n);
    CHECK(ea==ga32, "argmax_simd_bl32 n=%zu exp=%zu got=%zu", n, ea, ga32);
  }
  // all-max values
  { std::vector<uint32_t> v(5, 0xffffffffu); v[3]=0xffffffffu; size_t g = cryptanalysislib::internal::argmin_simd_bl32<uint32x8_t>(v.data(), 5); printf("argmin_simd_bl32(5 x UINT32_MAX) = %zu expected 0\n", g); }
  { std::vector<uint32_t> v(5, 0); size_t g = cryptanalysislib::internal::argmax_simd<uint32x8_t>(v.data(), 5); printf("argmax_simd(5 x 0) = %zu expected 0\n", g); }
  printf("fails=%d\n", fails);
}
