#include "common.h"
#include "algorithm/count.h"
#include "algorithm/find.h"
#include "algorithm/mismatch.h"
template<typename T> void run(const char* name){
  for (size_t n : SIZES) for (size_t off=0; off<3; off++) {
    auto v = rvec<T>(n+off, 4);
    T val = 1;
    long e = std::count(v.begin()+off, v.end(), val);
    long g = cryptanalysislib::internal::count_uXX_simd<T>(v.data()+off, n, val);
    CHECK(e==g, "count_uXX_simd<%s> n=%zu off=%zu exp=%ld got=%ld", name, n, off, e, g);
    if (n) { long g2 = cryptanalysislib::count(v.begin()+off, v.end(), val); CHECK(e==g2, "count<%s> n=%zu off=%zu exp=%ld got=%ld", name, n, off, e, g2);}
    // find: place val at random position, others are not val
    for (size_t pos = 0; pos <= n; pos += (n>20? 7:1)) {
      std::vector<T> w(n+off, 2); if (pos<n) w[off+pos]=val;
      size_t gi = cryptanalysislib::internal::find_uXX_simd<T>(w.data()+off, n, val);
      CHECK(gi==pos, "find_uXX_simd<%s> n=%zu off=%zu exp=%zu got=%zu", name, n, off, pos, gi);
      // mismatch
      std::vector<T> a(n+off, 3), b(n+off, 3); if (pos<n) b[off+pos]=9;
      size_t mi = cryptanalysislib::internal::mismatch_simd_uXX<T>(a.data()+off, b.data()+off, n);
      CHECK(mi==pos, "mismatch_simd_uXX<%s> n=%zu off=%zu exp=%zu got=%zu", name, n, off, pos, mi);
    }
  }
}
int main(){
  run<uint8_t>("u8"); run<uint16_t>("u16"); run<uint32_t>("u32"); run<uint64_t>("u64");
  // find on empty vector via public api
  CASE("find_empty", { std::vector<uint32_t> e; auto it = cryptanalysislib::find(e.begin(), e.end(), 1u); printf("find(empty)==end: %d\n", it==e.end()); });
  printf("fails=%d\n", fails);
}
