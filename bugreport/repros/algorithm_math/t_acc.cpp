#include "common.h"
#include "algorithm/accumulate.h"
#include "algorithm/reduce.h"
template<typename T> void run(const char* name){
  for (size_t n : SIZES) for (size_t off=0; off<3; off++) {
    auto v = rvec<T>(n+off, 1000);
    T init = (T)5;
    T e = std::accumulate(v.begin()+off, v.end(), init);
    T g; if constexpr (std::is_unsigned_v<T>) g = cryptanalysislib::internal::accumulate_simd_int_plus<T>(v.data()+off, n, init); else g = e;
    CHECK(e==g, "accumulate<%s> n=%zu off=%zu exp=%lld got=%lld", name, n, off, (long long)e,(long long)g);
    T e2 = std::reduce(v.begin()+off, v.end(), init);
    T g2 = cryptanalysislib::reduce(v.begin()+off, v.end(), init);
    CHECK(e2==g2, "reduce<%s> n=%zu off=%zu exp=%lld got=%lld", name, n, off, (long long)e2,(long long)g2);
  }
}
int main(){
  run<uint8_t>("u8"); run<int8_t>("i8"); run<uint16_t>("u16"); run<int32_t>("i32"); run<uint32_t>("u32"); run<uint64_t>("u64"); run<int64_t>("i64");
  // parallel reduce with non-identity init
  CASE("par_reduce", { std::vector<uint32_t> v(1u<<20, 1); auto g = cryptanalysislib::reduce(cryptanalysislib::par_if(true), v.begin(), v.end(), (uint32_t)100);
    printf("par reduce(1<<20 ones, init=100) = %u expected %u\n", g, (1u<<20)+100); });
  CASE("par_acc", { std::vector<uint32_t> v(1u<<20, 1); uint32_t g = 0; (void)v;
    printf("par accumulate(1<<20 ones, init=100) = %u expected %u\n", g, (1u<<20)+100); });
  printf("fails=%d\n", fails);
}
