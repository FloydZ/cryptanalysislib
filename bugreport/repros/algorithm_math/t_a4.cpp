#include "common.h"
#include "algorithm/equal.h"
#include "algorithm/exclusive_scan.h"
#include "algorithm/inclusive_scan.h"
#include "algorithm/prefixsum.h"
int main(){
  { std::vector<int> a{1,2,3}, b{1,2,3}, c{1,9,3};
    printf("equal(a,a-copy)=%d expected 1; equal(a,c)=%d expected 0\n", (int)cryptanalysislib::equal(a.begin(), a.end(), b.begin()), (int)cryptanalysislib::equal(a.begin(), a.end(), c.begin())); }
  for (size_t n : SIZES) {
    auto v = rvec<uint32_t>(n, 100);
    std::vector<uint32_t> e(n), g(n, 0xdead);
    std::exclusive_scan(v.begin(), v.end(), e.begin(), 7u);
    auto r = cryptanalysislib::exclusive_scan(v.begin(), v.end(), g.begin(), 7u);
    CHECK(e==g && r==g.end(), "exclusive_scan n=%zu", n);
    std::inclusive_scan(v.begin(), v.end(), e.begin(), std::plus<uint32_t>());
    std::fill(g.begin(), g.end(), 0xdead);
    auto r2 = cryptanalysislib::inclusive_scan(v.begin(), v.end(), g.begin(), std::plus<uint32_t>());
    CHECK(e==g && r2==g.end(), "inclusive_scan(op) n=%zu %s", n, n>1? (printf("[g0=%u e0=%u g_last=%u e_last=%u ret_off=%ld] ", g[0], e[0], g[n-1], e[n-1], (long)(r2-g.begin())),""):"");
    std::inclusive_scan(v.begin(), v.end(), e.begin(), std::plus<uint32_t>(), 7u);
    std::fill(g.begin(), g.end(), 0xdead);
    auto r3 = cryptanalysislib::inclusive_scan(v.begin(), v.end(), g.begin(), 7u, std::plus<uint32_t>());
    CHECK(e==g && r3==g.end(), "inclusive_scan(init,op) n=%zu", n);
    std::inclusive_scan(v.begin(), v.end(), e.begin());
    auto w = v; cryptanalysislib::algorithm::prefixsum(w.data(), n);
    CHECK(e==w, "prefixsum(ptr) n=%zu", n);
  }
  // non-commutative op with init: std::inclusive_scan applies op(init, x0)
  { std::vector<int> v{3,4}; std::vector<int> e(2), g(2); auto op=[](int a,int b){return a*10+b;};
    std::inclusive_scan(v.begin(), v.end(), e.begin(), op, 1);
    cryptanalysislib::inclusive_scan(v.begin(), v.end(), g.begin(), 1, op);
    printf("inclusive_scan noncomm: got {%d,%d} expected {%d,%d}\n", g[0], g[1], e[0], e[1]); }
  printf("fails=%d\n", fails);
}
