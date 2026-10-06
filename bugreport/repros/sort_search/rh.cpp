#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "sort/robinhoodsort.h"
template<typename T> void t(std::vector<T> v, const char *what) {
	auto r = v; std::sort(r.begin(), r.end());
	fprintf(stderr, "--- %s (T=%zuB, n=%zu)\n", what, sizeof(T), v.size());
	rhmergesort<T>(v.data(), v.size());
	printf("%s: %s\n", what, v == r ? "ok" : "MISMATCH");
	if (v != r && v.size() <= 8) { for (auto x : v) printf("%llu ", (unsigned long long)x); printf("\n"); }
}
int main() {
	setvbuf(stdout, nullptr, _IONBF, 0);
	// all-equal: range r=1, r/4 = 0 < n => count path, fine. try two distinct values far apart
	t<uint32_t>({5,5,5,5}, "u32 all equal");
	t<uint8_t>(std::vector<uint8_t>(16, 255), "u8 all 255");
	t<uint8_t>({0,255}, "u8 {0,255}");
	t<uint16_t>({0,65535,0,65535}, "u16 {0,max,0,max}");
	t<uint32_t>({0,1000000,7}, "u32 {0,1e6,7}");
	t<uint32_t>({1000000,0}, "u32 {1e6,0}");
	t<uint64_t>({0x100000000ull, 1}, "u64 {2^32,1}");
	t<uint64_t>({3, 0x500000002ull, 1, 0x700000000ull}, "u64 {3,5*2^32+2,1,7*2^32}");
	t<int32_t>({-5, 2000000000, 0}, "i32 {-5,2e9,0}");
	return 0;
}
