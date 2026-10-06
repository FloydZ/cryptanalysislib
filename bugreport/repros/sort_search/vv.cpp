#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "sort/vv_radixsort.h"
int main() {
	setvbuf(stdout, nullptr, _IONBF, 0);
	for (int call = 0; call < 3; call++) {
		std::vector<uint32_t> v(64); for (size_t i = 0; i < v.size(); i++) v[i] = (uint32_t)(i * 2654435761u);
		auto r = v; std::sort(r.begin(), r.end());
		printf("call %d ...\n", call);
		vv_radix_sort(v.data(), v.size());
		printf("call %d: %s\n", call, v == r ? "ok" : "MISMATCH");
	}
	// second size larger than the first: static buffer was sized for the first call only
	std::vector<uint8_t> a(4, 1); vv_radix_sort(a.data(), a.size());
	std::vector<uint8_t> b(4096); for (auto &x : b) x = rand(); vv_radix_sort(b.data(), b.size());
	printf("done\n");
}
