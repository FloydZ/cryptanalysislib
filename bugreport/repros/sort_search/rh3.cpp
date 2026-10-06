#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "sort/robinhoodsort.h"
int main() {
	std::vector<uint64_t> v = {0x100000001ull, 0x1ull, 0x100000001ull, 0x1ull, 0x100000001ull, 0x1ull, 0x100000001ull, 0x1ull};
	auto r = v; std::sort(r.begin(), r.end());
	rhmergesort<uint64_t>(v.data(), v.size());
	for (auto x : v) printf("%#llx ", (unsigned long long)x); printf(" <- got\n");
	for (auto x : r) printf("%#llx ", (unsigned long long)x); printf(" <- expected\n");
}
