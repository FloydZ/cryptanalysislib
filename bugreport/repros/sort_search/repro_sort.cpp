#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <algorithm>
#include "sort/robinhoodsort.h"
#include "sort/common.h"
#include "sort/float_radixsort.h"
#include "sort/sorting_network/common.h"
template<typename V> void pr(const char *s, const V &v) { printf("%s", s); for (auto x : v) printf(" %lld", (long long)x); printf("\n"); }
int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	int which = argc > 1 ? atoi(argv[1]) : 0;
	if (which == 0 || which == 1) { // sortingnetwork_sort_i32x8
		int32_t a[8] = {1,0,2,3,4,5,6,7};
		sortingnetwork_sort_i32x8(a); pr("i32x8 {1,0,2..7} ->", std::vector<int32_t>(a, a+8));
		uint32_t b[8] = {7,6,5,4,3,2,1,0};
		sortingnetwork_sort_u32x8(b); pr("u32x8 {7..0} ->", std::vector<uint32_t>(b, b+8));
		int x = 5, y = 3; int32_MINMAX(x, y); printf("int32_MINMAX(5,3) -> (%d,%d)\n", x, y);
	}
	if (which == 0 || which == 2) { // sort_minmax_branchless
		uint32_t a = 5, b = 3; sort_minmax_branchless(a, b); printf("sort_minmax_branchless<u32>(5,3) -> (%u,%u)\n", a, b);
	}
	if (which == 0 || which == 3) { // hoare_partition with uint64
		std::vector<uint64_t> v = {0x100000000ull, 0x300000000ull, 0x200000000ull, 5};
		auto in = v;
		size_t p = hoare_partition(v.data(), 0, v.size()-1);
		pr("hoare_partition<u64> in ", in); pr("                    out", v); printf("  p=%zu\n", p);
		std::vector<uint8_t> c(300); for (size_t i = 0; i < 300; i++) c[i] = (uint8_t)(i * 37);
		auto cin = c; size_t q = hoare_partition(c.data(), 260, 299); // pivot index 260 stored in uint8_t -> 4
		bool same = std::equal(c.begin(), c.begin()+260, cin.begin());
		printf("hoare_partition<u8> head=260: prefix [0,260) untouched? %s, p=%zu\n", same ? "yes" : "NO", q);
	}
	if (which == 0 || which == 4) { // RadixSort::Sort(uint32_t*)
		uint32_t u[3] = {0x01000000u, 0x00000002u, 0x00000001u};
		RadixSort rs; uint32_t *idx = rs.Sort(u, 3).GetIndices();
		printf("RadixSort u32 {0x01000000,2,1} indices -> %u %u %u (expected 2 1 0)\n", idx[0], idx[1], idx[2]);
	}
	if (which == 5) { // rhsort: 32 equal minima + 1 maximum => aux all-sentinel => reads aux[-1]
		std::vector<uint32_t> v(32, 0); v.push_back(0xFFFFFFFFu);
		rhmergesort<uint32_t>(v.data(), v.size());
		printf("rhsort u32 32x0 + max: %s\n", std::is_sorted(v.begin(), v.end()) ? "sorted (but see UB)" : "UNSORTED");
	}
	if (which == 6) { // rhsort uint64 range > 2^32
		std::vector<uint64_t> v = {0x100000000ull, 3, 2, 1, 0x200000000ull, 7, 9, 8};
		auto r = v; std::sort(r.begin(), r.end());
		rhmergesort<uint64_t>(v.data(), v.size());
		pr("rhsort u64 ->", v); pr("expected     ", r);
	}
	if (which == 7) { std::vector<uint32_t> v; rhsort32<uint32_t>(v.data(), 0); printf("rhsort32 n=0 returned\n"); }
	return 0;
}
