// interpolation search edge cases; run one case per process (argv[1]=func, argv[2]=case) so hangs/crashes are isolated
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>
typedef unsigned long ulong;
#include "search/search.h"
using T = uint32_t;

struct Case { const char *name; std::vector<T> v; T key; };
static std::vector<Case> cases = {
	{"present middle",        {10,20,30,40,50,60,70,80}, 40},
	{"absent between",        {0,10,20}, 5},
	{"absent between2",       {10,20,30,40,50,60,70,80}, 45},
	{"key < all",             {10,20,30}, 5},
	{"key > all",             {10,20,30}, 35},
	{"n=1 present",           {7}, 7},
	{"n=1 absent",            {7}, 8},
	{"all equal present",     {5,5,5,5}, 5},
	{"all equal absent",      {5,5,5,5}, 6},
	{"dups first occurrence", {1,2,2,2,2,9}, 2},
	{"skewed present",        {1,2,3,4,1000000}, 4},
	{"present first",         {10,20,30,40}, 10},
	{"present last",          {10,20,30,40}, 40},
};

int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	int f = atoi(argv[1]); int c = atoi(argv[2]);
	auto &C = cases[c];
	auto id = [](const T &e) -> T { return e; };
	const auto &v = C.v;
	size_t lb = std::lower_bound(v.begin(), v.end(), C.key) - v.begin();
	bool present = lb < v.size() && v[lb] == C.key;
	long long got = -2;
	const char *fn = "";
	switch (f) {
		case 0: fn = "lower_bound_interpolation_3p_search"; got = lower_bound_interpolation_3p_search(v.begin(), v.end(), C.key, id) - v.begin(); break;
		case 1: fn = "lower_bound_interpolation_search2"; got = lower_bound_interpolation_search2(v.begin(), v.end(), C.key, id) - v.begin(); break;
		case 2: fn = "LowerBoundInterpolationSearch(iter)"; got = LowerBoundInterpolationSearch(v.begin(), v.end(), C.key, id) - v.begin(); break;
		case 3: fn = "LowerBoundInterpolationSearch(ptr)"; got = (long long)LowerBoundInterpolationSearch<T>(v.data(), C.key, 0, v.size(), id); break;
	}
	bool ok;
	if (present) ok = got == (long long)lb;
	else ok = (got == (long long)v.size()) || (got == -1) || (got == (long long)lb); // accept end / -1 / lower_bound position
	printf("%-38s %-24s got=%lld lb=%zu present=%d %s\n", fn, C.name, got, lb, present, ok ? "ok" : "WRONG");
	return ok ? 0 : 3;
}
