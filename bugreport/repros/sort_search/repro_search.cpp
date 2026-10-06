#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>
typedef unsigned long ulong;   // binary.h uses `ulong`, which is undeclared on macOS/libc++
#include "search/search.h"
using T = uint32_t;
int main(int argc, char **argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);
	int w = argc > 1 ? atoi(argv[1]) : 0;
	auto id = [](const T &e) -> T { return e; };
	auto eq = [](const T &a, const T &b) { return a == b; };
	auto lt = [](const T &a, const T &b) { return a < b; };
	if (w == 0 || w == 1) {
		T one[1] = {5};
		printf("bsearch({5},1,3)=%zu (exp 1)  bsearch_geq({5},1,7)=%zu (exp 1)\n", bsearch(one, 1, (T)3), bsearch_geq(one, 1, (T)7));
		printf("standard_binary_search({5},1,3)=%zu (exp -1)  monobound_binary_search(nullptr,0,3)=%zu (exp -1)\n",
		       standard_binary_search(one, 1, (T)3), monobound_binary_search((T*)nullptr, 0, (T)3));
		std::vector<T> v1 = {5};
		printf("lower_bound_standard_binary_search({5}, 7)=%td (exp 1)\n", lower_bound_standard_binary_search(v1.begin(), v1.end(), (T)7, id) - v1.begin());
	}
	if (w == 0 || w == 2) {
		double f[10]; for (int i = 0; i < 10; i++) f[i] = i;
		printf("bsearch_approx(0..9, v=7, da=0.1)=%lu (exp 7)\n", bsearch_approx(f, 10ul, 7.0, 0.1));
		printf("bsearch_leq({1,5,6},3,v=2)=%zu (doc: first elem <= v => exp 0)\n", bsearch_leq((const T[]){1,5,6}, 3, (T)2));
	}
	if (w == 0 || w == 3) {
		std::vector<T> v = {10, 20};
		printf("lower_bound_monobound_binary_search({10,20},10)=%td (exp 0)\n", lower_bound_monobound_binary_search(v.begin(), v.end(), (T)10, id) - v.begin());
		std::vector<T> u = {10, 20, 30};
		printf("upper_bound_monobound_binary_search({10,20,30},30)=%td (exp 2)\n", upper_bound_monobound_binary_search(u.begin(), u.end(), (T)30, id) - u.begin());
		printf("upper_bound_monobound_binary_search({10,20,30},25)=%td (exp 3=end)\n", upper_bound_monobound_binary_search(u.begin(), u.end(), (T)25, id) - u.begin());
		std::vector<T> t = {1, 2, 3, 4};
		printf("tripletapped_binary_search(it)({1,2,3,4},2)=%td (exp 1)\n", tripletapped_binary_search(t.begin(), t.end(), (T)2, id) - t.begin());
	}
	if (w == 0 || w == 4) {
		std::vector<T> v = {10, 20};
		printf("upper_bound_linear_search({10,20},10,eq)=%td (exp 0)\n", upper_bound_linear_search(v.begin(), v.end(), (T)10, eq) - v.begin());
		std::vector<T> d = {1, 2, 2, 3};
		printf("lower_bound_linear_search({1,2,2,3},2,lt)=%td (std::lower_bound=1)\n", lower_bound_linear_search(d.begin(), d.end(), (T)2, lt) - d.begin());
		printf("upper_bound_linear_search({1,2,2,3},2,lt)=%td (std::upper_bound=3)\n", upper_bound_linear_search(d.begin(), d.end(), (T)2, lt) - d.begin());
	}
	if (w == 5) { // Khuong: low[len_list] reads past the end
		T *a = (T *)malloc(3 * sizeof(T)); a[0] = 1; a[1] = 2; a[2] = 3;
		printf("Khuong_bin_search({1,2,3},3)=%zu\n", Khuong_bin_search(a, 3, (T)3));
	}
	if (w == 6) {
		std::vector<T> v = {0, 10, 20};
		printf("calling interpolation_search({0,10,20}, 5) ...\n");
		auto it = cryptanalysislib::search::interpolation_search(v.begin(), v.end(), (T)5, id);
		printf("returned %td\n", it - v.begin());
	}
	if (w == 7) {
		std::vector<T> v = {10, 20, 30};
		printf("LowerBoundInterpolationSearch(iter)({10,20,30},35)=%td (exp 3=end)\n", LowerBoundInterpolationSearch(v.begin(), v.end(), (T)35, id) - v.begin());
	}
	return 0;
}
