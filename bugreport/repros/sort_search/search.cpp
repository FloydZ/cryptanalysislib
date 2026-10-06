// differential tests for search/ vs std::lower_bound / std::upper_bound / find
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <map>
#include <string>
#include <functional>
typedef unsigned long ulong;
#include "search/search.h"

using T = uint32_t;
static std::mt19937_64 gen(99);
static std::map<std::string, int> failcnt;
static std::map<std::string, std::string> firstfail;

static void fail(const std::string &name, const std::vector<T> &v, T key, long long got, long long exp) {
	if (failcnt[name]++ == 0) {
		char buf[512];
		std::string s = "n=" + std::to_string(v.size()) + " data=[";
		for (size_t i = 0; i < v.size() && i < 12; i++) s += std::to_string(v[i]) + (i + 1 < v.size() ? "," : "");
		if (v.size() > 12) s += "...";
		snprintf(buf, sizeof buf, "] key=%u got=%lld expected=%lld", key, got, exp);
		firstfail[name] = s + buf;
	}
}

static const size_t sizes[] = {0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,100,1000,1024,1025};

static std::vector<T> gen_data(size_t n, int kind) {
	std::vector<T> v(n);
	for (size_t i = 0; i < n; i++) {
		switch (kind) {
			case 0: v[i] = (T)(gen() % (1u << 22)); break;       // sparse uniform
			case 1: v[i] = (T)(gen() % 8); break;                // many dups
			case 2: v[i] = 42; break;                            // all equal
			case 3: v[i] = (T)(i * 2 + 10); break;               // arithmetic, gaps
			case 4: v[i] = (gen() & 1) ? std::numeric_limits<T>::max() : 1; break;
		}
	}
	std::sort(v.begin(), v.end());
	return v;
}

int main() {
	setvbuf(stdout, nullptr, _IONBF, 0);
	auto id = [](const T &e) -> T { return e; };
	auto lt = [](const T &a, const T &b) -> bool { return a < b; };
	auto eq = [](const T &a, const T &b) -> bool { return a == b; };

	for (size_t n : sizes) if (n >= (getenv("MINN") ? (size_t)atoi(getenv("MINN")) : 0)) for (int kind = 0; kind < 5; kind++) for (int rep = 0; rep < 4; rep++) {
		auto v = gen_data(n, kind);
		std::vector<T> keys;
		for (size_t i = 0; i < n; i++) keys.push_back(v[i]);           // present keys
		keys.push_back(0); keys.push_back(std::numeric_limits<T>::max()); // smaller / larger than all
		if (n) { keys.push_back(v[0] ? v[0] - 1 : 0); keys.push_back(v[n-1] + 1); }
		for (int i = 0; i < 20; i++) keys.push_back((T)(gen() % (1u << 22)));
		for (T key : keys) {
			const size_t lb = std::lower_bound(v.begin(), v.end(), key) - v.begin();
			const size_t ub = std::upper_bound(v.begin(), v.end(), key) - v.begin();
			const bool present = lb < ub;
			const size_t last_occ = present ? ub - 1 : (size_t)-1;
			const size_t first_occ = present ? lb : (size_t)-1;
			const T *p = v.data();

			// --- fxt-style (index of first equal / n)
			{ size_t r = bsearch(p, n, key); size_t e = present ? lb : n; if (r != e) fail("bsearch", v, key, r, e); }
			{ size_t r = bsearch_geq(p, n, key); if (r != lb) fail("bsearch_geq", v, key, r, lb); }
			if (n > 0) { std::vector<size_t> ix(n); for (size_t i = 0; i < n; i++) ix[i] = i;
				size_t r = idx_bsearch(p, n, ix.data(), key); size_t e = present ? lb : n; if (r != e) fail("idx_bsearch", v, key, r, e); }

			// --- scandum-style (index of equal element or -1)
			auto chk_find = [&](const char *name, size_t r, bool want_last) {
				size_t e = present ? (want_last ? last_occ : first_occ) : (size_t)-1;
				if (present) { if (r >= n || v[r] != key) fail(name, v, key, (long long)r, (long long)e); }
				else if (r != (size_t)-1) fail(name, v, key, (long long)r, -1);
			};
			chk_find("standard_binary_search", standard_binary_search(p, n, key), true);
			chk_find("boundless_binary_search", boundless_binary_search(p, n, key), true);
			chk_find("doubletapped_binary_search", doubletapped_binary_search(p, n, key), true);
			chk_find("monobound_binary_search", monobound_binary_search(p, n, key), true);
			chk_find("tripletapped_binary_search(ptr)", tripletapped_binary_search(p, n, key), true);
			chk_find("monobound_quaternary_search", monobound_quaternary_search(p, n, key), true);
			chk_find("breaking_linear_search", breaking_linear_search(p, n, key), true);
			if (n > 0) chk_find("Khuong_bin_search", Khuong_bin_search(p, n, key), false);

			// --- iterator versions: "lower" = first occurrence, "upper" = last occurrence, absent -> end
			auto chk_it = [&](const char *name, std::vector<T>::const_iterator it, bool want_last) {
				size_t r = it - v.cbegin();
				size_t e = present ? (want_last ? last_occ : first_occ) : n;
				if (r != e) fail(name, v, key, r, e);
			};
			auto b = v.cbegin(), en = v.cend();
			chk_it("upper_bound_standard_binary_search", upper_bound_standard_binary_search(b, en, key, id), true);
			chk_it("lower_bound_standard_binary_search", lower_bound_standard_binary_search(b, en, key, id), false);
			chk_it("upper_bound_monobound_binary_search", upper_bound_monobound_binary_search(b, en, key, id), true);
			chk_it("lower_bound_monobound_binary_search", lower_bound_monobound_binary_search(b, en, key, id), false);
			chk_it("tripletapped_binary_search(it)", tripletapped_binary_search(b, en, key, id), true);
			chk_it("upper_bound_breaking_linear_search", upper_bound_breaking_linear_search(b, en, key, id), true);
			chk_it("lower_bound_breaking_linear_search", lower_bound_breaking_linear_search(b, en, key, id), false);
			// linear with equality predicate (how tests/search/linear.cpp uses them)
			chk_it("lower_bound_linear_search(eq)", lower_bound_linear_search(b, en, key, eq), false);
			chk_it("upper_bound_linear_search(eq)", upper_bound_linear_search(b, en, key, eq), true);
			// with less-than (documented semantics)
			{ size_t r = lower_bound_linear_search(b, en, key, lt) - b; if (r != lb) fail("lower_bound_linear_search(lt) vs std::lower_bound", v, key, r, lb); }
			{ size_t r = upper_bound_linear_search(b, en, key, lt) - b; if (r != ub) fail("upper_bound_linear_search(lt) vs std::upper_bound", v, key, r, ub); }

			{ size_t r = lower_bound_standard_binary_search(b, en, key, id) - b; if (r != lb) fail("lower_bound_standard_binary_search vs std::lower_bound", v, key, r, lb); }
			if (present) { size_t r = upper_bound_monobound_binary_search(b, en, key, id) - b; if (r != last_occ) fail("upper_bound_monobound_binary_search(present key)", v, key, r, last_occ); }
			if (present) { size_t r = tripletapped_binary_search(b, en, key, id) - b; if (r >= n || v[r] != key) fail("tripletapped_binary_search(it)(present key)", v, key, r, last_occ); }
			if (present && n > 0) { size_t r = Khuong_bin_search(p, n, key); if (r >= n || v[r] != key) fail("Khuong(present key)", v, key, (long long)r, first_occ); }
			// --- true lower_bound
			{ size_t r = branchless_lower_bound(b, en, key, lt) - b; if (r != lb) fail("branchless_lower_bound(cmp)", v, key, r, lb); }
			{ size_t r = branchless_lower_bound(b, en, key, id) - b; if (r != lb) fail("branchless_lower_bound(hash)", v, key, r, lb); }
			{ size_t r = cryptanalysislib::search::lower_bound(b, en, key, lt) - b; if (r != lb) fail("search::lower_bound", v, key, r, lb); }
		}
	}
	for (auto &[k, c] : failcnt) printf("FAIL %-55s x%-5d first: %s\n", k.c_str(), c, firstfail[k].c_str());
	printf("search: %zu failing functions\n", failcnt.size());
}
