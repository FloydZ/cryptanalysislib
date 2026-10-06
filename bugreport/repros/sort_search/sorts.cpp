// differential tests for sort/ algorithms vs std::sort
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <string>

#include <functional>
#include <cstdlib>
#include "sort/robinhoodsort.h"
#include "sort/vergesort.h"
#include "sort/timsort.h"
#include "sort/vv_radixsort.h"
#include "sort/common.h"
#include "sort/float_radixsort.h"
#include "sort/sorting_network/common.h"

static std::mt19937_64 gen(1234);
static int fails = 0;
#define FAIL(...) do { if (fails++ < 60) { printf(__VA_ARGS__); printf("\n"); } } while(0)

static const size_t sizes[] = {0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,100,127,128,129,255,256,257,511,512,513,1000,4096,70000};

template<typename T>
std::vector<T> gen_input(size_t n, int kind) {
	std::vector<T> v(n);
	for (size_t i = 0; i < n; i++) {
		switch (kind) {
			case 0: v[i] = (T)gen(); break;                         // random full range
			case 1: v[i] = (T)(gen() % 4); break;                   // many dups
			case 2: v[i] = (T)42; break;                            // all equal
			case 3: v[i] = (T)i; break;                             // sorted
			case 4: v[i] = (T)(n - i); break;                       // reversed
			case 5: v[i] = (gen() & 1) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min(); break;
			case 6: v[i] = (T)(gen() % 1000); break;                // small range
			case 7: v[i] = std::numeric_limits<T>::max() - (T)(gen() % 3); break;
		}
	}
	return v;
}
static const char *kinds[] = {"random","dups4","allequal","sorted","reversed","minmax","range1000","nearmax"};

template<typename T, typename F>
void check(const char *name, F f, size_t maxn = (size_t)-1, size_t minn = 0) {
	for (size_t n : sizes) {
		if (n > maxn || n < minn) continue;
		for (int kind = 0; kind < 8; kind++) {
			auto v = gen_input<T>(n, kind);
			auto ref = v;
			std::sort(ref.begin(), ref.end());
			f(v.data(), n);
			if (v != ref) {
				FAIL("[%s] T=%zu bytes n=%zu kind=%s: mismatch", name, sizeof(T), n, kinds[kind]);
			}
		}
	}
}

template<typename T>
void run_all() {
	if constexpr (sizeof(T) < 8) check<T>("rhmergesort", [](T *p, size_t n){ rhmergesort<T>(p, n); }, (size_t)-1, 1);
	check<T>("vergesort", [](T *p, size_t n){ vergesort::vergesort(p, p+n, [](T a, T b){return a<b;}); });
	check<T>("timsort", [](T *p, size_t n){ gfx::timsort(p, p+n); });
}

int main() {
	setvbuf(stdout, nullptr, _IONBF, 0);
	// counting sort u8
	// counting_sort_u8: header does not compile on ARM (simd/neon.h)

	run_all<uint8_t>();
	run_all<uint16_t>();
	run_all<uint32_t>();
	run_all<uint64_t>();
	run_all<int32_t>();

	// StaticSort / StaticTimSort for network sizes
	auto net = [](auto N_c) {
		constexpr unsigned N = decltype(N_c)::value;
		for (int kind = 0; kind < 8; kind++) {
			for (int rep = 0; rep < 20; rep++) {
				auto v = gen_input<uint32_t>(N, kind);
				auto r = v; std::sort(r.begin(), r.end());
				auto w = v;
				StaticSort<N>()(v.data());
				if (v != r) FAIL("[StaticSort<%u>] kind=%s mismatch", N, kinds[kind]);
				StaticTimSort<N>()(w.data());
				if (w != r) FAIL("[StaticTimSort<%u>] kind=%s mismatch", N, kinds[kind]);
			}
		}
	};
	net(std::integral_constant<unsigned,1>{}); net(std::integral_constant<unsigned,2>{});
	net(std::integral_constant<unsigned,3>{}); net(std::integral_constant<unsigned,7>{});
	net(std::integral_constant<unsigned,8>{}); net(std::integral_constant<unsigned,9>{});
	net(std::integral_constant<unsigned,15>{}); net(std::integral_constant<unsigned,16>{});
	net(std::integral_constant<unsigned,17>{}); net(std::integral_constant<unsigned,31>{});
	net(std::integral_constant<unsigned,32>{}); net(std::integral_constant<unsigned,33>{});
	net(std::integral_constant<unsigned,63>{}); net(std::integral_constant<unsigned,64>{});

	// scalar sorting network sort_i32x8 / u32x8 (non-AVX fallback)
	for (int kind = 0; kind < 8; kind++) {
		for (int rep = 0; rep < 50; rep++) {
			auto v = gen_input<int32_t>(8, kind);
			auto r = v; std::sort(r.begin(), r.end());
			auto orig = v;
			sortingnetwork_sort_i32x8(v.data());
			if (v != r) {
				FAIL("[sortingnetwork_sort_i32x8] kind=%s in=%d,%d,%d,%d,%d,%d,%d,%d out=%d,%d,%d,%d,%d,%d,%d,%d",
				     kinds[kind], orig[0],orig[1],orig[2],orig[3],orig[4],orig[5],orig[6],orig[7],
				     v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
				break;
			}
		}
		for (int rep = 0; rep < 50; rep++) {
			auto v = gen_input<uint32_t>(8, kind);
			auto r = v; std::sort(r.begin(), r.end());
			sortingnetwork_sort_u32x8(v.data());
			if (v != r) { FAIL("[sortingnetwork_sort_u32x8] kind=%s mismatch", kinds[kind]); break; }
		}
	}

	// sort_minmax_branchless
	{
		auto t = [](auto a0, auto b0) {
			auto a = a0, b = b0;
			sort_minmax_branchless(a, b);
			if (a != std::min(a0,b0) || b != std::max(a0,b0))
				FAIL("[sort_minmax_branchless] T=%zuB in=(%llu,%llu) out=(%llu,%llu)", sizeof(a0),
				     (unsigned long long)a0, (unsigned long long)b0, (unsigned long long)a, (unsigned long long)b);
		};
		t((uint32_t)5, (uint32_t)3);
		t((uint16_t)5, (uint16_t)3);
		t((uint64_t)5, (uint64_t)3);
		t((int32_t)5, (int32_t)3);
	}

	// hoare_partition / fulcrum_partition: check permutation + partition property
	{
		for (int rep = 0; rep < 200; rep++) {
			size_t n = 2 + gen() % 40;
			std::vector<uint64_t> v(n);
			for (auto &x : v) x = gen();          // full 64-bit values
			auto sorted_in = v; std::sort(sorted_in.begin(), sorted_in.end());
			auto w = v;
			size_t p = hoare_partition(w.data(), 0, n - 1);
			auto ws = w; std::sort(ws.begin(), ws.end());
			bool ok = ws == sorted_in;
			for (size_t i = 0; i < p && ok; i++) if (w[i] > w[p]) ok = false;
			for (size_t i = p+1; i < n && ok; i++) if (w[i] < w[p]) ok = false;
			if (!ok) { FAIL("[hoare_partition<uint64_t>] n=%zu not a permutation/partition (p=%zu)", n, p); break; }
		}
		for (int rep = 0; rep < 200; rep++) {
			size_t n = 2 + gen() % 40;
			std::vector<uint64_t> v(n);
			for (auto &x : v) x = gen();
			auto sorted_in = v; std::sort(sorted_in.begin(), sorted_in.end());
			auto w = v;
			size_t p = fulcrum_partition(w.data(), 0, n - 1);
			auto ws = w; std::sort(ws.begin(), ws.end());
			bool ok = ws == sorted_in;
			for (size_t i = 0; i < p && ok; i++) if (w[i] > w[p]) ok = false;
			for (size_t i = p+1; i < n && ok; i++) if (w[i] < w[p]) ok = false;
			if (!ok) { FAIL("[fulcrum_partition<uint64_t>] n=%zu bad", n); break; }
		}
	}

	// float RadixSort
	{
		RadixSort rs;
		const size_t fs[] = {1,2,3,7,8,9,16,17,100,1000,10,1000,5000,3};
		for (size_t n : fs) {
			for (int kind = 0; kind < 4; kind++) {
				std::vector<float> v(n);
				for (size_t i = 0; i < n; i++) {
					switch (kind) {
						case 0: v[i] = (float)((int64_t)(gen() % 2000001) - 1000000) / 7.f; break;
						case 1: v[i] = -(float)(gen() % 1000) - 1.f; break;          // all negative
						case 2: v[i] = (gen()&1) ? 0.0f : -0.0f; break;
						case 3: v[i] = (float)(gen() % 1000); break;
					}
				}
				uint32_t *idx = rs.Sort(v.data(), n).GetIndices();
				std::vector<float> out(n);
				std::vector<uint32_t> perm(idx, idx+n);
				std::sort(perm.begin(), perm.end());
				bool isperm = true;
				for (size_t i = 0; i < n; i++) if (perm[i] != i) isperm = false;
				bool sorted = true;
				for (size_t i = 1; i < n; i++) if (v[idx[i-1]] > v[idx[i]]) sorted = false;
				if (!isperm || !sorted) FAIL("[RadixSort float] n=%zu kind=%d perm=%d sorted=%d", n, kind, isperm, sorted);
			}
			std::vector<uint32_t> u(n);
			for (auto &x : u) x = (uint32_t)gen();
			uint32_t *idx = rs.Sort(u.data(), n).GetIndices();
			for (size_t i = 1; i < n; i++) if (u[idx[i-1]] > u[idx[i]]) { FAIL("[RadixSort u32] n=%zu", n); break; }
		}
	}

	printf("sorts: %d failures\n", fails);
	return 0;
}
