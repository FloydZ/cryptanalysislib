#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/nn.h"

constexpr size_t LS = 1u << 20u;
constexpr size_t d = 1;
constexpr size_t dk = 22;
constexpr static NN_Config config{256, 4, 300, 64, LS, dk, d, 0, 512};

size_t si, sj;

NN<config> algo{};

#ifdef USE_AVX2
#include "common.h"

/// bruteforce the two lists between the given start and end indices.
/// NOTE: uses avx2
/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
/// NOTE: assumes that list size is multiple of 4.
/// NOTE: this check every element on the left against 4 on the right
/// \param e1 end index of list 1
/// \param e2 end index list 2
template<typename Element, typename T>
void bruteforce_avx2_256(const Element *L1,
                         const Element *L2,
                         const size_t e1,
                         const size_t e2) noexcept {

	constexpr size_t s1 = 0, s2 = 0;
	ASSERT(e1 >= s1);
	ASSERT(e2 >= s2);

	/// difference of the memory location in the right list
	const __m128i loadr1 = {(4ull << 32u), (8ul) | (12ull << 32u)};
	const __m128i loadr2 = {1ull | (5ull << 32u), (9ul) | (13ull << 32u)};
	const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
	const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

	for (size_t i = s1; i < e1; ++i) {
		const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
		const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);
		const __m256i li3 = _mm256_set1_epi64x(L1[i][2]);
		const __m256i li4 = _mm256_set1_epi64x(L1[i][3]);

		/// NOTE: only possible because L2 is a continuous memory block
		/// NOTE: reset every loop
		T *ptr_r = (T *) L2;

		for (size_t j = s2; j < s2 + (e2 + 3) / 4; ++j, ptr_r += 16) {
			const __m256i ri = _mm256_i32gather_epi64((const long long int *) ptr_r, loadr1, 8);
			const int m1 = compare_256_64(li1, ri);

			if (m1) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *) ptr_r, loadr2, 8);
				const int m1 = compare_256_64(li2, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *) ptr_r, loadr3, 8);
					const int m1 = compare_256_64(li3, ri);

					if (m1) {
						const __m256i ri = _mm256_i32gather_epi64((const long long int *) ptr_r, loadr4, 8);
						const int m1 = compare_256_64(li4, ri);
						if (m1) {
							const size_t jprime = j * 4 + __builtin_ctz(m1);
							if (algo.compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
								si = i;
								sj = jprime;
							}
						}
					}
				}
			}
		}
	}
}

static void BM_avx2_256(benchmark::State &state) {
	using Element = typename NN<config>::Element;

	for (auto _: state) {
		bruteforce_avx2_256<Element, uint64_t>(algo.L1, algo.L2, state.range(0), state.range(0));
		algo.solution_l = si;
		algo.solution_l = sj;
	}
	state.SetComplexityN(state.range(0));
}


BENCHMARK(BM_avx2_256)->RangeMultiplier(2)->Range(128, 1u << 16)->Complexity();
#endif

static void BM_simd_256(benchmark::State &state) {
	for (auto _: state) {
		algo.bruteforce_simd_256(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_simd_256_ux4(benchmark::State &state) {
	for (auto _: state) {
		algo.bruteforce_simd_256(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_simd_256_32_ux8(benchmark::State &state) {
	for (auto _: state) {
		algo.bruteforce_simd_256_32_ux8<4>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


static void BM_simd_256_64_4x4(benchmark::State &state) {
	for (auto _: state) {
		algo.bruteforce_simd_256_64_4x4(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_simd_256)->RangeMultiplier(2)->Range(128, 1u << 16)->Complexity();
BENCHMARK(BM_simd_256_ux4)->RangeMultiplier(2)->Range(128, 1u << 16)->Complexity();
BENCHMARK(BM_simd_256_32_ux8)->RangeMultiplier(2)->Range(128, 1u << 16)->Complexity();
BENCHMARK(BM_simd_256_64_4x4)->RangeMultiplier(2)->Range(1024, 1u << 16)->Complexity();

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	algo.generate_random_instance(false);

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
