#include <gtest/gtest.h>

#include "random.h"
#include "algorithm/mq/fes.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr static uint32_t n = 22;
constexpr static uint32_t m = 16;
constexpr static uint32_t k = n > 16 ? 16 : n/2;
constexpr static uint32_t w = 10;

uint32_t Fq_internal[496];
uint32_t Fl_internal[34];
uint32_t Fl[33 * m];


using namespace cryptanalysislib;

/// NOTE: creates m equal equation systems
void create_equation_system(const uint32_t x=0) {
	uint32_t mask = ((1ull << k) - 1u) & 0xffffffff;
	// for (uint32_t i = 0; i < 496; i++) { Fq_internal[i] = rng() & mask; }
	// for (uint32_t i = 0; i < n + 1; i++) { Fl_internal[i] = rng() & mask; }

    // deterministic version, maybe good for testing
	for (uint32_t i = 0; i < 496; i++) { Fq_internal[i] = ((7*i*mask + k*mask - n*m*k)/ n) & mask; }
	for (uint32_t i = 0; i < n + 1; i++) { Fl_internal[i] = ((i*mask + k*mask))/n & mask; }
	Fl_internal[n + 1] = 0;

	/* clone original system in all lanes */
	for (uint32_t j = 0; j < m; j++) {
		for (uint32_t i = 1; i < n + 1; i++) {
			Fl[j + m * i] = Fl_internal[i];
		}
	}

    if (x == 0) { return; }
	for (uint32_t j = 0; j < m; j++) {
		const uint32_t tmp = feslite_naive_evaluation(n, Fq_internal, &Fl[j], m, x);
		Fl[j] = tmp;
	}
}

/// \return true/false if correct or nto
bool check_solutions(const int size[m],
					 const uint32_t *buffer,
					 const uint32_t count,
					 const uint32_t x = 0) {
	for (uint32_t lane = 0; lane < m; lane++) {
		printf("# found %d solutions in lane %d\n", size[lane], lane);
	}

    bool ret = true;

    /* check solutions */
	for (uint32_t lane = 0; lane < m; lane++) {
		if (size[lane] == 0) {
			printf("not ok: SKIP / no solutions to test\n");
			ret = false;
			continue;
		}

		for (int i = 0; i < size[lane]; i++) {
			if (x != 0) {
				if (buffer[count*lane + i] != x) {
					printf("not ok: non golden solution reported %08x != %08x in lane %d\n",
					        buffer[count * lane + i], x, lane);
				}
			}
			uint32_t y = feslite_naive_evaluation(n, Fq_internal, Fl, m, buffer[count * lane + i], w);
			if (y != 0) {
                ret = false;
				printf("not ok: incorrectly reported F[%08x] = %08x in lane %d\n",
				        buffer[count * lane + i], y, lane);
				break;
			}
		}

		for (int i = 0; i < size[lane]; i++) {
			for (int j = i + 1; j < size[lane]; j++) {
				if (buffer[count * lane + i] == buffer[count * lane + j]) {
                    ret = false;
					printf("not ok:returned buffer[%d] = buffer[%d] in lane %d\n",
					       i, j, lane);
					i = size[lane];
					break;
				}
			}
		}
	}

    return ret;
}

/// checks, that every lane found min(count, #solutions) solutions with at most
/// `max_weight` bits set (0: any weight), all of them are in the brute force list
void check_brute_force(const int size[m],
                       const uint32_t *buffer,
                       const uint32_t count,
                       const uint32_t max_weight) {
	// NOTE: all lanes contain the same system
	std::vector<uint32_t> sols;
	for (uint32_t y = 0; y < (1u << n); ++y) {
		if ((max_weight > 0) && ((uint32_t)__builtin_popcount(y) > max_weight)) { continue; }
		if (feslite_naive_evaluation(n, Fq_internal, Fl, m, y) == 0) { sols.push_back(y); }
	}

	const size_t expected = std::min<size_t>(count, sols.size());
	for (uint32_t lane = 0; lane < m; lane++) {
		EXPECT_EQ(size_t(size[lane]), expected) << lane;
		for (int i = 0; i < size[lane]; i++) {
			const uint32_t y = buffer[count * lane + i];
			EXPECT_TRUE(std::find(sols.begin(), sols.end(), y) != sols.end()) << y;
		}
	}
}

#ifdef USE_AVX2
TEST(mq, full) {
	constexpr int count = 32;
	uint32_t buffer[m * count];
	int size[m];
	const uint32_t x = (1<<9);
	create_equation_system(x);
    feslite_avx2_enum_16x16(n, m, Fq_internal, Fl, count, buffer, size);
    const auto r = check_solutions(size, buffer, count);
    EXPECT_EQ(r, true);
	check_brute_force(size, buffer, count, 0);
}

TEST(mq, weight) {
	/* run kernel with small solution limit */
	const int count = 32;
	uint32_t buffer[m * count];
	int size[m];

	const uint32_t x = (1<<10) ^ (1<<9);
	create_equation_system(x);

	EXPECT_EQ(feslite_avx2_enum_16x16_w(n, m, w, Fq_internal, Fl, count, buffer, size), 0);
    const auto r = check_solutions(size, buffer, count, x);
    EXPECT_EQ(r, true);
	check_brute_force(size, buffer, count, w);

	// other weights
	for (const uint32_t w2 : {1u, 2u, 5u, 8u, 9u, 13u}) {
		EXPECT_EQ(feslite_avx2_enum_16x16_w(n, m, w2, Fq_internal, Fl, count, buffer, size), 0);
		check_brute_force(size, buffer, count, w2);
	}
}

#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
