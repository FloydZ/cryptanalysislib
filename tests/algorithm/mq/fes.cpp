#include <gtest/gtest.h>

#include "random.h"
#include "algorithm/mq/fes.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using ::testing::InitGoogleTest;
using ::testing::Test;

uint32_t Fq_internal[496];
uint32_t Fl_internal[34];

constexpr static uint32_t n = 22;
constexpr static uint32_t m = 16;
constexpr static uint32_t k = n > 16 ? 16 : n/2;
constexpr static uint32_t w = 10;

using namespace cryptanalysislib;

TEST(mq, simple) {
	combination_revdoor c(10, 4);
	uint32_t k1 = 1, k2 = 2;
	do {
		c.print_deltaset();
		std::cout << " " << k1 << " " << k2 << std::endl;
	} while (c.next(&k1, &k2));
	return;

	uint32_t mask = ((1ull << k) - 1u) & 0xffffffff;
	for (uint32_t i = 0; i < 496; i++) { Fq_internal[i] = rng() & mask; }
	for (uint32_t i = 0; i < n + 1; i++) { Fl_internal[i] = rng() & mask; }
	// for (uint32_t i = 0; i < 496; i++) { Fq_internal[i] = ((7*i*mask + k*mask - n*m*k)/ n) & mask; }
	// for (uint32_t i = 0; i < n + 1; i++) { Fl_internal[i] = ((i*mask + k*mask))/n & mask; }
	Fl_internal[n + 1] = 0;

	/* clone original system in all lanes */
	uint32_t Fl2[33 * m];
	for (uint32_t j = 0; j < m; j++) {
		for (uint32_t i = 1; i < n + 1; i++) {
			Fl2[j + m * i] = Fl_internal[i];
		}
	}
	for (uint32_t j = 0; j < m; j++) {
		// Fl2[k] = 1;
		// const uint32_t x = (1<<8) ^ (1<<9);
		const uint32_t x = (1<<9);
		const uint32_t tmp = feslite_naive_evaluation(n, Fq_internal, &Fl2[j], m, x);
		Fl2[j] = tmp;
	}

	//for (uint32_t i = 0; i < 33; i++) {
	//	for (uint32_t j = 0; j < m; j++) {
	//		Fl2[i * m + j] = Fl_internal[i];
	//	}
	//	// implant the solution
	//	uint32_t x = 0;
	//	Fl2[i] = feslite_naive_evaluation(n, Fq_internal, &Fl2[i], m, x);
	//}

	/* run kernel with small solution limit */
	const int count = 32;
	uint32_t buffer[m * count];
	int size[m];
	// feslite_kernel_solve(kernel, n, m, Fq_internal, Fl2, count, buffer, size);
    //feslite_avx2_enum_16x16(n, m, Fq_internal, Fl2, count, buffer, size);
	feslite_avx2_enum_16x16_w(n, m, w, Fq_internal, Fl2, count, buffer, size);

	/* check solution number: one lane have reached the cap*/
	// bool enough = false;
	for (uint32_t lane = 0; lane < m; lane++) {
		printf("# found %d solutions in lane %d\n", size[lane], lane);
		//enough |= (size[lane] == count);
	}
	// if (!enough) {
	// 	printf("not ok: did NOT reach %d solutions\n", count);
	// }

    /* check solutions */
	for (uint32_t lane = 0; lane < m; lane++) {
		if (size[lane] == 0) {
			printf("not ok: SKIP / no solutions to test\n");
			continue;
		}

		for (int i = 0; i < size[lane]; i++) {
			uint32_t y = feslite_naive_evaluation(n, Fq_internal, Fl2, m, buffer[count * lane + i], w);
			if (y != 0) {
				printf("not ok: incorrectly reported F[%08x] = %08x in lane %d\n",
				        buffer[count * lane + i], y, lane);
				break;
			}
		}
		for (int i = 0; i < size[lane]; i++) {
			for (int j = i + 1; j < size[lane]; j++) {
				if (buffer[count * lane + i] == buffer[count * lane + j]) {
					printf("not ok:returned buffer[%d] = buffer[%d] in lane %d\n",
					       i, j, lane);
					i = size[lane];
					break;
				}
			}
		}
	}

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
