#include <gtest/gtest.h>

#include "random.h"
#include "algorithm/mq/fes.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using ::testing::InitGoogleTest;
using ::testing::Test;

uint32_t Fq_internal[496];
uint32_t Fl_internal[34];

constexpr static uint32_t n = 32;
constexpr static uint32_t m = n;

using namespace cryptanalysislib;

TEST(mq, simple) {
	const uint32_t k = n > 16 ? 16 : n/2;
	uint32_t mask = ((1ull << k) - 1) & 0xffffffff;
	for (uint32_t i = 0; i < 496; i++) {
		Fq_internal[i] = rng() & mask;
	}

	for (uint32_t i = 0; i < n + 1; i++) {
		Fl_internal[i] = rng() & mask;
	}

	Fl_internal[n + 1] = 0;



	/* clone original system in all lanes */
	uint32_t Fl2[33 * m];
	for (uint32_t i = 0; i < 33; i++)
		for (uint32_t j = 0; j < m; j++)
			Fl2[i * m + j] = Fl_internal[i];

	/* run kernel with small solution limit */
	const int count = 32;
	uint32_t buffer[m * count];
	int size[m];
	// feslite_kernel_solve(kernel, n, m, Fq_internal, Fl2, count, buffer, size);
    feslite_avx2_enum_16x16(n, m, Fq_internal, Fl2, count, buffer, size);

	/* check solution number: one lane have reached the cap*/
	bool enough = false;
	for (uint32_t lane = 0; lane < m; lane++) {
		//printf("# kernel [%s] found %d solutions in lane %d\n", name, size[lane], lane);
		enough |= (size[lane] == count);
	}
	if (!enough)
		printf("not ok: did NOT reach %d solutions\n", count);

    /* check solutions */
	for (uint32_t lane = 0; lane < m; lane++) {
		if (size[lane] == 0) {
			printf("not ok: SKIP / no solutions to test\n");
			continue;
		}

		for (int i = 0; i < size[lane]; i++) {
			uint32_t y = feslite_naive_evaluation(n, Fq_internal, Fl2, m, buffer[count * lane + i]);
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
