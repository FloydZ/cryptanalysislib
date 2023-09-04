#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "combination/fq/chase.h"
#include "list/enumeration/fq.h"
#include "math/ipow.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
constexpr uint32_t n = 6, q = 3, w = 3;


TEST(Fq, grey) {
	auto c = Combinations_Fq_Chase(n,q,w);
	const size_t size = c.gray_size;
	uint16_t *cl = (uint16_t *)malloc(size * sizeof(uint16_t));
	ASSERT(cl);
	c.changelist_mixed_radix_grey(cl);
	for (int i = 0; i < size; ++i) {
		printf("%d ", cl[i]);
	}
	printf("\n");

	free(cl);
}

TEST(Fq, chase) {
	auto c = Combinations_Fq_Chase(n,q,w);
	const size_t size = c.chase_size;
	uint16_t *cl = (uint16_t *)malloc(size * sizeof(uint16_t));
	ASSERT(cl);

	int r = 0, j = 0;
	for (uint32_t i = 0; i < bc(n, w); ++i) {
		c.print_chase_state(r, j);
		c.chase(&r, &j);
		/// TODO check
	}

	free(cl);
}

TEST(Fq, changelist_chase) {
	auto c = Combinations_Fq_Chase(n,q,w);
	constexpr size_t size = bc(n, w);
	auto cl = std::array<std::pair<uint16_t, uint16_t>, size>();

	c.changelist_chase(cl.data());
	for (uint32_t i = 0; i < bc(n, w); ++i) {
		/// TODO check
	}
}

TEST(Fq, build_list) {
	auto c = Combinations_Fq_Chase(n,q,w);
	constexpr size_t size = bc(n, w);
	auto chase_cl = std::array<std::pair<uint16_t, uint16_t>, size>();
	uint16_t *gray_cl = (uint16_t *)malloc(size * sizeof(uint16_t));

	c.changelist_mixed_radix_grey(gray_cl);
	c.changelist_chase(chase_cl.data());
	c.build_list(gray_cl, chase_cl.data());

	for (uint32_t i = 0; i < bc(n, w); ++i) {
		/// TODO check
	}
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
