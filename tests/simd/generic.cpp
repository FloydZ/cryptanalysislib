#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO write more tests for different I
// TODO write more tests if N == 8, s.t T = uint32x8_t
using I = uint32_t;
TEST(generic, random) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		T t1 = T::random();

		uint32_t atleast_one_not_zero = false;
		for (uint32_t i = 0; i < limbs; ++i) {
			if (t1.d[i] > 0) {
				atleast_one_not_zero = true;
				//	break;
			}
		}

		ASSERT_EQ(atleast_one_not_zero, true);
	});
}

TEST(generic, set1) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		T t1 = T::set1(0);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t1.d[i], 0);
		}

		T t2 = T::set1(1);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t2.d[i], 1);
		}
	});
}

TEST(generic, set) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		uint32_t pos = 5;
		I data[limbs] = {0};
		data[pos] = 1;
		T t1 = T::setr(data);

		for (uint32_t i = 0; i < limbs; ++i) {
			if (i == pos) {
				EXPECT_EQ(t1.d[i], 1);
				continue;
			}
			EXPECT_EQ(t1.d[i], 0);
		}

		//t1 = T::set(data);
		//for (uint32_t i = 0; i < limbs; ++i) {
		//	  if (i == (limbs-pos)){
		//		  EXPECT_EQ(t1.d[i] , 1);
		//		  continue;
		//	  }
		//	  EXPECT_EQ(t1.d[i] , 0);
		//}
	});
}

TEST(generic, unalinged_load) {
	I data[1024] = {0};
	constexpr_for<6, 32, 4>([&data](const auto limbs) {
		using T = TxN_t<I, limbs>;

		T t1 = T::unaligned_load(data);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t1.d[i], 0u);
		}
	});
}

TEST(generic, alinged_load) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		alignas(256) I data[limbs] = {0};

		T t1 = T::aligned_load(data);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t1.d[i], 0u);
		}
	});
}

TEST(generic, unalinged_store) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		alignas(32) I data[limbs] = {0};

		T t1 = T::random();
		T::unaligned_store(data, t1);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t1.d[i], data[i]);
		}
	});
}

TEST(gerenric, alinged_store) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		alignas(256) I data[limbs] = {0};

		T t1 = T::random();
		T::aligned_store(data, t1);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t1.d[i], data[i]);
		}
	});
}

// TODO finish
TEST(uint8x32_t, logic) {
	constexpr_for<6, 32, 4>([](const auto limbs) {
		using T = TxN_t<I, limbs>;
		const T t1 = T::set1(0);
		const T t2 = T::set1(1);
		T t3 = T::set1(2);

		t3 = t1 + t2;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 1);
		}

		t3 = t2 - t1;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 1);
		}

		t3 = t2 - t2;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 0);
		}

		t3 = t1 ^ t2;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 1);
		}

		t3 = t1 | t2;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 1);
		}

		t3 = t1 & t2;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 0);
		}

		t3 = ~t1;
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], uint8_t(-1u));
		}

		t3 = T::mullo(t1, t2);
		for (uint32_t i = 0; i < limbs; ++i) {
			EXPECT_EQ(t3.d[i], 0);
		}

		t3 = T::slli(t1, 1);
		for (uint32_t i = 0; i < 32; ++i) {
			EXPECT_EQ(t3.d[i], 0);
		}

		t3 = T::slli(t2, 1);
		for (uint32_t i = 0; i < 32; ++i) {
			EXPECT_EQ(t3.d[i], 2);
		}
	});
}

TEST(gerenric, info) {
	TxN_t<uint16_t, 128>::info();
}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
