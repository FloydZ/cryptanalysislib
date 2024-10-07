#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_packed_vector.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Row = FqPackedVector<32, 3, uint64_t>;

// return true if correct, false if not
bool correct(const uint64_t t, const uint64_t a, const uint64_t b) {
	Row row1, row2, row3;
	row1.zero();
	row2.zero();
	row3.zero();
	row1.__data[0] = a;
	row2.__data[0] = b;

	Row::add(row3, row1, row2);
	uint64_t data = t;
	for (uint32_t i = 0; i < Row::length(); i++) {
		if ((data & 3u) != row3.get(i)) {
			row1.print();
			row2.print();
			row3.print();
			return false;
		}

		data >>= 2;
	}
	return true;
}

bool correct128(const __uint128_t t, const __uint128_t a, const __uint128_t b) {
	using Row = FqPackedVector<64, 3, uint64_t>;
	Row row1, row2, row3;
	row3.zero();
	row1.__data[0] = a;
	row1.__data[1] = a >> 64;
	row2.__data[0] = b;
	row2.__data[1] = b >> 64;

	Row::add(row3, row1, row2);
	__uint128_t data = t;
	for (uint32_t i = 0; i < Row::length(); i++) {
		if ((data & 3) != row3.get(i)) {
			row3.print();
			// t.print();
			return false;
		}

		data >>= 2;
	}
	return true;
}

bool correct256(const uint64x4_t t, const uint64x4_t a, const uint64x4_t b) {
	using Row = FqPackedVector<128, 3, uint64_t>;
	Row row1, row2, row3;
	row3.zero();
	row1.__data[0] = a.v64[0];
	row1.__data[1] = a.v64[1];
	row1.__data[2] = a.v64[2];
	row1.__data[3] = a.v64[3];
	row2.__data[0] = b.v64[0];
	row2.__data[1] = b.v64[1];
	row2.__data[2] = b.v64[2];
	row2.__data[3] = b.v64[3];

	Row::add(row3, row1, row2);
	uint64_t datas[4] = {t.v64[0], t.v64[1], t.v64[2], t.v64[3]};

	for (uint32_t j = 0; j < 4; j++) {
		uint64_t data = datas[j];
		for (uint32_t i = 0; i < 32; i++) {
			if ((data & 3) != row3.get(i)) {
				row1.print();
				row2.print();
				row3.print();
				t.print();
				return false;
			}

			data >>= 2;
		}
	}

	return true;
}

TEST(kAryPackedContainer3, test) {
	uint64_t a = 0b10010110, b = 0b00010101, t;
	t = Row::add_T(a, b);
	EXPECT_EQ(true, correct(t, a, b));
}

TEST(kAryPackedContainer3, popcnt_T) {
	uint64_t a = 0;
	EXPECT_EQ(0, Row::popcnt_T(a));
	a = 1;
	EXPECT_EQ(1, Row::popcnt_T(a));
	a = 2;
	EXPECT_EQ(1, Row::popcnt_T(a));

	__uint128_t b = 0;
	EXPECT_EQ(0, Row::popcnt_T<__uint128_t>(b));
	b = 1;
	EXPECT_EQ(1, Row::popcnt_T<__uint128_t>(b));
	b = 2;
	EXPECT_EQ(1, Row::popcnt_T<__uint128_t>(b));

#ifdef USE_AVX2
	uint64x4_t c = uint64x4_t::set1(0);
	EXPECT_EQ(0, Row::popcnt256_T(c));
	c = uint64x4_t::setr(1, 0, 0, 0);
	EXPECT_EQ(1, Row::popcnt256_T(c));
	c = uint64x4_t::setr(2, 0, 0, 0);
	EXPECT_EQ(1, Row::popcnt256_T(c));
#endif
}

TEST(kAryPackedContainer3, times2_T) {
	Row row1;
	uint64_t t;
	row1.zero();

	t = Row::times2_T(row1.ptr()[0]);
	EXPECT_EQ(t, 0);
	//Row::print(t);

	row1.one();
	t = Row::times2_T(row1.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);

	row1.two();
	t = Row::times2_T(row1.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);
}

TEST(kAryPackedContainer3, net_T) {
	Row row1;
	uint64_t t;
	row1.zero();

	t = Row::neg_T(row1.ptr()[0]);
	EXPECT_EQ(t, 0);
	//Row::print(t);

	row1.one();
	t = Row::neg_T(row1.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);

	row1.two();
	t = Row::neg_T(row1.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);
}

TEST(kAryPackedContainer3, sub_T) {
	Row row1, row2;
	uint64_t t = 0;
	row1.zero();
	row2.zero();

	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 0);

	row1.one();
	row2.zero();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);

	row1.two();
	row2.zero();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);

	row1.two();
	row2.one();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);

	row1.two();
	row2.two();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 0);
	//Row::print(t);

	row1.one();
	row2.two();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);

	row1.one();
	row2.one();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 0);
	//Row::print(t);

	row1.zero();
	row2.two();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);

	row1.zero();
	row2.one();
	t = Row::sub_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);
}

TEST(kAryPackedContainer3, add_T) {
	Row row1, row2;
	uint64_t t;
	row1.zero();
	row2.zero();

	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 0);

	row1.one();
	row2.zero();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);
	//Row::print_binary(t, 0, 32);

	row1.two();
	row2.zero();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);
	//Row::print_binary(t, 0, 32);

	row1.zero();
	row2.one();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);
	//Row::print_binary(t, 0, 32);

	row1.zero();
	row2.two();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 12297829382473034410u);
	//Row::print(t);
	//Row::print_binary(t, 0, 32);

	row1.two();
	row2.one();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 0);
	//Row::print(t);
	//Row::print_binary(t, 0, 32);

	row1.two();
	row2.two();
	t = Row::add_T(row1.ptr()[0], row2.ptr()[0]);
	EXPECT_EQ(t, 6148914691236517205u);
	//Row::print(t);
}

TEST(kAryPackedContainer3, add128_T) {
	__uint128_t row1, row2, t;
	row1 = 0;
	row2 = 0;

	t = Row::add_T<__uint128_t>(row1, row2);
	EXPECT_EQ(true, correct128(t, row1, row2));

	// set it to one
	row1 = (__uint128_t(6148914691236517205u) << 64) | 6148914691236517205u;
	t = Row::add_T<__uint128_t>(row1, row2);
	EXPECT_EQ(true, correct128(t, row1, row2));

	// set it to two
	row1 = (__uint128_t(12297829382473034410u) << 64) | 12297829382473034410u;
	t = Row::add_T<__uint128_t>(row1, row2);
	EXPECT_EQ(true, correct128(t, row1, row2));

	row1 = (__uint128_t(6148914691236517205u) << 64) | 6148914691236517205u;
	row2 = (__uint128_t(12297829382473034410u) << 64) | 12297829382473034410u;
	t = Row::add_T<__uint128_t>(row1, row2);
	EXPECT_EQ(true, correct128(t, row1, row2));
}

// TODO currently there is the transition from avx2 to generic simd
// TEST(kAryPackedContainer3, add256_T) {
//     using S = Row::S;
// 	S row1 = S::set1(0),
// 	  row2 = S::set1(0),
// 	  t;
// 
// 	t = Row::add256_T(row1, row2);
// 	EXPECT_EQ(true, correct256(t, row1, row2));
// 
// 	// set it to one
// 	row1 = uint64x4_t::setr(6148914691236517205u, 6148914691236517205u, 6148914691236517205u, 6148914691236517205u);
// 	t = Row::add256_T(row1, row2);
// 	EXPECT_EQ(true, correct256(t, row1, row2));
// 
// 	// set it to two
// 	row1 = uint64x4_t::setr(12297829382473034410u, 12297829382473034410u, 12297829382473034410u, 12297829382473034410u);
// 	t = Row::add256_T(row1, row2);
// 	EXPECT_EQ(true, correct256(t, row1, row2));
// 
// 	row1 = uint64x4_t::setr(12297829382473034410u, 12297829382473034410u, 12297829382473034410u, 12297829382473034410u);
// 	row2 = uint64x4_t::setr(6148914691236517205u, 6148914691236517205u, 6148914691236517205u, 6148914691236517205u);
// 	t = Row::add256_T(row1, row2);
// 	EXPECT_EQ(true, correct256(t, row1, row2));
// }

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
