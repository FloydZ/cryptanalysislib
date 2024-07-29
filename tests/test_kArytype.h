#include <cstdint>
TEST(T, DoesNotLeak) {
    auto*l = new S;
    delete l;
}

TEST(T, constexpr) {
	constexpr S l1 = 1;
	EXPECT_EQ(l1.value(), 1);

	constexpr S l2 = 0;
	EXPECT_EQ(l2.value(), 0);

	constexpr S l3 = l1 + l2;
	EXPECT_EQ(l3.value(), 1);

	constexpr S l4 = l1 - l2;
	EXPECT_EQ(l4.value(), 1);

	constexpr S l5 = l1 * l2;
	EXPECT_EQ(l5.value(), 0);

	constexpr S l6 = l2 / l1;
	EXPECT_EQ(l6.value(), 0);


	constexpr S l7 = l2 + 0;
	EXPECT_EQ(l7.value(), 0);

	constexpr S l8 = l2 - 0;
	EXPECT_EQ(l8.value(), 0);

	constexpr S l9 = l2 * 0;
	EXPECT_EQ(l9.value(), 0);

	constexpr S l10 = l2 / 1;
	EXPECT_EQ(l10.value(), 0);
}

TEST(T, Simple) {
	S l1;
	unsigned int t1 = fastrandombytes_uint64()% PRIME;
	l1 = t1;
	EXPECT_EQ(l1, t1);
}

TEST(T, EdgeCases) {
	S l1;
	unsigned int t1 = PRIME;
	l1 = t1;
	EXPECT_EQ(l1, 0);
}

TEST(T, Zero) {
	S l1, l2, l3;
	l1 = l2 = l3 = 0;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 0);
	EXPECT_EQ(l1, 0);
	EXPECT_EQ(l2, 0);

	l3 = 1;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 0);
	EXPECT_EQ(l1, 0);
	EXPECT_EQ(l2, 0);
}

TEST(T, one) {
	S l1, l2, l3;
	l1 = l2 = l3 = 1;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 2 % PRIME);

	EXPECT_EQ(l1, 1);
	EXPECT_EQ(l2, 1);
}

TEST(T, add_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		uint64_t t1 = fastrandombytes_uint64(1ull << 63);
		uint64_t t2 = fastrandombytes_uint64(1ull << 63);


		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		EXPECT_EQ(l3.value(), (t1 + t2) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, add_signed_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		const int64_t t1 = fastrandombytes_uint64(1ull << 63);
		const int64_t t2 = fastrandombytes_uint64(1ull << 63);

		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		EXPECT_EQ(l3.value(), uint64_t(t1 + t2) % PRIME);
		EXPECT_EQ(l1, uint64_t(t1) % PRIME);
		EXPECT_EQ(l2, uint64_t(t2) % PRIME);
	}
}

TEST(T, add_uint64_t) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		const uint64_t t1 = fastrandombytes_uint64(1ull << 63);
		const uint64_t t2 = fastrandombytes_uint64(1ull << 63);

		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		const uint64_t t3 = (t1 + t2) % PRIME;
		EXPECT_EQ(l3.value(), t3);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, sub_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		unsigned int t1 = fastrandombytes_uint64(PRIME);
		unsigned int t2 = fastrandombytes_uint64(PRIME);

		l1 = t1;
		l2 = t2;

		l3 = l1 - l2;
		EXPECT_EQ(l3.value(), (t1 - t2 + PRIME) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, sub_signed_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		signed int t1 = fastrandombytes_uint64(PRIME);
		signed int t2 = fastrandombytes_uint64(PRIME);

		l1 = t1;
		l2 = t2;

		l3 = l1 - l2;
		EXPECT_EQ(l3.value(), (t1 - t2 + PRIME) % PRIME);
		EXPECT_EQ(l1, (t1 + PRIME) % PRIME);
		EXPECT_EQ(l2, (t2 + PRIME) % PRIME);
	}
}

TEST(T, sub_uint64_t) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		uint64_t t1 = (fastrandombytes_uint64(PRIME));
		uint64_t t2 = (fastrandombytes_uint64(PRIME));

		l1 = t1;
		l2 = t2;

		l3 = l1 - l2;
		EXPECT_EQ(l3.value(), (t1 - t2 + PRIME) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, addmul_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		unsigned int t1 = fastrandombytes_uint64(1ull << 15);
		unsigned int t2 = fastrandombytes_uint64(1ull << 15);
		unsigned int t3 = fastrandombytes_uint64(1ull << 15);

		l1 = t1;
		l2 = t2;
		l3 = t3;

		l3.addmul(l1, l2);
		EXPECT_EQ(l3.value(), (t3+(t1*t2)% PRIME) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, addmul_signed_simple) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		signed int t1 = fastrandombytes_uint64(1ull << 15);
		signed int t2 = fastrandombytes_uint64(1ull << 15);
		signed int t3 = fastrandombytes_uint64(1ull << 15);

		l1 = t1;
		l2 = t2;
		l3 = t3;

		l3.addmul(l1, l2);
		EXPECT_EQ(l3, (t3+(t1*t2)% PRIME) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}

TEST(T, add_mul_uint64_t) {
	S l1, l2, l3;

	for (size_t i = 0; i < TESTSIZE; ++i) {
		uint64_t t1 = fastrandombytes_uint64(1ull << 15);
		uint64_t t2 = fastrandombytes_uint64(1ull << 15);
		uint64_t t3 = fastrandombytes_uint64(1ull << 15);

		l1 = t1;
		l2 = t2;
		l3 = t3;

		l3.addmul(l1, l2);
		EXPECT_EQ(l3, (t3+(t1*t2)% PRIME) % PRIME);
		EXPECT_EQ(l1, t1 % PRIME);
		EXPECT_EQ(l2, t2 % PRIME);
	}
}


TEST(T, comparison_simple) {
	S l1, l2;
	unsigned int t1 = fastrandombytes_uint64(PRIME - 1);
	unsigned int t2 = t1 + 1;

	l1 = t1;
	l2 = t2;

	EXPECT_EQ(true, l1 != l2);
	EXPECT_EQ(false, l1 == l2);
	EXPECT_EQ(true, l1 < l2);
	EXPECT_EQ(true, l1 <= l2);
	EXPECT_EQ(false, l1 > l2);
	EXPECT_EQ(false, l1 >= l2);

	EXPECT_EQ(true, l1 == l1);
	EXPECT_EQ(true, l2 == l2);
}
