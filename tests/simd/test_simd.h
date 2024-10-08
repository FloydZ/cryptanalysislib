﻿TEST(T, random) {
	const S t1 = S::random();

	uint32_t atleast_one_not_zero = false;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		if (t1.d[i] > 0) {
			atleast_one_not_zero = true;
			//	break;
		}
	}

	ASSERT_EQ(atleast_one_not_zero, true);
}

TEST(T, set1) {
	constexpr S t1 = S::set1(0);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], 0);
	}

	const S t2 = S::set1(1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t2.d[i], 1);
	}
}

TEST(T, unalinged_load) {
	constexpr S::limb_type data[S::LIMBS] = {0};

	constexpr S t1 = S::unaligned_load(data);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], 0u);
	}
}

TEST(T, alinged_load) {
	alignas(256) constexpr  S::limb_type data[S::LIMBS] = {0};
	constexpr S t1 = S::aligned_load(data);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], 0u);
	}
}

TEST(T, unalinged_store_zero) {
	constexpr S t1 = S::set1(0);
	S::limb_type data[S::LIMBS] = {0};
	S::unaligned_store(data, t1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], data[i]);
	}
}

TEST(T, unaligned_store) {
	const S t1 = S::random();
	S::limb_type data[S::LIMBS] = {0};

	S::unaligned_store(data, t1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], data[i]);
	}
}

TEST(T, aligned_store) {
	const S t1 = S::random();
	alignas(256) S::limb_type data[S::LIMBS] = {0};

	S::aligned_store(data, t1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t1.d[i], data[i]);
	}
}

TEST(T, constexpr_logic) {
	constexpr S t1 = S::set1(0);
	constexpr S t2 = S::set1(1);
	S t3 = t1 + t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t3.d[i], 1);
	}

	constexpr S t4 = t2 - t1;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t4.d[i], 1);
	}

	constexpr S t5 = t2 - t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t5.d[i], 0);
	}

	constexpr S t6 = t1 ^ t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t6.d[i], 1);
	}

	constexpr S t7 = t1 | t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t7.d[i], 1);
	}

	constexpr S t8 = t1 & t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t8.d[i], 0);
	}

	constexpr S t9 = ~t1;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t9.d[i], (S::limb_type)(-1ull));
	}

	constexpr S t10 = S::mullo(t1, t2);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t10.d[i], 0);
	}

	constexpr S t11 = S::slli(t1, 1u);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t11.d[i], 0);
	}

	constexpr S t12 = S::slli(t2, 1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t12.d[i], 2);
	}
}

TEST(T, logic) {
	const S t1 = S::set1(0);
	const S t2 = S::set1(1);
	S t3 = t1 + t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t3.d[i], 1);
	}

	const S t4 = t2 - t1;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t4.d[i], 1);
	}

	const S t5 = t2 - t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t5.d[i], 0);
	}

	const S t6 = t1 ^ t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t6.d[i], 1);
	}

	const S t7 = t1 | t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t7.d[i], 1);
	}

	const S t8 = t1 & t2;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t8.d[i], 0);
	}

	const S t9 = ~t1;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t9.d[i], (S::limb_type)(-1ull));
	}

	const S t10 = S::mullo(t1, t2);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t10.d[i], 0);
	}

	const S t11 = S::slli(t1, 1u);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t11.d[i], 0);
	}

	const S t12 = S::slli(t2, 1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(t12.d[i], 2);
	}
}

TEST(T, all_equal) {
	for (uint8_t j = 0; j < 255; j++) {
		const S t1 = S::set1(j);
		const bool t = S::all_equal(t1);
		EXPECT_EQ(true, t);
	}

	if constexpr (S::LIMBS > 1) {
		for (uint8_t j = 0; j < 255; j++) {
			S t1 = S::set1(j);
			t1[1] = j + 1;

			const bool t = S::all_equal(t1);
			EXPECT_EQ(false, t);
		}
	}
}

TEST(T, reverse) {
	S::limb_type d[S::LIMBS];
	for (uint32_t i = 0; i < S::LIMBS; ++i) { d[i] = i; }

	const S t1 = S::template load<false>(d);
	const S t2 = S::reverse(t1);
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		const auto k1 = t1.d[S::LIMBS - i - 1u];
		const auto k2 = t2.d[i];
		EXPECT_EQ(k1, k2);
	}
}

TEST(T, compare) {
	if constexpr (S::LIMBS <= 64) {
		S t1 = S::set1(1);
		S t2 = S::set1(0);
		uint64_t v1 = S::gt(t2, t1);
		uint64_t v2 = S::lt(t2, t1);
		uint64_t v3 = S::gt(t1, t2);
		uint64_t v4 = S::lt(t1, t2);

		uint64_t k2 = S::LIMBS == 64 ? -1ull : (1ull << (S::LIMBS)) - 1ull;
		EXPECT_EQ(v1, 0);
		EXPECT_EQ(v2, k2);
		EXPECT_EQ(v3, k2);
		EXPECT_EQ(v4, 0);

		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			t2.d[i] = 1;

			k2 ^= 1ull << i;

			v1 = S::gt(t2, t1);
			v2 = S::lt(t2, t1);
			v3 = S::gt(t1, t2);
			v4 = S::lt(t1, t2);

			EXPECT_EQ(v1, 0);
			EXPECT_EQ(v2, k2);
			EXPECT_EQ(v3, k2);
			EXPECT_EQ(v4, 0);
		}


		t1 = S::set1((S::limb_type)-1ull);
		t2 = S::set1(0);
		v1 = S::gt(t2, t1);
		v2 = S::lt(t2, t1);
		v3 = S::gt(t1, t2);
		v4 = S::lt(t1, t2);

		k2 = S::LIMBS == 64 ? -1ull : (1ull << (S::LIMBS)) - 1ull;
		EXPECT_EQ(v1, 0);
		EXPECT_EQ(v2, k2);
		EXPECT_EQ(v3, k2);
		EXPECT_EQ(v4, 0);

		// NOTE: only valid test if unsigned
		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			t2.d[i] = 1;

			v1 = S::gt(t2, t1);
			v2 = S::lt(t2, t1);
			v3 = S::gt(t1, t2);
			v4 = S::lt(t1, t2);

			EXPECT_EQ(v1, 0);
			EXPECT_EQ(v2, k2);
			EXPECT_EQ(v3, k2);
			EXPECT_EQ(v4, 0);
		}
	}
}

TEST(T, move) {
	if constexpr ((S::LIMBS <= 64) && ((S::LIMBS * sizeof(S::limb_type)) <= 64)) {
		S t1 = S::set1(0);
		uint64_t val = 0;
		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			const uint64_t ret = S::move(t1);
			EXPECT_EQ(ret, val);

			t1[i] = (S::limb_type)-1ull;
			val ^= 1ull << i;
		}
	}
}
