TEST(NAME, access) {
	K t;
	for (uint32_t i = 0; i < K::length(); ++i) {
		t.set(i,i);
	}
	for (uint32_t i = 0; i < K::length(); ++i) {
		const auto d = t.get(i);
		EXPECT_EQ(d, i % PRIME);

		const auto d2 = t[i];
		EXPECT_EQ(d2, i % PRIME);
	}

	for (uint32_t i = 0; i < K::length(); ++i) {
		t[i] = 0;
	}

	for (uint32_t i = 0; i < K::length(); ++i) {
		const auto d = t.get(i);
		EXPECT_EQ(d, 0);

		const auto d2 = t[i];
		EXPECT_EQ(d2, 0);
	}
}

TEST(NAME, random) {
	auto t = K();
	t.random();
	for (uint32_t i = 0; i < K::length(); i++){
		EXPECT_LE(t.get(i), PRIME);
	}
}

TEST(NAME, comparsion) {
	auto t1 = K();
	auto t2 = K();
	for (uint32_t i = 1; i < K::length(); i++){
		t2.set(1, i);
		EXPECT_EQ(t1.is_lower(t2), true);
		EXPECT_EQ(t1.is_greater(t2), false);
		EXPECT_EQ(t1.is_equal(t2), false);
		EXPECT_EQ(t2.is_equal(t2), true);
	}
}


TEST(NAME, mod_T) {
	constexpr uint32_t qbits = bits_log2(PRIME);
	for (uint64_t i = 0; i < PRIME; i++) {
		const auto d = K::mod_T<uint32_t>(i);
		EXPECT_EQ(d, i % PRIME);
	}

	for (uint64_t i = 0; i < PRIME; i++) {
		const auto d1 = ((i % PRIME) << qbits) | (i % PRIME);
		const auto d2 = K::mod_T<uint32_t>((i << qbits) | i);
		EXPECT_EQ(d1, d2);
	}
}

TEST(NAME, add) {
	auto t1 = K();
	auto t2 = K();
	auto t3 = K();

	for (uint32_t a = 0; a < K::size() - 1; ++a) {
		for (uint32_t b = a+1; b < K::size(); ++b) {
			for (size_t j = 0; j < NR_TESTS; ++j) {
				t3.zero();
				t1.random();
				t2.random();

				K::add(t3, t1, t2, a, b);
				for (uint32_t i = a; i < b; i++) {
					EXPECT_EQ(t3.get(i), (t1.get(i) + t2.get(i)) % PRIME);
				}

				for (uint32_t i = 0; i < a; i++) {
					EXPECT_EQ(t3.get(i), 0);
				}

				for (uint32_t i = b; i < K::size(); i++) {
					EXPECT_EQ(t3.get(i), 0);
				}
			}
		}
	}
}

TEST(NAME, sub) {
	auto t1 = K();
	auto t2 = K();
	auto t3 = K();

	for (uint32_t a = 0; a < K::size() - 1; ++a) {
		for (uint32_t b = a+1; b < K::size(); ++b) {
			for (size_t j = 0; j < NR_TESTS; ++j) {
				t3.zero();
				t1.random();
				t2.random();

				K::sub(t3, t1, t2, a, b);
				for (uint32_t i = a; i < b; i++) {
					EXPECT_EQ(t3.get(i), (t1.get(i) + PRIME - t2.get(i)) % PRIME);
				}

				for (uint32_t i = 0; i < a; i++) {
					EXPECT_EQ(t3.get(i), 0);
				}

				for (uint32_t i = b; i < K::size(); i++) {
					EXPECT_EQ(t3.get(i), 0);
				}
			}
		}
	}
}

TEST(NAME, mul) {
	auto t1 = K();
	auto t2 = K();
	auto t3 = K();

	for (uint32_t a = 0; a < K::size() - 1; ++a) {
		for (uint32_t b = a+1; b < K::size(); ++b) {
			for (size_t j = 0; j < NR_TESTS; ++j) {
				t3.zero();
				t1.random();
				t2.random();

				K::mul(t3, t1, t2, a, b);
				for (uint32_t i = a; i < b; i++) {
					EXPECT_EQ(t3.get(i), (t1.get(i) * t2.get(i)) % PRIME);
				}

				for (uint32_t i = 0; i < a; i++) {
					EXPECT_EQ(t3.get(i), 0);
				}

				for (uint32_t i = b; i < K::size(); i++) {
					EXPECT_EQ(t3.get(i), 0);
				}
			}
		}
	}
}
