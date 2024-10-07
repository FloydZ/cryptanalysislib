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
		t.set(0, i);
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
	constexpr uint32_t qbits = ceil_log2(PRIME);
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


TEST(NAME, HashSimple) {
	K b1;
	constexpr uint32_t qbits = ceil_log2(PRIME);
	constexpr uint32_t limit = 64;
	for (uint32_t l = 0; l < n-1u; ++l) {
		for (uint32_t h = l+1u; h < n; ++h) {
			const uint64_t size = (h - l) * qbits;
			if (size > limit) { continue; }

			if ((cryptanalysislib::popcount::popcount(5) == 1)) {
				b1.zero();
				b1.minus_one(l, h);
				const uint64_t t = b1.hash(l, h);
				const uint64_t mask = ((h - l) * qbits) == 64ull ? -1ull : (1ull << ((h - l) * qbits)) - 1ull;
				EXPECT_EQ(t, mask);
			}


			b1.zero();
			b1.one(l, h);
			uint64_t t2 = b1.hash(l, h);
			const uint64_t qmask = (1ull<<qbits) - 1ull;
			for (uint32_t i = 0; i < h-l; ++i) {
				const uint64_t c = t2 & qmask;
				EXPECT_EQ(c, 1ul);

				t2 >>= qbits;
			}

			for (size_t k = 0; k < NR_TESTS; ++k) {
				b1.zero();
				K::DataType r = rng() % PRIME;

				for (uint32_t i = l; i < h; ++i) {
					b1.set(r, i);
				}

				uint64_t t3 = b1.hash(l, h);
				for (uint32_t i = 0; i < h - l; ++i) {
					const uint64_t c = t3 & qmask;
					EXPECT_EQ(c, r);

					t2 >>= qbits;
				}
			}

			for (size_t k = 0; k < NR_TESTS; ++k) {
				b1.random(l, h);

				uint64_t t3 = b1.hash(l, h);
				for (uint32_t i = 0; i < h - l; ++i) {
					const uint64_t c = t3 & qmask;
					EXPECT_EQ(c, b1.get(i+l));

					t3 >>= qbits;
				}
			}
		}
	}
}

TEST(NAME, HashCompare) {
	FqPackedVector<n, PRIME, T> b1,b2;
	constexpr uint32_t qbits = ceil_log2(PRIME);
	constexpr uint64_t limit = 63;
	const uint64_t qmask = (1ull<<qbits) - 1ull;

	auto compute_limit = [](const uint32_t t){
	  uint64_t p = PRIME;
	  uint64_t ret = p;

	  for (uint32_t i = 1; i < t; ++i) {
		  p *= PRIME;
		  ret += p;
	  }
	  return ret;
	};


	for (uint32_t l = 0; l < n-1u; ++l) {
		for (uint32_t h = l + 1u; h < n; ++h) {
			const uint64_t size1 = (h - l) * qbits;
			const uint64_t size2 = compute_limit(h-l);
			if (size1 > limit) { continue; }

			for (size_t i = 0; i < 1; ++i) {
				b1.zero();
				b2.zero();
				b1.random(l, h);
				b2.random(l, h);
				const uint64_t v1 = b1.hash(l, h);
				const uint64_t v2 = b2.hash(l, h);

				uint64_t t1 = v1;
				uint64_t t2 = v2;
				for (uint32_t o = 0; o < (h-l); ++o) {
					const uint64_t c1 = t1 & qmask;
					const uint64_t c2 = t2 & qmask;
					EXPECT_EQ(c1, b1.get(o+l));
					EXPECT_EQ(c2, b2.get(o+l));

					t1 >>= qbits;
					t2 >>= qbits;
				}

				// EXPECT_LT(v1, size2);
				// EXPECT_LT(v2, size2);

				// const size_t v3 = b1.hash(l, h);
				if (b1.is_lower(b2, l, h)) {
					EXPECT_LT(v1, v2);
				}

				if (b1.is_greater(b2, l, h)) {
					EXPECT_GT(v1, v2);
				}
			}
		}
	}
}
