/// if this test fails, something really bad is going on
TEST(ListName, simple) {
	List L{LS, 1};
}

TEST(ListName, copy) {
	List L{LS, 1}, L2{LS, 1};

	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.is_correct(m), true);

	L2 = L;
	EXPECT_EQ(L2.load(), LS);
	EXPECT_EQ(L2.size(), LS);
	EXPECT_EQ(L2.is_correct(m), true);
}

TEST(ListName, base_random) {
	List L{LS, 1};
	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < std::min((uint32_t)K, (uint32_t)N); ++j) {
			EXPECT_EQ(L[i].get_label(j) == L[i].get_value(j), true);
		}
	}
}

TEST(ListName, sort_level) {
	const uint32_t k_lower = 0, k_higher = N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(k_lower, k_higher);

	ASSERT(L.is_sorted(k_lower, k_higher));
	EXPECT_EQ(L.is_correct(m), true);
}

TEST(ListName, sort_level_constexpr) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.template sort_level<0, N>();
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_correct(m), true);
}


TEST(ListName, sort_level_target) {
	const uint32_t k_lower = 0, k_higher=N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	Element rnd;
	rnd.random();

	L.sort_level(0, N, rnd.label);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(rnd.label, false, k_lower, k_higher), true);
}

TEST(ListName, sort_level_target_constexpr) {
	const uint32_t k_lower = 0, k_higher=N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	Element rnd;
	rnd.random();

	L.template sort_level<0, N>(rnd.label);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(rnd.label, false, k_lower, k_higher), true);
}

TEST(ListName, search_level) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.search_level(L[i], 0, N);
		EXPECT_EQ(pos, i);
	}
}

TEST(ListName, linear_search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.linear_search(L[i]);
		EXPECT_EQ(pos, i);
	}
}

TEST(ListName, binary_search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.binary_search(L[i]);
		EXPECT_EQ(pos, i);
	}
}

TEST(ListName, interpolation_search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	if constexpr (N < 64) {
		for (size_t i = 0; i < LS; ++i) {
			const size_t pos = L.template interpolation_search<0, N>(L[i]);
			EXPECT_EQ(pos, i);
		}
	}
}

TEST(ListName, lreal_search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos1 = L.linear_search(L[i]);
		const size_t pos2 = L.binary_search(L[i]);
		const size_t pos3 = L.template interpolation_search<0, N>(L[i]);
		ASSERT_EQ(pos1, i);
		ASSERT_EQ(pos2, i);
		ASSERT_EQ(pos3, i);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos1 = L.template linear_search<0, N>(L[i]);
		const size_t pos2 = L.template binary_search<0, N>(L[i]);
		ASSERT_EQ(pos1, i);
		ASSERT_EQ(pos2, i);
	}
}

TEST(ListName, search_boundaries) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, N);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const auto pos = L.search_boundaries(L[i], 0, N);
		EXPECT_EQ(pos.first, i);
		EXPECT_EQ(pos.second, i + 1);
	}
}


TEST(ListName, search_boundaries_constexpr) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level<0, N>();
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(0, N), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, N), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const auto pos = L.template search_boundaries<0, N>(L[i]);
		EXPECT_EQ(pos.first, i);
		EXPECT_EQ(pos.second, i + 1);
	}
}
