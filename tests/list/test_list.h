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
		for (uint32_t j = 0; j < std::min(K, N); ++j) {
			EXPECT_EQ(L[i].get_label(j) == L[i].get_value(j), true);
		}
	}
}

TEST(ListName, sort_level) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}
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

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.search_level(L[i], 0, 20);
		EXPECT_EQ(pos, i);
	}
}

TEST(ListName, search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.search(L[i]);
		EXPECT_EQ(pos, i);
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

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const auto pos = L.search_boundaries(L[i], 0, 20);
		EXPECT_EQ(pos.first, i);
		EXPECT_EQ(pos.second, i + 1);
	}
}
