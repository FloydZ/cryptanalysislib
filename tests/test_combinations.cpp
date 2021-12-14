#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <array>

#include "m4ri/m4ri.h"
#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint64_t kk = 3;
constexpr uint64_t e1 = 2;
constexpr uint64_t em1 = 2;

TEST(CombinationsIndex2, Left) {
	constexpr uint64_t N = 105;
	constexpr uint64_t T = 2;
	constexpr uint64_t size = bc(N,T);
	uint64_t loops = 0;
	uint16_t *P = (uint16_t *)malloc(T * sizeof(uint16_t));

	CombinationsIndex c(P, N, T, 0);

	do {
		loops += 1;
		/*for (int i = 0; i < T; ++i) {
			std::cout << P[i] << " ";
		}
		std::cout << "\n";*/
	}while(c.next(P));

	free(P);
	//TODO not finished implementing EXPECT_EQ(loops, size);

}

TEST(Combinations_Chase_BinaryContainer_Compare_Chase2, Left) {
	using TestBinaryContainer = BinaryContainer<n>;
	uint64_t nperm = 0;
	Element e, e_tmp;
	TestBinaryContainer e2, e2_tmp;
	e.zero(); e_tmp.zero(); e2.zero(); e2_tmp.zero();
	uint64_t pos1 = 0, pos2 = 0;

	Combinations_Chase<decltype(e.get_value().data().get_type())> c{n, kk};
	c.left_init(e.get_value().data().data().data());
	c.left_step(e.get_value().data().data().data(), true);
	uint64_t rt = 1;

	Combinations_Chase2<TestBinaryContainer> cc{n, kk, 0};
	for (int j = n - kk; j < n; ++j) {
		e2[j] = true;
	}

	while(rt != 0) {
		e_tmp = e;

		rt = c.left_step(e.get_value().data().data().data());

		nperm += 1;
		EXPECT_EQ(false, e_tmp.get_value().is_equal(e.get_value(), 0, n));
	}
	EXPECT_EQ(nperm, bc(n, kk));

	uint64_t nperm2 = 0;
	rt = 1;
	while(rt != 0) {
		e2_tmp = e2;
		rt = cc.next(e2, &pos1, &pos2);

		nperm2 += 1;
		EXPECT_EQ(false, e2_tmp.is_equal(e2, 0, n));
	}
	EXPECT_EQ(nperm2, bc(n, kk));
	EXPECT_EQ(nperm, nperm2);

}

TEST(Combinations_Chase_VV_Binary2, Left) {
	constexpr uint64_t N = 105;
	constexpr uint64_t T = 2;
	constexpr uint64_t size = bc(N,T);
	uint64_t loops = 0;
	uint16_t *P = (uint16_t *)malloc(T * sizeof(uint16_t));

	Combinations_Chase_VV_Binary c(N, T, 0);

	while(c.next(P)) {
		loops += 1;
		/*for (int i = 0; i < T; ++i) {
			std::cout << P[i] << " ";
		}
		std::cout << "\n";*/
	}

	free(P);
	EXPECT_EQ(loops, size-1);

}

TEST(Combinations_Chase_VV_Binary, simple) {
	constexpr uint64_t N = 20;
	constexpr uint64_t T = 3;
	constexpr uint64_t size = bc(N,T);
	uint64_t loops = 0;

	uint16_t *comb = (uint16_t *)malloc(size * T * sizeof(uint16_t));
	uint16_t *diff = (uint16_t *)malloc(size * sizeof(uint16_t));
	Combinations_Chase_VV_Binary::chase2(N, T, comb, diff);
	for(uint32_t i = 0; i < size; i++) {
		loops += 1;

		/*for (int j = 0; j < T; ++j) {
			std::cout << comb[i*T +j] << " ";
		}
		std::cout << ", " << diff[i] << "\n";*/
	}

	free(diff);
	free(comb);
	EXPECT_EQ(loops, size);

}

TEST(Combinations_Chase2, simple) {
	const uint64_t N = n;
	const uint64_t T = 4;

	uint64_t pos1 = 0, pos2 = 0;

	BinaryContainer<n> container, tmp;
	container.zero(); tmp.zero();

	for (int j = N-T; j < N; ++j) {
		container[j] = true;
	}

	uint64_t nperm = 0;
	Combinations_Chase2<BinaryContainer<n>> cc{N, T, 0};
	do {
		nperm += 1;
		EXPECT_EQ(false, container.is_equal(tmp, 0, n));
		EXPECT_LE(pos1, n);
		EXPECT_LE(pos2, n);

		tmp = container;
	} while (cc.next(container, &pos1, &pos2) != 0);

	EXPECT_EQ(nperm, bc(N, T));

}

TEST(Combinations_Chase_T_u64, Left) {
	uint64_t nperm = 0;
	uint64_t data[n] = {0};
	uint64_t data2[n] = {0};

	Combinations_Chase<uint64_t> c{n, kk, 0};
	c.left_init(data);
	uint64_t rt = c.left_step(data, true);

	while(rt != 0) {

		rt = c.left_step(data);
		nperm += 1;
		EXPECT_NE(0, memcmp(data2, data, n * sizeof(uint64_t)));
		memcpy(data2, data, n * sizeof(uint64_t));
	}

	EXPECT_EQ(nperm, bc(n, kk));
}

TEST(Combinations_Chase_M4RI, generate_diff_list) {
	Combinations_Chase_M4RI c{n, kk};
	std::vector<std::pair<uint64_t, uint64_t>> v;
	c.generate_diff_list(v);

	// first check that they all differ from each other.
	for (int i = 1; i < v.size(); ++i) {
		EXPECT_EQ(false, (v[i].first == v[i-1].first) && (v[i].second == v[i-1].second));
	}

	// check size.
	EXPECT_EQ(v.size(), bc(n, kk));
}

TEST(Combinations_Chase_M4RI, Left) {
	uint64_t nperm = 0;
	mzd_t *p = mzd_init(1, n);
	mzd_t *p_tmp = mzd_init(1, n);

	Combinations_Chase_M4RI c{n, kk};
	c.left_init(p);
	uint64_t rt = c.left_step(p, true);

	while(rt != 0) {
		// little bit of debugging help
		//print_matrix("p", p);

		mzd_copy(p_tmp, p);

		rt = c.left_step(p);
		nperm += 1;
		EXPECT_NE(0, mzd_cmp(p_tmp, p));

	}


	EXPECT_EQ(nperm, bc(n, kk));
	mzd_free(p);
	mzd_free(p_tmp);
}

TEST(Combinations_Chase_M4RI, Left_Start) {
	uint64_t nperm = 0;
	mzd_t *p = mzd_init(1, n);
	mzd_t *p_tmp = mzd_init(1, n);

	const uint64_t start = 1;
	Combinations_Chase_M4RI c{n, kk, start};
	c.left_init(p);
	uint64_t rt = c.left_step(p, true);

	while(rt != 0) {
		// little bit of debugging help
		// print_matrix("p", p);

		mzd_copy(p_tmp, p);

		rt = c.left_step(p);
		nperm += 1;
		EXPECT_NE(0, mzd_cmp(p_tmp, p));
	}


	EXPECT_EQ(nperm, bc(n-start, kk));
	mzd_free(p);
	mzd_free(p_tmp);
}

TEST(Combinations_Chase_Element, Left) {
	uint64_t nperm = 0;
	Element e, e_tmp; e.zero(); e_tmp.zero();

	Combinations_Chase<decltype(e.get_value().data().get_type())> c{n, kk};
	c.left_init(e.get_value().data().data().data());
	c.left_step(e.get_value().data().data().data(), true);
	uint64_t rt = 1;

	while(rt != 0) {
		// little bit of debugging help
		// std::cout << e;

		e_tmp = e;

		rt = c.left_step(e.get_value().data().data().data());

		nperm += 1;
		EXPECT_NE(0, e_tmp.is_equal(e, 0, n));
	}

	EXPECT_EQ(nperm, bc(n, kk));
}

TEST(Combinations_Chase_Binary_BinaryContainer, Left) {
	uint64_t nperm = 0;
	using BinaryContainer2 = BinaryContainer<n>;
	BinaryContainer2 e1, e2; e1.zero(); e2.zero();
	Combinations_Chase_Binary c{n, kk};
	c.left_init(e1.data().data());
	c.left_step(e1.data().data(), true);
	uint64_t rt = 1;

	while(rt != 0) {
		// little bit of debugging help
		// std::cout << e1 << "\n";
		rt = c.left_step(e1.data().data());

		nperm += 1;
		EXPECT_EQ(false, e2.is_equal(e1, 0, n));
		e2 = e1;
	}

	EXPECT_EQ(nperm, bc(n, kk));
}

TEST(Combinations_Chase_T_u64, diff) {
	uint64_t nperm = 0;
	uint64_t data[n] = {0};
	uint64_t data2[n] = {0};
	uint32_t pos1 = 0, pos2 = 0;

	Combinations_Chase<uint64_t> c{n, kk, 0};
	c.left_init(data);
	uint64_t rt = c.left_step(data, true);
	memcpy(data2, data, n * sizeof(uint64_t));

	while(rt != 0) {
		rt = c.left_step(data);
		nperm += 1;
		EXPECT_NE(0, memcmp(data2, data, n * sizeof(uint64_t)));
		Combinations_Chase<uint64_t>::diff(data, data2, n, &pos1, &pos2);

		memcpy(data2, data, n * sizeof(uint64_t));
	}

	EXPECT_EQ(nperm, bc(n, kk));
}

// test: T = int64_t
TEST(Combinations_Chase_Ternary_T_64, Left) {
	uint64_t nperm = 0;
	const uint64_t start = 0;
	int64_t data[n] = {0};
	int64_t data2[n] = {0};

	Combinations_Chase_Ternary<int64_t> c{n, e1, em1, start};
	c.left_init(data);
	uint64_t rt = 1;

	while(rt != 0) {
		rt = c.left_step(data);

		nperm += 1;
		EXPECT_NE(0, memcmp(data2, data, n * sizeof(uint64_t)));
		memcpy(data2, data, n * sizeof(int64_t));
	}

	EXPECT_EQ(nperm, bc(n-start, e1)*bc(n-e1-start, em1));
}


TEST(Combinations_TernaryRow, left) {
	uint64_t nperm = 0;
	constexpr uint32_t nn = 6, e1 = 2, e2 = 2;
	using Row = kAryPackedContainer_T<uint64_t, 3, nn>;
	using ChangeList = std::pair<uint16_t, uint16_t>;

	Combinations_Chase_TernaryRow<Row, ChangeList> c{nn, e1, e2};
	Row data, data2; data.zero(); data2.zero();
	ChangeList cl;
	int64_t oot = 0;
	c.left_init(data);

	do {
		std::cout << data << ", " << cl.first << ":" << cl.second << ", " <<  (oot > 0 ? "two changed" : "one changed") << "\n";
		nperm += 1;
		data2 = data;
		oot = c.left_step(data, cl);
	} while(oot != 0);

	EXPECT_EQ(nperm, bc(nn, e1)*bc(nn-e1, e2));
}

TEST(Combinations_TernaryRow, leftsingle) {
	// only iterates single symbol.
	uint64_t nperm = 0;
	using Row = kAryPackedContainer_T<uint64_t, 3, 10>;
	using ChangeList = std::pair<uint16_t, uint16_t>;

	Combinations_Chase_TernaryRow<Row, ChangeList> c{10, e1, em1};
	Row data, data2; data.zero(); data2.zero();
	ChangeList cl;
	int64_t oot = 0;
	c.left_single_init(data, 2);

	do {
		// std::cout << data << ", " << cl.first << ":" << cl.second << ", " <<  (oot > 0 ? "two changed" : "one changed") << " " << nperm << "\n";
		nperm += 1;
		data2 = data;
		oot = c.left_single_step(data, cl);
	} while(oot != 0);

	EXPECT_EQ(nperm, bc(10, e1));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}