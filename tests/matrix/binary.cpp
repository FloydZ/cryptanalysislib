#include <gtest/gtest.h>
#include <iostream>

#include "matrix/binary_matrix.h"
#include "container/fq_vector.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using T = uint64_t;
constexpr uint32_t nrows = 64;
constexpr uint32_t ncols = 64;
using M  = FqMatrix<T, nrows, ncols, 2>;
using MT = FqMatrix<T, ncols, nrows, 2>;


TEST(BinaryMatrix, Init) {
	M m = M{};
}

TEST(BinaryMatrix, random) {
	M m = M{};
	m.random();

	bool atleast_one_not_zero = false;
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			if (m.get(i, j) > 0) {
				atleast_one_not_zero = true;
				goto finish;
			}
		}
	}

	finish:
	EXPECT_EQ(atleast_one_not_zero, true);
}

TEST(BinaryMatrix, zero) {
	M m = M{};
	m.random();
	m.zero();

	bool allzero = true;
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			if (m.get(i, j) > 0) {
				allzero = false;
				goto finish;
			}
		}
	}

	finish:
	EXPECT_EQ(allzero, true);
}

TEST(BinaryMatrix, add) {
	M m1 = M{}, m2 = M{}, m3 = M{};
	m1.random();
	m2.zero();
	m3.random();

	M::add(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), m1.get(i, j));
		}
	}


	m1.random(); m2.random(); m3.random();
	M::add(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), (m1.get(i, j) + m2.get(i, j)) % 2);
		}
	}
}

TEST(BinaryMatrix, sub) {
	M m1 = M{}, m2 = M{}, m3 = M{};
	m1.random();
	m2.zero();
	m3.random();

	M::sub(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), m1.get(i, j));
		}
	}

	m1.random(); m2.random(); m3.random();
	M::sub(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), (m1.get(i, j) - m2.get(i, j) + 2) % 2);
		}
	}
}

TEST(BinaryMatrix, InitFromString) {
	char *ptr = (char *)malloc(nrows * ncols + 1);
	EXPECT_NE(ptr, nullptr);
	using T = int;
	T *ptr2 = (T *)malloc(nrows * ncols * sizeof(T));
	EXPECT_NE(ptr2, nullptr);

	for (uint32_t i = 0; i < nrows*ncols; ++i) {
		const int a = fastrandombytes_uint64() % 2;
		ptr2[i] = a;
		sprintf(ptr + i, "%d", a);
	}

	M m = M(ptr);

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			ASSERT_EQ(m.get(i, j), ptr2[i*ncols + j]);
		}
	}

	free(ptr);
	free(ptr2);
}

TEST(BinaryMatrix, SimpleTranspose) {
	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();

	M::transpose(mt, m);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(j, i));
		}
	}
}

TEST(BinaryMatrix, Transpose) {
	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();

	const uint32_t srow = 5;
	const uint32_t scol = 10;

	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::transpose(mt, m, srow, scol);

	// testing the transpose
	for (uint32_t i = srow; i < nrows; ++i) {
		for (uint32_t j = scol; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(j, i));
		}
	}
}

TEST(BinaryMatrix, subMatrix) {
	constexpr uint32_t srow = 10;
	constexpr uint32_t scol = 10;
	constexpr uint32_t erow = nrows;
	constexpr uint32_t ecol = ncols;
	using MT = FqMatrix<T, erow-srow, ecol-scol, 2>;

	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();

	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_matrix(mt, m, srow, scol, erow, ecol);
	for (uint32_t i = srow; i < nrows; ++i) {
		for (uint32_t j = scol; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(i-srow, j-scol));
		}
	}
}

TEST(BinaryMatrix, gaus) {
	M m = M{};
	m.random();
	const uint32_t rank = m.gaus();
	ASSERT_GT(rank, 0);

	std::cout << rank << std::endl;
	m.print();

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < rank; ++j) {
			ASSERT_EQ(m.get(i, j), i==j);
		}
	}
}

TEST(BinaryMatrix, matrix_echelonize_partial_plusfix_opt) {
	M m = M{};
	MT mt = MT{};
	m.random();

	mzp_t *P = mzp_init(ncols);
	constexpr uint32_t l = 10;
	const uint32_t rank = m.template matrix_echelonize_partial_plusfix_opt
	        <ncols, nrows, l>
	                      (m, mt, 3, l, P);
	ASSERT_GT(rank, 0);

	std::cout << rank << std::endl;
	m.print();

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < rank; ++j) {
			ASSERT_EQ(m.get(i, j), i==j);
		}
	}
}

TEST(BinaryMatrix, fixgaus) {
	M m = M{};
	m.random();

	uint32_t perm[ncols]= {0};
	for (uint32_t i = 0; i < ncols; ++i) {
		perm[i] = i;
	}

	const uint32_t rank = m.gaus();
	const uint32_t rank2 = m.fix_gaus(perm, rank, nrows, ncols);
	ASSERT_GT(rank, 0);

	//std::cout << rank << std::endl;
	//m.print();

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < rank2; ++j) {
			if (i == j) {
				ASSERT_EQ(m.get(i, j), 1u);
				continue;
			}
			ASSERT_EQ(m.get(i, j), 0u);
		}
	}
}

TEST(BinaryMatrix, permute) {
	M m = M{};
	MT mt = MT{};
	uint32_t perms[ncols];
	for (uint32_t i = 0; i < ncols; ++i) {
		perms[i] = i;
	}

	m.identity();
	mt.identity();
	m.permute_cols(mt, perms, ncols);

	// m.print();
	// mt.print();
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			if (i == j) {
				EXPECT_EQ(m.get(i, j), mt.get(i, j));
				continue;
			}
		}
	}
}

int main(int argc, char **argv) {
	random_seed(time(0));
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
