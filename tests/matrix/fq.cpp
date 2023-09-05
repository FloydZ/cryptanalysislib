#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "matrix/matrix.h"
#include "container/fq_vector.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using T = uint64_t;
constexpr uint32_t q = 5;
constexpr uint32_t nrows = 100;
constexpr uint32_t ncols = 110;
using M  = FqMatrix<T, nrows, ncols, q>;
using MT = FqMatrix<T, ncols, nrows, q>;

TEST(FqMatrix, Init) {
	M m = M{};
}

TEST(FqMatrix, random) {
	M m = M{};
	m.random();

	bool atleast_one_not_zero = false;
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			if (m.get(i, j) > 0) {
				atleast_one_not_zero = true;
				goto finish;
			}
		}
	}

	finish:
	EXPECT_EQ(atleast_one_not_zero, true);
}

TEST(FqMatrix, zero) {
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

TEST(FqMatrix, InitFromString) {
	char *ptr = (char *)malloc(nrows * ncols + 1);
	EXPECT_NE(ptr, nullptr);
	using T = int;
	T *ptr2 = (T *)malloc(nrows * ncols * sizeof(T));
	EXPECT_NE(ptr2, nullptr);

	for (uint32_t i = 0; i < nrows*ncols; ++i) {
		const int a = fastrandombytes_uint64() % q;
		ptr2[i] = a;
		sprintf(ptr + i, "%d", a);
	}

	M m = M{ptr};

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			ASSERT_EQ(m.get(i, j), ptr2[i*ncols + j]);
		}
	}

	free(ptr);;
	free(ptr2);
}

TEST(FqMatrix, SimpleTranspose) {
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

TEST(FqMatrix, Transpose) {
	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();
	MT mt2 = MT{mt};

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

	// testing the non transposed part
	for (uint32_t i = 0; i < srow; ++i) {
		for (uint32_t j = 0; j < scol; ++j) {
			EXPECT_EQ(mt2.get(i, j), mt.get(i, j));
		}
	}
}

TEST(FqMatrix, sub_Transpose_extended) {
	constexpr uint32_t srow = 5;
	constexpr uint32_t scol = 10;
	constexpr uint32_t erow = 95;
	constexpr uint32_t ecol = 90;
	//using MT = FqMatrix<T, ncols-srow, nrows-scol, q>;
	//using MT = FqMatrix<T, ncols, nrows, q>;
	using MT = FqMatrix<T, ecol-scol, erow-srow, q>;
	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();
	MT mt2 = MT{mt};


	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_transpose(mt, m, srow, scol, erow, ecol);

	// testing the transpose
	for (uint32_t i = srow; i < erow; ++i) {
		for (uint32_t j = scol; j < ecol; ++j) {
			const auto a = m.get(i, j);
			const auto b = mt.get(j-scol, i-srow);
			EXPECT_EQ(a, b);
		}
	}
}

TEST(FqMatrix, subTranspose) {
	constexpr uint32_t srow = 10;
	constexpr uint32_t scol = 10;
	using MT = FqMatrix<T, ncols-srow, nrows-scol, q>;

	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();


	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_transpose(mt, m, srow, scol);
	for (uint32_t i = srow; i < nrows; ++i) {
		for (uint32_t j = scol; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(j-srow, i-scol));
		}
	}
}

TEST(FqMatrix, subMatrix) {
	constexpr uint32_t srow = 10;
	constexpr uint32_t scol = 10;
	constexpr uint32_t erow = nrows;
	constexpr uint32_t ecol = ncols;
	using MT = FqMatrix<T, erow-srow, ecol-scol, q>;

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

TEST(FqMatrix, gaus) {
	M m = M{};
	m.random();
	const uint32_t rank = m.gaus();
	ASSERT_GT(rank, 0);

	//std::cout << rank << std::endl;
	//m.print();

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < rank; ++j) {
			ASSERT_EQ(m.get(i, j), i==j);
		}
	}
}

TEST(FqMatrix, fixgaus) {
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

TEST(FqMatrix, inplace_matrix_vector_mult) {
	using V = FqMatrix<T, 1, ncols, q>;
	using VT = FqMatrix<T, ncols, 1, q>;
	V v = V{};
	VT vt = VT{};
	M m  = M{};

	m.random();
	v.set(0);
	m.matrix_vector_mul(v);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			ASSERT_EQ(m.get(i, j), 0u);
		}
	}

	m.random();
	m.matrix_col_vector_mul(vt);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			ASSERT_EQ(m.get(i, j), 0u);
		}
	}
}

TEST(FqMatrix, matrix_vector_mult) {
	using V = FqMatrix<T, 1, ncols, q>;
	using VT = FqMatrix<T, nrows, 1, q>;
	V v = V{};
	VT vt = VT{};
	M m = M{};

	m.random();
	v.set(0);
	vt.random();
	m.matrix_vector_mul(vt, v);
	for (uint32_t i = 0; i < nrows; ++i) {
		ASSERT_EQ(vt.get(i, 0u), 0u);
	}

	vt.set(0);
	v.random();
	m.matrix_col_vector_mul(vt, vt);
	for (uint32_t i = 0; i < nrows; ++i) {
		ASSERT_EQ(vt.get(i, 0), 0u);
	}
}

TEST(FqMatrix, matrix_matrix_mult) {
	constexpr uint32_t ncols_prime = 120;
	using Min = FqMatrix<T, ncols, ncols_prime, q>;
	using Mout = FqMatrix<T, nrows, ncols_prime, q>;

	M m = M{};
	Min m_in = Min{};
	Mout m_out = Mout{};

	m.random();
	m_out.random();
	m_in.zero();

	m.matrix_matrix_mul(m_out, m_in);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols_prime; ++j) {
			ASSERT_EQ(m_out.get(i, j), 0u);
		}
	}
}

TEST(FqMatrix, permute) {
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

TEST(FqMatrix, matrix_row_vector_mul2) {
	using ValueType = kAryContainer_T<uint8_t, ncols, q>;
	using LabelType = kAryContainer_T<uint8_t, nrows, q>;
	M m = M{};
	auto v = ValueType {};
	auto l = LabelType {};
	m.random();
	l.random();
	v.zero();

	m.template matrix_row_vector_mul2<LabelType, ValueType>(l, v);
	for (uint32_t i = 0; i < nrows; ++i) {
		EXPECT_EQ(l.get(i), 0);
	}

	l.zero();
	v.random();
	m.matrix_row_vector_mul2(l, v);
	for (uint32_t i = 0; i < nrows; ++i) {
		uint64_t sum = 0;
		for (uint32_t j = 0; j < ncols; ++j) {
			sum += v.get(j) * m.get(i, j);
		}

		sum = sum % q;
		EXPECT_EQ(l.get(i), sum);
	}
}


int main(int argc, char **argv) {
	random_seed(time(0));
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
