#include <gtest/gtest.h>
#include <cstdint>

// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#define SSLWE_CONFIG_SET
#define G_k                     0
#define G_l                     0
#define G_d                     3                   // Depth of the search Tree
#define G_n                     20u
#define LOG_Q                   10u
#define G_q                     (1u << LOG_Q)
#define G_w                     (G_n * 3 / 8)       // NTRU encrypt
#define SORT_INCREASING_ORDER
#define VALUE_KARY
static  std::vector<uint64_t>   __level_translation_array{{0, 2, 4, 6, 8, G_n}};
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{0,0,0}} }};

#include "value.h"
#include "element.h"
#include "matrix.h"
#include "test.h"

#include "m4ri/m4ri.h"
#include "fplll/nr/matrix.h"
#include "glue_fplll.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

const unsigned int nr = G_n;
const unsigned int nc = G_n;

// does what the names says
static inline void clear_vector(word *row, const uint64_t limbs) {
	for (int i = 0; i < limbs; ++i) {
		row[i] = 0;
	}
}

static inline word sum_up_row(word *row, const uint64_t limbs) {
	word ret = 0;
	for (int i = 0; i < limbs; ++i) {
		ret ^= row[i];
	}
	return ret;
}

template<class Label>
static inline bool is_zero(Label &l) {
	for (int i = 0; i < l.size(); ++i) {
		if (l.data()[i] != 0)
			return false;
	}
	return true;
}

//  res = row * A^T(i, *), where A^T is already the input.
static inline uint64_t mult_rows_transposed(const word *row, mzd_t *AT, const uint64_t i) {
	word res = 0;

	for (int j = 0; j < AT->width; ++j) {
		res ^= row[j] & AT->rows[i][j];
	}
	return __builtin_popcountll(res)%2;
}

// mutliplies two rows
// res = row * A^T(i, *)
static inline uint64_t mult_rows(const word *row, mzd_t *A, const uint64_t i) {
	mzd_t *AT = mzd_init(A->ncols, A->nrows);
	mzd_transpose(AT, A);
	word res = mult_rows_transposed(row, AT, i);
	mzd_free(AT);
	return res;
}


TEST(hybrid_split, compile) {
	const uint64_t l = (G_n/2)+1;
	const uint64_t g = (G_n/2)+1;
    const uint64_t r= 5;

	fplll::ZZ_mat<kAryType> A, Al, Ag;
	A.resize(G_n, G_n);
	for (int i = 0; i < G_n; ++i) {
		for (int j = 0; j < G_n; ++j) {
			A(i,j).get_data()=i+j;
		}
	}

	hybrid_split<kAryType>(Al, Ag, A, l, g,r);

	for (int i = 0; i < A.r; ++i) {
		for (int j = r; j < A.c; ++j) {
			if (i < l)
				EXPECT_EQ(A(i, j), Al(i, j-r));

			if (i >= (G_n-g))
				EXPECT_EQ(A(i, j), Ag(i-(G_n-g), j-r));
		}
	}

	// std::cout << A << "\n";
	// std::cout << Al << "\n";

}

TEST(createBasis, compile) {
    const uint64_t l = (G_n/2)+3;
    const uint64_t g = (G_n/2)+3;
    const uint64_t r= 5;

    fplll::ZZ_mat<kAryType> A, Al, Ag;
    fplll::ZZ_mat<mpz_t> B;
    A.resize(G_n, G_n);
    for (int i = 0; i < G_n; ++i) {
        for (int j = 0; j < G_n; ++j) {
            A(i,j).get_data()=i+j;
        }
    }

    hybrid_split<kAryType>(Al, Ag, A, l, g,r);
    create_hybrid_basis(B,Al,r);
    for (int i = 0; i < B.r-1; ++i) {
        for (int j = 0; j < B.c-1; ++j) {
            if (i==j && i<G_n-r)
                EXPECT_EQ(B(i, j), G_q);
            else if (i==j && i>=G_n-r)
                EXPECT_EQ(B(i, j), 1);
            else if (i>=G_n-r && j < (G_n-r))
                EXPECT_EQ(B(i, j), Al(i - (G_n - r), j).get_data().data());
            else
                EXPECT_EQ(B(i,j),0);
        }
    }

}



/*                                  NEW ERA OF MATRIX IMPL                          */
//TODO for all test; edge cases, asymmetric size, wrong input output size.
TEST(new_vector_matrix_product, symmetric_fplll_k_ary_one_simple) {
	fplll::ZZ_mat<kAryType> mm(G_n, G_n);
	mm.fill(1);

	Matrix_T<fplll::ZZ_mat<kAryType>> m(mm);
	Label l{};
	Value v{};

	for (int i = 0; i < v.size(); ++i) {
		v.data()[i] = 1;
	}

	l.zero();

	new_vector_matrix_product<Label, Value, fplll::ZZ_mat<kAryType>>(l, v, m);
	for (int i = 0; i < l.size(); ++i) {
		EXPECT_NE(l[i],0);
	}
	for (int i = 0; i < v.size(); ++i) {
		EXPECT_EQ(v[i],1);
	}
	for (int i = 0; i < m.get_rows(); ++i) {
		for (int j = 0; j < m.get_cols(); ++j) {
			EXPECT_EQ(m.value(i, j),1);
		}
	}
}

TEST(new_vector_matrix_product, symmetric_mzd_one_simple) {
	mzd_t *mm = mzd_init(G_n, G_n);

	Matrix_T<mzd_t *> m(mm);
	m.fill(1);
	
	using BinaryLabel = Label_T<BinaryContainer<G_n>>;
	using BinaryValue = Value_T<BinaryContainer<G_n>>;

	BinaryLabel l{};
	BinaryValue v{};

	for (int i = 0; i < v.size(); ++i) {
		v.data()[i] = true;
	}

	l.zero();

	new_vector_matrix_product<BinaryLabel, BinaryValue, mzd_t *>(l, v, m);
	EXPECT_NE(sum_up_row(l.data().data().data(), l.data().limbs()),0);
	for (int i = 0; i < v.size(); ++i) {
		EXPECT_EQ(v[i],1);
	}
	for (int i = 0; i < m.get_rows(); ++i) {
		for (int j = 0; j < m.get_cols(); ++j) {
			EXPECT_EQ(m.value(i, j),1);
		}
	}

	mzd_free(mm);
}

TEST(new_vector_matrix_product, symmetric_mzd_random) {
	auto bit_random = []() {
		uint64_t a;
        fastrandombytes(&a, 8);
		return a%2;
	};

	const uint32_t rows = G_n;
	const uint32_t columns = G_n + 10;

	mzd_t *mm_T = mzd_init(columns, rows);
	mzd_t *mm_2 = mzd_init(rows, columns);
	mzd_t *mm_2_T;

	mzd_t *mm_l = mzd_init(1, columns);
	mzd_t *mm_v = mzd_init(1, rows);

	// set the debugging matrix everywhere on random()
	for (int i = 0; i < mm_2->nrows; ++i) {
		for (int j = 0; j < mm_2->ncols; ++j) {
			mzd_write_bit(mm_2, i, j, bit_random());
		}
	}
	for (int j = 0; j < mm_2->ncols; ++j) {
		mzd_write_bit(mm_l, 0, j, 0);
		mzd_write_bit(mm_v, 0, j,  bit_random());
	}

	mm_2_T = mzd_transpose(NULL, mm_2);
	mzd_copy(mm_T, mm_2_T);
	Matrix_T<mzd_t *> m_T(mm_T);

	using BinaryLabel = Label_T<BinaryContainer<columns>>;
	using BinaryValue = Value_T<BinaryContainer<rows>>;

	BinaryLabel l{};
	BinaryValue v{};
	l.zero();

	for (int i = 0; i < v.size(); ++i) {
		v.data()[i] = mzd_read_bit(mm_v, 0, i);
	}

	//_mzd_mul_naive(mm_l, mm_v, mm_2_T, 0);
	mzd_mul_naive(mm_l, mm_v, mm_2);
	new_vector_matrix_product<BinaryLabel, BinaryValue, mzd_t *>(l, v, m_T);

	for (int i = 0; i < v.size(); ++i) {
		EXPECT_EQ(v[i],mzd_read_bit(mm_v, 0, i));
	}

	EXPECT_EQ(m_T.get_rows(), columns);
	EXPECT_EQ(m_T.get_cols(), rows);
	EXPECT_EQ(mm_l->ncols,  l.size());


	for (int i = 0; i < m_T.get_rows(); ++i) {
		for (int j = 0; j < m_T.get_cols(); ++j) {
			EXPECT_EQ(m_T.value(i, j),mzd_read_bit(mm_2_T, i, j));
		}
	}

	for (int i = 0; i < v.size(); ++i) {
		EXPECT_EQ(mzd_read_bit(mm_v, 0, i),v.data()[i]);
	}

	for (int i = 0; i < l.size(); ++i) {
		EXPECT_EQ(mzd_read_bit(mm_l, 0, i),l.data()[i]);
	}

	mzd_free(mm_2_T);
	mzd_free(mm_l);
	mzd_free(mm_v);
	mzd_free(mm_2);
	mzd_free(mm_T);

}

TEST(new_vector_matrix_product, asymmetric_fplll_random) {
	const uint32_t rows = G_n;
	const uint32_t columns = G_n + 10;

	auto bit_random = []() {
		return rand()%G_q;
	};

	using T = kAryType;
	using Label = Label_T<kAryContainer_T<kAryType, columns>>;
	using Value = Value_T<kAryContainer_T<kAryType, rows>>;

	fplll::Matrix<T> mm_2_T(columns, rows);
	fplll::NumVect<T> mm_2_v(1, rows);
	fplll::NumVect<T> mm_2_l(1, columns);

	Label l{};
	Value v{};

	for (int i = 0; i < columns; ++i) {
		for (int j = 0; j < rows; ++j) {
			mm_2_T(i, j) = bit_random();
		}
	}
	for (int i = 0; i < rows; ++i) {
		mm_2_v[i] = bit_random();
	}
	for (int i = 0; i < columns; ++i) {
		mm_2_l[i] = 0;
	}

	ZZ_mat<kAryType> mmm_2_T(columns, rows);
	for (int i = 0; i < columns; ++i) {
		for (int j = 0; j < rows; ++j) {
			//TODO does not work mmm_2_T(i, j) = mm_2_T(i, j);
		}
	}

	Matrix_T<ZZ_mat<T>> m_T(mmm_2_T);

	vector_matrix_product<T>(mm_2_l, mm_2_l, mm_2_T);
	new_vector_matrix_product<Label, Value, ZZ_mat<T>>(l, v, m_T);

	for (int i = 0; i < m_T.get_rows(); ++i) {
		for (int j = 0; j < m_T.get_cols(); ++j) {
			EXPECT_EQ(m_T.value(i, j),mmm_2_T(i, j));
		}
	}

	for (int i = 0; i < v.size(); ++i) {
		EXPECT_EQ(mm_2_v[i],v.data()[i]);
	}

	for (int i = 0; i < l.size(); ++i) {
		EXPECT_EQ(mm_2_l[i],l.data()[i]);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand() * time(NULL));
	return RUN_ALL_TESTS();
}
