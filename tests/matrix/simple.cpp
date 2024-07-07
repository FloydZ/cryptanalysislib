#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "permutation/permutation.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using T = uint64_t;
constexpr uint32_t q = 3;
constexpr uint32_t nrows = 110;
constexpr uint32_t ncols = 120;

constexpr bool packed = true;
using M = FqMatrix<T, nrows, ncols, q, packed>;
using MT = FqMatrix<T, ncols, nrows, q, packed>;


TEST(FqMatrix, Init) {
	M m = M{};
	(void) m;
}

TEST(FqMatrix, SubScription) {
	M m = M{};
	m.random();

	for (uint32_t i = 0; i < M::ROWS; ++i) {
		for (uint32_t j = 0; j < M::COLS; ++j) {
			ASSERT_EQ(m[i][j], m.get(i, j));
		}
	}
}

TEST(FqMatrix, random) {
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

TEST(FqMatrix, add) {
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


	m1.random();
	m2.random();
	m3.random();
	M::add(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), (m1.get(i, j) + m2.get(i, j)) % q);
		}
	}
}

TEST(FqMatrix, sub) {
	M m1 = M{}, m2 = M{}, m3 = M{};
	m1.random();
	m2.zero();
	m3.random();


	m1.random();
	m2.random();
	m3.random();
	M::sub(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), (m1.get(i, j) - m2.get(i, j) + q) % q);
		}
	}
}

TEST(FqMatrix, InitFromString) {
	char *ptr = (char *) malloc(nrows * ncols + 1);
	EXPECT_NE(ptr, nullptr);
	using T = int;
	T *ptr2 = (T *) malloc(nrows * ncols * sizeof(T));
	EXPECT_NE(ptr2, nullptr);

	for (uint32_t i = 0; i < nrows * ncols; ++i) {
		const int a = fastrandombytes_uint64() % q;
		ptr2[i] = a;
		sprintf(ptr + i, "%d", a);
	}

	M m = M{ptr};

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			ASSERT_EQ(m.get(i, j), ptr2[i * ncols + j]);
		}
	}

	free(ptr);
	;
	free(ptr2);
}

TEST(FqMatrix, SimpleTranspose) {
	constexpr_for<400, 430, 3>([](const auto __nrows) {
		constexpr_for<400, 480, 3>([__nrows](const auto __ncols) {
			using M = FqMatrix<T, __nrows, __ncols, q, true>;
			using MT = FqMatrix<T, __ncols, __nrows, q, true>;
			M m = M{};
			MT mt = MT{};
			m.random();
			mt.random();

			M::transpose(mt, m);
			for (uint32_t i = 0; i < __nrows; ++i) {
				for (uint32_t j = 0; j < __ncols; ++j) {
					EXPECT_EQ(m.get(i, j), mt.get(j, i));
				}
			}
		});
	});
}

TEST(FqMatrix, Transpose) {
	constexpr_for<400, 430, 3>([](const auto __nrows) {
		constexpr_for<400, 470, 3>([__nrows](const auto __ncols) {
			using M = FqMatrix<T, __nrows, __ncols, q, true>;
			using MT = FqMatrix<T, __ncols, __nrows, q, true>;
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
		});
	});
}

TEST(FqMatrix, sub_Transpose_extended) {
	constexpr uint32_t srow = 5;
	constexpr uint32_t scol = 10;
	constexpr uint32_t erow = 95;
	constexpr uint32_t ecol = 90;
	using MT = FqMatrix<T, ecol - scol, erow - srow, q, packed>;
	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();


	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_transpose(mt, m, srow, scol, erow, ecol);

	// testing the transpose
	for (uint32_t i = srow; i < erow; ++i) {
		for (uint32_t j = scol; j < ecol; ++j) {
			const auto a = m.get(i, j);
			const auto b = mt.get(j - scol, i - srow);
			EXPECT_EQ(a, b);
		}
	}
}

TEST(FqMatrix, subTranspose) {
	constexpr uint32_t srow = 10;
	constexpr uint32_t scol = 10;
	using MT = FqMatrix<T, ncols - srow, nrows - scol, q, packed>;

	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();


	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_transpose(mt, m, srow, scol);
	for (uint32_t i = srow; i < nrows; ++i) {
		for (uint32_t j = scol; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(j - srow, i - scol));
		}
	}
}

TEST(FqMatrix, subMatrix) {
	constexpr uint32_t srow = 10;
	constexpr uint32_t scol = 10;
	constexpr uint32_t erow = nrows;
	constexpr uint32_t ecol = ncols;
	using MT = FqMatrix<T, erow - srow, ecol - scol, q, packed>;

	M m = M{};
	MT mt = MT{};
	m.random();
	mt.random();

	EXPECT_GT(nrows, srow);
	EXPECT_GT(ncols, scol);

	M::sub_matrix(mt, m, srow, scol, erow, ecol);
	for (uint32_t i = srow; i < nrows; ++i) {
		for (uint32_t j = scol; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), mt.get(i - srow, j - scol));
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
			ASSERT_EQ(m.get(i, j), i == j);
		}
	}
}

// TODO disable because ARM CI pipline has not enough space
//TEST(FqMatrix, m4ri) {
//	constexpr_for<400, 430, 3>([](const auto __nrows) {
//		constexpr_for<410, 440, 3>([__nrows](const auto __ncols) {
//			if constexpr (__nrows <= __ncols) {
//				using M = FqMatrix<T, __nrows, __ncols, q, true>;
//				M m = M{};
//				m.random();
//				const uint32_t rank = m.m4ri();
//				ASSERT_GT(rank, __nrows - 20);
//
//				for (uint32_t i = 0; i < rank; ++i) {
//					for (uint32_t j = 0; j < rank; ++j) {
//						EXPECT_EQ((bool) m.get(i, j), i == j);
//					}
//				}
//			}
//		});
//	});
//}

// failed in der osx pipline
//TEST(FqMatrix, markov_gaus) {
//	constexpr_for<400, 430, 3>([](const auto __nrows) {
//		constexpr_for<400, 470, 3>([__nrows](const auto __ncols) {
//			if constexpr (__nrows <= __ncols) {
//				using M = FqMatrix<T, __nrows, __ncols, q, true>;
//				using MT = FqMatrix<T, __ncols, __nrows, q, true>;
//				constexpr uint32_t l = 10;
//				constexpr uint32_t c = 5;
//				M m = M{};
//				Permutation P(ncols);
//				uint32_t rank = 0;
//
//				while (true) {
//					m.random();
//					rank = m.gaus(__nrows - l);
//					ASSERT_GT(rank, 0);
//					if (rank >= __nrows - l) { break; }
//				}
//
//				for (uint32_t k = 0; k < 10; ++k) {
//					uint32_t rank2 = m.template markov_gaus<c, __nrows - l>(P);
//					EXPECT_EQ(rank, rank2);
//
//					for (uint32_t i = 0; i < __nrows; ++i) {
//						for (uint32_t j = 0; j < __nrows - l; ++j) {
//							EXPECT_EQ(m.get(i, j), i == j);
//						}
//					}
//				}
//			}
//		});
//	});
//}


//TEST(FqMatrix, fixgaus) {
//	constexpr_for<400, 430, 3>([](const auto __nrows) {
//		constexpr_for<400, 470, 3>([__nrows](const auto __ncols) {
//			if constexpr (__nrows <= ncols) {
//				using M = FqMatrix<T, __nrows, __ncols, q, true>;
//				using MT = FqMatrix<T, __ncols, __nrows, q, true>;
//				M m = M{};
//				m.random();
//
//				Permutation P{ncols};
//				const uint32_t rank = m.gaus();
//				const uint32_t rank2 = m.fix_gaus(P, rank, __nrows);
//				ASSERT_GT(rank, __nrows - 10);
//
//				for (uint32_t i = 0; i < __nrows; ++i) {
//					for (uint32_t j = 0; j < rank2; ++j) {
//						if (i == j) {
//							ASSERT_EQ(m.get(i, j), 1u);
//							continue;
//						}
//						ASSERT_EQ(m.get(i, j), 0u);
//					}
//				}
//			}
//		});
//	});
//}

// test is only valid on clang. Gcc has problems compiling it.
//TEST(FqMatrix, mult) {
//	constexpr_for<1, 100, 11>([](const auto __nrows) {
//		constexpr_for<2, 100, 11>([__nrows](const auto __ncols) {
//			constexpr_for<1, 100, 11>([__nrows, __ncols](const auto __ncols_prime) {
//				using AT = FqMatrix<T, __nrows, __ncols, q, true>;
//				using BT = FqMatrix<T, __ncols, __ncols_prime, q, true>;
//				using CT = FqMatrix<T, __nrows, __ncols_prime, q, true>;
//
//				AT a = AT{};
//				BT b = BT{};
//				CT c = CT{};
//				a.random();
//				b.random();
//				c.random();
//				AT::mul(c, a, b);
//
//				//for (uint32_t i = 0; i < __nrows; ++i) {
//				//	for (uint32_t j = 0; j < rank2; ++j) {
//				//		if (i == j) {
//				//			ASSERT_EQ(m.get(i, j), 1u);
//				//			continue;
//				//		}
//				//		ASSERT_EQ(m.get(i, j), 0u);
//				//	}
//				//}
//			});
//		});
//	});
//}


TEST(FqMatrix, permute) {
	M m = M{};
	MT mt = MT{};
	Permutation P(ncols);

	m.identity();
	mt.identity();
	m.permute_cols(mt, P);

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
	//random_seed(time(0));
	random_seed(0);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
