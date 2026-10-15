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


TEST(FqMatrix, from_string) {
	// q <= 10
	{
		using A = FqMatrix<T, 2, 3, 3, packed>;
		const A a{"012210"};
		EXPECT_EQ(a.get(0, 0), 0); EXPECT_EQ(a.get(0, 1), 1); EXPECT_EQ(a.get(0, 2), 2);
		EXPECT_EQ(a.get(1, 0), 2); EXPECT_EQ(a.get(1, 1), 1); EXPECT_EQ(a.get(1, 2), 0);
	}

	// 10 < q <= 36: letters
	{
		using A = FqMatrix<T, 2, 3, 17, packed>;
		const A a{"09aGbg"};
		EXPECT_EQ(a.get(0, 0), 0); EXPECT_EQ(a.get(0, 1), 9); EXPECT_EQ(a.get(0, 2), 10);
		EXPECT_EQ(a.get(1, 0), 16); EXPECT_EQ(a.get(1, 1), 11); EXPECT_EQ(a.get(1, 2), 16);
	}

	// q > 36: separated decimal numbers
	{
		using A = FqMatrix<T, 2, 3, 257, false>;
		const A a{"0 256 17\n100, 1,255"};
		EXPECT_EQ(a.get(0, 0), 0); EXPECT_EQ(a.get(0, 1), 256); EXPECT_EQ(a.get(0, 2), 17);
		EXPECT_EQ(a.get(1, 0), 100); EXPECT_EQ(a.get(1, 1), 1); EXPECT_EQ(a.get(1, 2), 255);
	}
}

TEST(FqMatrix, copy_sub) {
	using Big = FqMatrix<T, 10, 12, 5, packed>;
	using Small = FqMatrix<T, 3, 4, 3, packed>;
	Big b; b.random();
	const Big orig = b;
	Small a; a.random();

	constexpr uint32_t srow = 6, scol = 7;
	b.copy_sub(a, srow, scol);
	for (uint32_t i = 0; i < 10; ++i) {
		for (uint32_t j = 0; j < 12; ++j) {
			const bool inside = (i >= srow) && (i < srow + 3) && (j >= scol) && (j < scol + 4);
			if (inside) {
				EXPECT_EQ(b.get(i, j), a.get(i - srow, j - scol));
			} else {
				EXPECT_EQ(b.get(i, j), orig.get(i, j));
			}
		}
	}
}

TEST(FqMatrix, zero_row) {
	M m; m.random();
	const M orig = m;
	m.zero_row(5);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), (i == 5) ? 0 : orig.get(i, j));
		}
	}
}

TEST(FqMatrix, Init) {
	M m = M{};
	(void) m;
}

TEST(FqMatrix, SubScription) {
	M m = M{};
	m.random();

	for (uint32_t i = 0; i < M::ROWS; ++i) {
		for (uint32_t j = 0; j < M::COLS; ++j) {
			EXPECT_EQ(m[i][j], m.get(i, j));
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

	M::sub(m3, m1, m2);
	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m3.get(i, j), m1.get(i, j));
		}
	}

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
		const int a = rng() % q;
		ptr2[i] = a;
		sprintf(ptr + i, "%d", a);
	}

	M m = M{ptr};

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < ncols; ++j) {
			EXPECT_EQ(m.get(i, j), ptr2[i * ncols + j]);
		}
	}

	free(ptr);
	;
	free(ptr2);
}

TEST(FqMatrix, SimpleTranspose) {
	constexpr_for<400, 430, 7>([](const auto __nrows) {
		constexpr_for<400, 480, 11>([__nrows](const auto __ncols) {
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
	constexpr_for<400, 430, 7>([](const auto __nrows) {
		constexpr_for<400, 470, 11>([__nrows](const auto __ncols) {
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
	EXPECT_GT(rank, 0);

	//std::cout << rank << std::endl;
	//m.print();

	for (uint32_t i = 0; i < nrows; ++i) {
		for (uint32_t j = 0; j < rank; ++j) {
			EXPECT_EQ(m.get(i, j), i == j);
		}
	}
}

// NOTE disable because ARM CI pipline has not enough space
#ifndef USE_NEON
TEST(FqMatrix, m4ri) {
	constexpr_for<400, 430, 9>([](const auto __nrows) {
		constexpr_for<410, 440, 10>([__nrows](const auto __ncols) {
			if constexpr (__nrows <= __ncols) {
				using M = FqMatrix<T, __nrows, __ncols, q, true>;
				M m = M{};
				m.random();
				const uint32_t rank = m.m4ri();
				EXPECT_GT(rank, __nrows - 20);

				for (uint32_t i = 0; i < rank; ++i) {
					for (uint32_t j = 0; j < rank; ++j) {
						EXPECT_EQ((bool) m.get(i, j), i == j);
					}
				}
			}
		});
	});
}

TEST(FqMatrix, m4ri_precompute_table) {
	using M2 = FqMatrix<T, 100, 130, q, true>;
	for (uint32_t k = 0; k < 10; ++k) {
		M2 a; a.random();
		M2 b = a;
		const uint32_t r1 = a.template m4ri<4, false>();
		const uint32_t r2 = b.template m4ri<4, true>();
		EXPECT_EQ(r1, r2);
		EXPECT_TRUE(M2::is_equal(a, b));
	}
}

/// q = 5, 7: pivots other than 1 and q-1
template<const uint64_t qq>
static void markov_gaus_q() {
	constexpr uint32_t nr = 60, nc = 90, l = 10, c = 5;
	using MM = FqMatrix<T, nr, nc, qq, true>;
	MM m;
	Permutation P(nc);
	uint32_t rank = 0;
	while (true) {
		m.random();
		rank = m.gaus(nr - l);
		if (rank >= nr - l) { break; }
	}

	for (uint32_t k = 0; k < 50; ++k) {
		const uint32_t rank2 = m.template markov_gaus<c, nr - l>(P);
		EXPECT_EQ(rank, rank2);
		for (uint32_t i = 0; i < nr; ++i) {
			for (uint32_t j = 0; j < nr - l; ++j) {
				EXPECT_EQ(m.get(i, j), i == j);
			}
		}
	}
}

TEST(FqMatrix, markov_gaus_q) {
	markov_gaus_q<5>();
	markov_gaus_q<7>();
}

TEST(FqMatrix, markov_gaus) {
	constexpr_for<400, 430, 7>([](const auto __nrows) {
		constexpr_for<400, 440, 11>([__nrows](const auto __ncols) {
			if constexpr (__nrows <= __ncols) {
				using M = FqMatrix<T, __nrows, __ncols, q, true>;
				// using MT = FqMatrix<T, __ncols, __nrows, q, true>;
				constexpr uint32_t l = 10;
				constexpr uint32_t c = 5;
				M m = M{};
				Permutation P(__ncols);
				uint32_t rank = 0;

				while (true) {
					m.random();
					rank = m.gaus(__nrows - l);
					EXPECT_GT(rank, 0);
					if (rank >= __nrows - l) { break; }
				}

				for (uint32_t k = 0; k < 10; ++k) {
					uint32_t rank2 = m.template markov_gaus<c, __nrows - l>(P);
					EXPECT_EQ(rank, rank2);

					for (uint32_t i = 0; i < __nrows; ++i) {
						for (uint32_t j = 0; j < __nrows - l; ++j) {
							EXPECT_EQ(m.get(i, j), i == j);
						}
					}
				}
			}
		});
	});
}


TEST(FqMatrix, fixgaus) {
	constexpr_for<400, 430, 7>([](const auto __nrows) {
		constexpr_for<400, 470, 11>([__nrows](const auto __ncols) {
			if constexpr (__nrows <= ncols) {
				using M = FqMatrix<T, __nrows, __ncols, q, true>;
				// using MT = FqMatrix<T, __ncols, __nrows, q, true>;
				M m = M{};
				m.random();

				Permutation P{__ncols};
				const uint32_t rank = m.gaus();
				const uint32_t rank2 = m.fix_gaus(P, rank, __nrows);
				EXPECT_GT(rank, __nrows - 10);

				for (uint32_t i = 0; i < __nrows; ++i) {
					for (uint32_t j = 0; j < rank2; ++j) {
						if (i == j) {
							EXPECT_EQ(m.get(i, j), 1u);
							continue;
						}
						EXPECT_EQ(m.get(i, j), 0u);
					}
				}
			}
		});
	});
}

TEST(FqMatrix, mult) {
	constexpr_for<3, 50, 40>([](const auto __nrows) {
		constexpr_for<7, 50, 40>([__nrows](const auto __ncols) {
			constexpr_for<11, 60, 40>([__nrows, __ncols](const auto __ncols_prime) {
				using AT = FqMatrix<T, __nrows, __ncols, q, true>;
				using BT = FqMatrix<T, __ncols, __ncols_prime, q, true>;
				using CT = FqMatrix<T, __nrows, __ncols_prime, q, true>;

				AT a = AT{};
				BT b = BT{};
				CT c = CT{};
				a.identity();
				b.identity();
				c.identity();
				std::cout << __nrows << " " << __ncols << " " << __ncols_prime << std::endl;
				AT::mul(c, a, b);

				constexpr auto min = std::min({(size_t)__nrows, (size_t)__ncols, (size_t)__ncols_prime,});
				for (uint32_t i = 0; i < min; ++i) {
					for (uint32_t j = 0; j < min; ++j) {
						if (i == j) {
							EXPECT_EQ(c.get(i, j), 1u);
							continue;
						}
						EXPECT_EQ(c.get(i, j), 0u);
					}
				}
			});
		});
	});
}
#endif

/// q > 2**16
TEST(FqMatrix, mult_large_q) {
	constexpr uint64_t qq = (1ull << 31) - 1ull;
	constexpr uint32_t n = 8, m = 5;
	using A = FqMatrix<uint64_t, n, n, qq, false>;
	using B = FqMatrix<uint64_t, n, m, qq, false>;
	A a, b, c;
	B d, e;
	a.random(); b.random(); d.random();
	A::mul(c, a, b);
	A::template mul<m>(e, a, d);

	for (uint32_t i = 0; i < n; ++i) {
		for (uint32_t j = 0; j < n; ++j) {
			unsigned __int128 s = 0;
			for (uint32_t k = 0; k < n; ++k) {
				s += (unsigned __int128)a.get(i, k) * b.get(k, j);
			}
			EXPECT_EQ(c.get(i, j), (uint64_t)(s % qq));
		}
		for (uint32_t j = 0; j < m; ++j) {
			unsigned __int128 s = 0;
			for (uint32_t k = 0; k < n; ++k) {
				s += (unsigned __int128)a.get(i, k) * d.get(k, j);
			}
			EXPECT_EQ(e.get(i, j), (uint64_t)(s % qq));
		}
	}
}

TEST(FqMatrix, augment) {
	using L = FqMatrix<T, 3, 4, q, packed>;
	using R = FqMatrix<T, 3, 2, q, packed>;
	using O = FqMatrix<T, 5, 6, q, packed>;
	L l; l.random();
	R r; r.random();
	O o; o.random();
	O::augment(o, l, r);
	for (uint32_t i = 0; i < 5; ++i) {
		for (uint32_t j = 0; j < 6; ++j) {
			const uint32_t expect = (i >= 3) ? 0 : ((j < 4) ? l.get(i, j) : r.get(i, j - 4));
			EXPECT_EQ(o.get(i, j), expect);
		}
	}
}

/// vector matrix multiplication, transposed and normal, vs a naive reference
template<const uint64_t qq, const bool pk, typename TT=T>
static void vec_mul_check() {
	constexpr uint32_t nr = 37, nc = 70;
	using MM = FqMatrix<TT, nr, nc, qq, pk>;
	using VR = typename std::conditional<pk, FqPackedVector<nr, qq, TT>, FqNonPackedVector<nr, qq, TT>>::type;
	using VC = typename std::conditional<pk, FqPackedVector<nc, qq, TT>, FqNonPackedVector<nc, qq, TT>>::type;
	MM m; m.random();
	VR r; VC c;

	for (uint32_t k = 0; k < 10; ++k) {
		// out = in * M
		r.random(); c.random();
		m.mul(c, r);
		for (uint32_t i = 0; i < nc; ++i) {
			unsigned __int128 s = 0;
			for (uint32_t j = 0; j < nr; ++j) { s += (unsigned __int128)m.get(j, i) * r.get(j); }
			EXPECT_EQ((uint64_t)c.get(i), (uint64_t)(s % qq));
		}

		// out = M * in
		r.random(); c.random();
		m.mul(r, c);
		for (uint32_t i = 0; i < nr; ++i) {
			unsigned __int128 s = 0;
			for (uint32_t j = 0; j < nc; ++j) { s += (unsigned __int128)m.get(i, j) * c.get(j); }
			EXPECT_EQ((uint64_t)r.get(i), (uint64_t)(s % qq));
		}
	}
}

TEST(FqMatrix, vector_mul) {
	vec_mul_check<3, true>();
	vec_mul_check<3, false>();
	vec_mul_check<7, true>();
	vec_mul_check<2, false>();
	vec_mul_check<(1ull << 31) - 1ull, false, uint64_t>();
}

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
	rng_seed(0);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
