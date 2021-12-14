#ifndef SMALLSECRETLWE_MATRIX_H
#define SMALLSECRETLWE_MATRIX_H

#include "m4ri/m4ri.h"
#include "m4ri/strassen.h"
#include "kAry_type.h"
#include "label.h"
#include "value.h"

#include <limits>
#include <vector>
#include <array>
#include <cmath>

#include <gmpxx.h>
#include <gmp.h>

template<class Matrix>
class Matrix_T {
public:
	Matrix_T(Matrix A) : m(A) {}

	// The uncommented functions are commented out, because auto function cannot be virtual
	// const auto value(const uint64_t i, const uint64_t j) const = 0;

	// virtual auto limb(const uint64_t i, const uint64_t j) = 0;
	// virtual auto uint64_t limb(const uint64_t i, const uint64_t j) const = 0;

	// auto operator()(uint64_t i) = 0;
	// const auto operator()(const uint64_t i) const = 0;

	// uint64_t operator()(uint64_t i, uint64_t j) = 0;
	// const uint64_t operator()(uint64_t i, uint64_t j) const = 0;

	// auto *ptre(const uint64_t i) = 0;

	virtual bool binary() const = 0;

	// Modify Functions
	virtual void fill(const uint64_t a) = 0;
	virtual void resize(const uint64_t i, const uint64_t j) = 0;

	// Access Functions
	virtual uint64_t get_rows() const = 0;
	virtual uint64_t get_cols() const = 0;

	virtual uint64_t limbs() const = 0;

private:
	Matrix m;
};

template<>
class Matrix_T<mzd_t *> {
public:

	Matrix_T(mzd_t *A) : m(A) {}

	const auto value(const uint64_t i, const uint64_t j) const  { return mzd_read_bit(m, i, j); };

	auto limb(const uint64_t i, const uint64_t j) { ASSERT(i < uint64_t(m->nrows) && j < uint64_t(m->width)); return m->rows[i][j]; }
	const auto limb(const uint64_t i, const uint64_t j) const { ASSERT(i < uint64_t(m->nrows) && j < uint64_t(m->width) ); return m->rows[i][j]; };

	auto operator()(uint64_t i) { ASSERT(i < uint64_t(m->nrows)); return m->rows[i]; }
	const auto operator()(const uint64_t i) const { ASSERT(i < uint64_t(m->nrows)); return m->rows[i]; }

	auto operator()(uint64_t i, uint64_t j) { return mzd_read_bit(m, i, j); }
	const auto operator()(uint64_t i, uint64_t j) const { return mzd_read_bit(m, i, j); }

	auto *ptr(const uint64_t i) { ASSERT(i < uint64_t(m->nrows)); return m->rows[i]; };
	const auto *ptr(const uint64_t i) const { ASSERT(i < uint64_t(m->nrows)); return m->rows[i]; };

	constexpr bool binary() const { return true; }

	// maybe make sure that only 0/1 is a valid input
	void fill(const uint64_t a) {
		for (int i = 0; i < m->nrows; ++i) {
			for (int j = 0; j < m->nrows; ++j) {
				mzd_write_bit(m, i, j, a&1u);
			}
		}
	};

	void gen_uniform(const uint64_t bits) { mzd_randomize(m); };
	void gen_identity(const uint64_t d) {
		ASSERT(d == uint64_t(m->nrows) && d == uint64_t(m->ncols));
		for (uint64_t i = 0; i < d; ++i) {
			mzd_write_bit(m, i, i, 1);
		}
	}


	void resize(const uint64_t i, const uint64_t j) { mzd_free(m); m = mzd_init(i, j); };

	uint64_t get_rows() const { return m->nrows; };
	uint64_t get_cols() const { return m->ncols; };

	uint64_t limbs() const { return m->width; }
	const auto matrix() const { return m; }

private:
	mzd_t *m;
};

#ifdef USE_FPLLL
#include "fplll/nr/matrix.h"
#include "glue_fplll.h"


/// Floyd: needed to implement this function, because otherwise i would needed to implement a custom << operator for every template.
template<typename ZT>
static void print_matrix(fplll::ZZ_mat<ZT> &m) {
	for (int i = 0; i < m.get_rows(); ++i) {
		for (int j = 0; j < m.get_cols(); ++j) {
			std::cout << m(i, j).data() << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

/// Abstraction over Label_Type = kAryType<T, T2, q>
/// \tparam Label_Type
template<class Label_Type>
class Matrix_T<ZZ_mat<Label_Type>> {
public:
	Matrix_T(ZZ_mat<Label_Type> &A) : m(A) {}

	const auto value(const uint64_t i, const uint64_t j) const { return m(i, j).get_data(); };

	auto limb(const uint64_t i, const uint64_t j) { return m(i, j).get_data(); }
	const auto limb(const uint64_t i, const uint64_t j) const { return m(i, j).get_data(); };

	auto operator()(uint64_t i) { return m[i]; }
	const auto operator()(const uint64_t i) const { return m[i]; }

	auto operator()(uint64_t i, uint64_t j) { return m(i, j); }
	const auto operator()(uint64_t i, uint64_t j) const { return m(i, j); }

	auto *ptr(const uint64_t i) { ASSERT(i < m.get_rows()); return &m.operator[](i).operator[](0); };
	const auto *ptr(const uint64_t i) const { ASSERT(i < m.get_rows()); return &m.operator[](i).operator[](0); };

	constexpr bool binary() const { return false; }

	void fill(const uint64_t a) { m.fill(a); };
	void gen_uniform(const uint64_t bits) { m.gen_uniform(bits); }
	void gen_identity(const uint64_t d) { m.gen_identity(d); }

	void resize(const uint64_t i, const uint64_t j) { m.resize(i, j); };

	 uint64_t get_rows() const { return m.get_rows(); };
	 uint64_t get_cols() const { return m.get_cols(); };
	 uint64_t limbs() const { return m.get_cols(); }

	const auto matrix() const { return m; }
private:
	ZZ_mat<Label_Type> m;
};


// glue code for fplll
// call with: 'NumVect<Z_NR<mpz_t>>(...);'
template<class T1>
void bkznegate(NumVect<T1> &row) {
	ASSERT(row.size() > 0 && "wrong size");

	NumVect<T1> zero{};
	zero.resize(row.size());

	for (int i = 0; i < row.size(); ++i) {
		zero[i] = 0;
	}
	zero.sub(row);
	row.swap(zero);
}

// glue code for fplll
// call with equal<Z_NR<mpz_t>>(...)
// returns true if equal
template<class T>
bool equal(const fplll::NumVect<T> &t1, const fplll::MatrixRow<T> &t2, const int n, const bool exact = false) {
	if (exact)
		if (t1.size() != t2.size())
			return false;

	for (int i = 0; i < n; ++i) {
		if (t1[i] != t2[i])
			return false;
	}

	return true;
}

// call with equal<Z_NR<mpz_t>>(...)
// this is its own functoin, because of the n parameter
template<class T>
bool equal(const fplll::NumVect<T> &t1, const fplll::NumVect<T> &t2, const int n, const bool exact = false) {
	if (exact)
		if (t1.size() != t2.size())
			return false;

	for (int i = 0; i < n; ++i) {
		if (t1[i] != t2[i])
			return false;
	}

	return true;
}
#endif

// IMPORTANT: input matrix MUST NOT be transposed
template<class Label, class Value, class T>
static void new_vector_matrix_product(Label &result,
                                      const Value &x,
                                      const Matrix_T<T> &m) {
	for (uint32_t j = 0; j < result.size(); j++)
		result.data()[j] = 0;

	if constexpr(Label::binary()) {
		// Allow m to be bigger than the vector
		ASSERT(m.limbs() >= x.data().limbs());
		ASSERT(m.get_cols() >= x.size());
		ASSERT(m.get_rows() == result.size());

		// const uint64_t len = m.get_cols();
		const uint64_t limbs = x.data().limbs(); // so the matrix can be bigger than the vector m.limbs();
		const auto limb_size = sizeof(result.data().get_type())*8;
		const auto shift = (m.get_cols()%limb_size);
		const auto high_bitmask = (m4ri_ffff >> (limb_size - (shift)) % limb_size);

		auto xor_up_row = [limbs, high_bitmask](const auto *a, const auto *b) {
			ASSERT(limbs > 0);
			uint64_t ret = 0;
			for (uint32_t i = 0; i < limbs-1; ++i) {
				ret += __builtin_popcountll(a[i]&b[i]);
			}
			ret += __builtin_popcountll((a[limbs-1]&b[limbs-1]) & high_bitmask);
			return ret%2;
		};

		// naive implementation.
		for (uint32_t i = 0; i < 1; i++) {
			for (uint32_t j = 0; j < m.get_rows(); j++) {
				bool bit = xor_up_row(x.data().data().data(), m.ptr(j));
				result[j] = bit;
			}
		}

	} else {
		for (int i = 0; i < m.get_rows(); i++) {
			for (int j = 0; j < m.get_cols(); j++) {
				auto a = x[i] * m.value(i, j);
				auto b = a + result.data()[j];
				result.data()[j] = b;
			}
		}
	}
}
#endif //SMALLSECRETLWE_MATRIX_H
