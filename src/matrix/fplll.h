#ifndef CRYPTANALYSISLIB_FPLLL_H
#define CRYPTANALYSISLIB_FPLLL_H
/// TODO

#include <gmpxx.h>
#include <gmp.h>

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
#endif//CRYPTANALYSISLIB_FPLLL_H
