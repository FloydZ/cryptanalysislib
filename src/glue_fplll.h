#ifndef SMALLSECRETLWE_GLUE_FPLLL_H
#define SMALLSECRETLWE_GLUE_FPLLL_H

#include <cstdint>

#include "fplll/nr/matrix.h"
#include "fplll/nr/nr_Z.inl"

#include "kAry_type.h"

using namespace fplll;

/* specialization */
template<>
inline Z_NR<uint16_t>::Z_NR() {}

template<>
inline Z_NR<uint16_t>::Z_NR(const Z_NR<uint16_t> &z) : data(z.data) {}

//template<>
//inline Z_NR<uint16_t>::Z_NR(const uint16_t &z) : data(z) {}

template<>
inline Z_NR<uint16_t>::~Z_NR() {}

template<>
inline void Z_NR<uint16_t>::operator=(long a) { data = a; }

template<>
inline void Z_NR<int16_t>::operator=(long a) { data = a; }

template<>
inline bool Z_NR<unsigned short>::operator==(long a) const {
	return this->data == a;
}


template<class T>
inline Z_NR<T>::Z_NR() {}

//template<>
//inline Z_NR<Label_Type>::Z_NR(const Z_NR<Label_Type> &z) : data(z.get_data()) {}

template<class T>
inline Z_NR<T>::Z_NR(const T &z) : data(z) {}

template<class T>
inline Z_NR<T>::Z_NR(const Z_NR<T> &z) : data(z.data) {}

template<class T>
inline Z_NR<T>::~Z_NR() {}

template<class T>
inline void Z_NR<T>::operator=(long a) { data = uint64_t (a); }

//template<>
//inline void Z_NR<Label_Type>::operator=(const Z_NR<Label_Type> &a) { data = a.data; }
//
//template<>
//inline void Z_NR<Label_Type>::randb(int bits) {
//	data = fastrandombytes_uint64();
//}
//
//
//template<>
//std::ostream &fplll::operator<<<unsigned short>(std::ostream &out, fplll::Z_NR<unsigned short> const &a) {
//	out << a.get_data();
//	return out;
//}
//
//template<>
//std::ostream &fplll::operator<<<kAry_Type_T<unsigned short, unsigned int>>(std::ostream &out,
//                                                                           fplll::Z_NR<kAry_Type_T<unsigned short, unsigned int>> const &a) {
//	out << a.get_data().data();
//	return out;
//}
//
//template<>
//bool fplll::Z_NR<kAry_Type_T<unsigned short, unsigned int>>::operator==(
//		fplll::Z_NR<kAry_Type_T<unsigned short, unsigned int>> const &obj) const {
//	return this->get_data().data() == obj.get_data().data();
//}
//
//template<>
//void fplll::Z_NR<kAry_Type_T<unsigned short, unsigned int> >::swap(
//		fplll::Z_NR<kAry_Type_T<unsigned short, unsigned int> > &b) {
//	kAry_Type_T<unsigned short, unsigned int> tmp = this->get_data();
//	this->data = b.data;
//	b.data = tmp;
//}
//
//template<typename T, typename T2>
//short operator^(short obj1, const fplll::Z_NR<kAry_Type_T<T, T2>> &obj2) {
//	return obj1 ^ obj2.get_data().data();
//}
//
//template<typename T, typename T2>
//short operator&(const kAry_Type_T<T, T2> &obj1, const fplll::Z_NR<kAry_Type_T<T, T2>> &obj2) {
//	return obj1.data() & obj2.get_data().data();
//}
//
//
//template<typename T, typename T2>
//bool operator==(const kAry_Type_T<T, T2> &obj1, const fplll::Z_NR<kAry_Type_T<T, T2>> &obj2) {
//	return obj1.data() == obj2.get_data().data();
//}
//
//
//template<typename T, typename T2>
//short operator&(short obj1, const fplll::Z_NR<kAry_Type_T<T, T2>> &obj2) {
//	return obj1 & obj2.get_data().data();
//}

//////////
template<>
inline Z_NR<int16_t>::Z_NR() {}

//template<>
//inline Z_NR<int16_t>::Z_NR(const Z_NR<int16_t> &a) {data = a.data; }


template<>
inline Z_NR<int16_t>::~Z_NR() {}

//template<>
//inline void Z_NR<uint16_t>::operator=(const Label_Type &a) { data = a.data(); }
template<>
inline void Z_NR<uint16_t>::operator=(const Z_NR<uint16_t> &a) { data = a.data; }



template<>
bool Z_NR<unsigned short>::operator==(fplll::Z_NR<unsigned short> const &a) const {
	return this->data == a.data;
}

template<typename T>
void row_copy(ZZ_mat<T> &A, const ZZ_mat<T> &B, const uint32_t row) {
	ASSERT(A.get_cols() == B.get_cols() && A.get_rows() == B.get_rows() && row < B.get_rows());
	for (int i = 0; i < B.get_cols(); ++i) {
		A(row, i) = B(row, i);
	}
}

template<typename T>
void to_row_copy(ZZ_mat<T> &y, const ZZ_mat<T> &B, const uint32_t row) {
	ASSERT(y.get_cols() == B.get_cols() && y.get_rows() ==1 && row < B.get_rows());
	for (int i = 0; i < B.get_cols(); ++i) {
		y(0, i) = B(row, i);
	}
}

template<typename T>
void from_row_copy(ZZ_mat<T> &B, const ZZ_mat<T> &y, const uint32_t row) {
	ASSERT(y.get_cols() == B.get_cols() && y.get_rows() == 1 && row < B.get_rows());
	for (int i = 0; i < B.get_cols(); ++i) {
		B(row, i) = y(0, i);
	}
}

/// e = y & B[row] (B as row base vectors)
template<typename T>
void row_and(ZZ_mat<T> &e, const ZZ_mat<T> &B, const ZZ_mat<T> &y, const uint32_t row) {
	ASSERT(e.get_cols() == B.get_cols() && e.get_cols() == y.get_cols() && e.get_rows() == 1 && y.get_rows() == 1 && row < B.get_rows());
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = B(row, i).get_si() & y(0, i).get_si();
	}
}

// e = e&y
template<typename T>
void row_and(ZZ_mat<T> &e, const ZZ_mat<T> &y) {
	ASSERT( e.get_cols() == y.get_cols() && e.get_rows() == 1 && y.get_cols() == 1 );
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = y(0, i).get_si() & e(0, i).get_si();
	}
}

/// e = y ^ B[row] (B as row base vectors)
template<typename T>
void row_xor(ZZ_mat<T> &e, const ZZ_mat<T> &B, const ZZ_mat<T> &y, const uint32_t row) {
	ASSERT(e.get_cols() == B.get_cols() && e.get_cols() == y.get_cols() && e.get_rows() == 1 && y.get_rows() == 1 && row < B.get_rows());
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = B(row, i).get_si() ^ y(0, i).get_si();
	}
}
// e = e^y
template<typename T>
void row_xor(ZZ_mat<T> &e, const ZZ_mat<T> &y) {
	ASSERT( e.get_cols() == y.get_cols() && e.get_rows() == 1 && y.get_cols() == 1 );
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = y(0, i).get_si() ^ e(0, i).get_si();
	}
}

/// e = y | B[row] (B as row base vectors)
template<typename T>
void row_or(ZZ_mat<T> &e, const ZZ_mat<T> &B, const ZZ_mat<T> &y, const uint32_t row) {
	ASSERT(e.get_cols() == B.get_cols() && e.get_cols() == y.get_cols() && e.get_rows() == 1 && y.get_rows() == 1 && row < B.get_rows());
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = B(row, i).get_si() | y(0, i).get_si();
	}
}

template<typename T>
void row_neg(ZZ_mat<T> &e, const ZZ_mat<T> &y) {
	ASSERT( e.get_rows() == 1);
	for (int i = 0; i < e.get_cols(); ++i) {
		e(0, i) = (y(0, i).get_si() ^ 1) % 2;
	}
}


template<typename T>
bool zero_row(const ZZ_mat<T> &A, const uint64_t row){
	for (int i = 0; i < A.get_cols(); ++i) {
		if (int(A(row, i).get_data()) % 2 != 0)
			return false;
	}

	return true;
}

template<>
bool zero_row<mpz_t>(const ZZ_mat<mpz_t> &A, const uint64_t row){
	fplll::Z_NR<mpz_t> two, tmp;
	two = 2;

	for (int i = 0; i < A.get_cols(); ++i) {
		tmp = A(row, i);
		if (tmp.get_si()%2 != 0)
			return false;
	}

	return true;
}

template<typename T>
uint64_t get_first_non_zero_row(const ZZ_mat<T> &A) {
	uint64_t r = 0;
	for (; r < A.get_rows(); r++) {
		if (zero_row<T>(A, r) == false)
			return r;
	}

	return r;
}

template<typename T>
uint64_t weight_column(const ZZ_mat<T> &A, const uint64_t column) {
	ASSERT(column < A.get_cols());
	uint64_t w = 0;
	for (int i = 0; i < A.get_rows(); ++i) {
		w += (int(A[i][column].get_data())%2) == 0 ? 0 : 1;
	}
	return w;
}

template<>
uint64_t weight_column<mpz_t>(const ZZ_mat<mpz_t> &A, const uint64_t column) {
	ASSERT(column < A.get_cols());

	fplll::Z_NR<mpz_t> two, tmp;
	two = 2;

	uint64_t w = 0;
	for (int i = 0; i < A.get_rows(); ++i) {
		tmp = A(i, column);
		w += tmp.get_si()%2 == 0 ? 0 : 1;
	}
	return w;
}

// returns the avg weight
template<typename T>
double weight_column(const ZZ_mat<T> &A) {
	uint64_t w = 0;
	for (int i = 0; i < A.get_cols(); ++i) {
		w += weight_column<T>(A, i);
	}

	return (double)w/(double)A.get_cols();
}

template<typename T>
uint64_t weight_row(const ZZ_mat<T> &A, const uint64_t row) {
	ASSERT(row < A.get_rows());
	uint64_t w = 0;
	for (int i = 0; i < A.get_cols(); ++i) {
		w += (int(A[row][i].get_data())%2) == 0 ? 0 : 1;
	}
	return w;
}

template<>
uint64_t weight_row<mpz_t>(const ZZ_mat<mpz_t> &A, const uint64_t row) {
	ASSERT(row < A.get_rows());

	fplll::Z_NR<mpz_t> two, tmp;
	two = 2;

	uint64_t w = 0;
	for (int i = 0; i < A.get_cols(); ++i) {
		tmp = A(row, i);
		w += tmp.get_si()%2 == 0 ? 0 : 1;
	}
	return w;
}

// returns the avg weight
template<typename T>
double weight_row(const ZZ_mat<T> &A) {
	uint64_t w = 0;
	const int r = get_first_non_zero_row<T>(A);
	for (int i = r; i < A.get_rows(); ++i) {
		w += weight_row<T>(A, i);
	}

	return (double)w/(double)(A.get_rows()-r);
}

template<typename T>
double weight_row_max(const ZZ_mat<T> &A, const uint64_t max_row) {
	ASSERT(max_row < A.get_rows());
	uint64_t w = 0;
	const int r = get_first_non_zero_row<T>(A);
	for (int i = r; i < max_row; ++i) {
		w += weight_row<T>(A, i);
	}

	return (double)w/(double)(max_row-r);
}

#endif //SMALLSECRETLWE_GLUE_FPLLL_H
