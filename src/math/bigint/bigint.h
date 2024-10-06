#ifndef CRYPTANALYSISLIB_BIGINT_H
#define CRYPTANALYSISLIB_BIGINT_H

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

template<typename T>
struct dbl_bitlen { using type = void; };
template<>
struct dbl_bitlen<uint8_t> { using type = uint16_t; };
template<>
struct dbl_bitlen<uint16_t> { using type = uint32_t; };
template<>
struct dbl_bitlen<uint32_t> { using type = uint64_t; };
template<>
struct dbl_bitlen<uint64_t> { using type = __uint128_t; };

///
/// \tparam N
/// \tparam T
template<size_t N,
         typename T = uint64_t,
         const bool sign=false>
#if __cplusplus > 201709L
	requires std::unsigned_integral<T>
#endif
struct big_int : std::array<T, N> {
private:
	///
	static_assert(N > 0);

	constexpr static size_t FP_DIGIT = sizeof(T);

	uint32_t used;
public:

	constexpr static size_t LIMBS = N;

	/// make sure that data is zero initialized
	constexpr big_int() noexcept{
		this->fill(static_cast<T>(0));
	}

	/// simple integer constructor
	/// \param a
	constexpr big_int(const T a) noexcept {
		this->fill(static_cast<T>(0));
		this->operator[](0) = a;
	}

	/// IMPORTANT: that copy constructor is auto casting the result down from a
	/// big_int<N + 1> (result of an addition) to a big_int<N>
	/// If you really one the EXACT result and cannot ignore the carry, you have to
	/// write the addition (just an example) like this:
	/// 	const auto tmp = a + b;
	/// and not like:
	/// 	const big_int<N> tmp = a + b; // this will fail if a and b are of type big_int<N>
	/// \tparam M
	/// \param a
	template<size_t M>
	constexpr big_int(const big_int<M> &a) noexcept {
		for (size_t i = 0; i < std::min(N, M); ++i) {
			this->operator[](i) = a[i];
		}
	}


	/// [ 0 | 1 | 2 | ... | N1 ]
	/// copies t[begin, end) into a new variable
	/// \tparam Begin
	/// \tparam End
	/// \tparam Padding
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \return
	template <size_t Begin,
			size_t End,
			size_t Padding=0,
			size_t N1>
	constexpr static auto take(const big_int< N1, T> t) noexcept {
		static_assert(End >= Begin, "invalid range");
		static_assert(End - Begin <= N1, "invalid range");

		big_int<End - Begin + Padding, T> res{};
		for (auto i = Begin; i < End; ++i) {
			res[i-Begin] = t[i];
		}

		return res;
	}

	///
	/// \tparam ResultLength
	/// \tparam T
	/// \tparam N1
	template<size_t ResultLength,
	         size_t N1>
	constexpr static auto take(big_int<N1, T> t,
	                           const size_t Begin,
	                           const size_t End,
	                           const size_t Offset = 0) noexcept {
		big_int<ResultLength, T> res{};
		for (auto i = Begin; i < End; ++i) {
			res[i - Begin + Offset] = t[i];
		}

		return res;
	}

	/// add N extra limbs (at msb side)
	/// \tparam N
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \return
	template<size_t M, size_t N1>
	constexpr static auto pad(big_int<N1, T> t) {
		return take<0, N1, M>(t);
	}

	/// take first N limbs
	/// first<N>(x) corresponds with x modulo (2^64)^N
	/// \tparam N
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \return
	template<size_t M, size_t N1>
	constexpr static auto first(big_int<N1, T> t) noexcept {
		return take<0, M>(t);
	}

	/// take first N limbs, runtime version
	/// first(x,N) corresponds with x modulo (2^64)^N
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \param N
	/// \return
	template<size_t N1>
	constexpr static auto first(const big_int<N1, T> t,
	                            const size_t M) noexcept {
		return take<N1>(t, 0, M);
	}

	/// N limbs, Kth limb set to one
	/// \tparam K
	/// \tparam N
	/// \tparam T
	/// \return
	template<size_t K, size_t M>
	constexpr static auto unary_encoding() noexcept {
		big_int<M, T> res{};
		res[K] = 1;
		return res;
	}

	///
	/// \tparam N
	/// \tparam T
	/// \param K
	/// \return
	template<size_t M>
	constexpr static auto unary_encoding(const size_t K) noexcept {
		big_int<M, T> res{};
		res[K] = 1;
		return res;
	}

private:
	/// addition function for inputs of the same length
	/// \tparam T
	/// \tparam N
	/// \param a
	/// \param b
	/// \return
	constexpr static inline auto add_same(const big_int<N, T> &a,
	                               		  const big_int<N, T> &b) noexcept {
		T carry{};
		big_int<N + 1, T> r{};

		for (auto i = 0U; i < N; ++i) {
			auto aa = a[i];
			auto sum = aa + b[i];
			auto res = sum + carry;
			carry = (sum < aa) | (res < sum);
			r[i] = res;
		}

		r[N] = carry;
		return r;
	}
public:

	/// \tparam T
	/// \tparam M
	/// \tparam N
	/// \param a
	/// \param b
	/// \return
	template<size_t M>
	constexpr static inline auto add(const big_int<M, T> &a,
	                                 const big_int<N, T> &b) noexcept {
		if constexpr (std::is_constant_evaluated()) {
			constexpr auto L = std::max(M, N);
			return add_same(pad<L - M>(a), pad<L - N>(b));
		} else {
			y = clen(MAX(a->used, b->used));
			oldused = MIN((unsigned int)c->used, FP_SIZE);
			c->used = y;

			t = 0;
			for (x = 0; x < y; x++) {
				t += ((fp_word) a->dp[x]) + ((fp_word) b->dp[x]);
				c->dp[x] = (fp_digit) t;
				t >>= DIGIT_BIT;
			}

			if (t != 0 && x < FP_SIZE) {
				c->dp[c->used++] = (fp_digit) t;
				++x;
			}

			c->used = x;
			for (; x < oldused; x++) {
				c->dp[x] = 0;
			}

			fp_clamp(c);

		}
	}


	/// NOTE that's cheating
	template<size_t M, typename TT>
	bool operator==(big_int<M, TT> const &tc) const {
		for (size_t i = 0; i < std::min(N, M); i++) {
			if (this->operator[](i) != tc[i]) {
				return false;
			}
		}

		return true;
	}

	///
	/// \tparam M
	/// \tparam TT
	/// \param tc
	/// \return
	template<size_t M, typename TT>
	bool operator<(big_int<M, TT> const &tc) const {
		// TODO maybe add to FqVector Datatype and add iterators
		return std::lexicographical_compare(this->begin(), this->end(),
		                                    tc.begin(), tc.end());
	}

	///
	/// \param os
	/// \param tc
	/// \return
	friend std::ostream& operator<<(std::ostream& os, big_int const &tc) {
		for (size_t i = 0; i < N; i++) {
			os << tc[i] << " ";
		}

		return os << ", size: " << N << std::endl;
	}
};




///
/// \tparam T
/// \tparam N
/// \param a
/// \param b
/// \return
template<typename T, size_t N>
constexpr auto subtract_same(big_int<N, T> a, big_int<N, T> b) {
	T carry{};
	big_int<N + 1, T> r{};

	for (auto i = 0U; i < N; ++i) {
		auto aa = a[i];
		auto diff = aa - b[i];
		auto res = diff - carry;
		carry = (diff > aa) | (res > diff);
		r[i] = res;
	}

	// sign extension
	r[N] = carry * static_cast<T>(-1);
	return r;
}

///
/// \tparam T
/// \tparam M
/// \tparam N
/// \param a
/// \param b
/// \return
//template<typename T, size_t M, size_t N>
//constexpr auto sub(big_int<M, T> a, big_int<N, T> b) {
//	constexpr auto L = std::max(M, N);
//	return subtract_same(cryptanalysislib::pad<L - M>(a), cryptanalysislib::pad<L - N>(b));
//}
//
/////
///// \tparam padding_limbs
///// \tparam M
///// \tparam N
///// \tparam T
///// \param u
///// \param v
///// \return
//template<size_t padding_limbs = 0U, size_t M, size_t N, typename T>
//constexpr inline auto mul(big_int<M, T> u, big_int<N, T> v) {
//	using TT = typename dbl_bitlen<T>::type;
//	constexpr uint32_t digits = std::numeric_limits<T>::digits;
//
//	big_int<M + N + padding_limbs, T> w{};
//	for (auto j = 0U; j < N; ++j) {
//		T k = 0U;
//		for (auto i = 0U; i < M; ++i) {
//			TT t = static_cast<TT>(u[i]) * static_cast<TT>(v[j]) + w[i + j] + k;
//			w[i + j] = static_cast<T>(t);
//			k = t >> digits;
//		}
//		w[j + M] = k;
//	}
//	return w;
//}
//
//template<typename Q, typename R>
//struct DivisionResult {
//	Q quotient;
//	R remainder;
//};
//
/////
///// \tparam M
///// \tparam N
///// \tparam T
///// \param u
///// \param v
///// \return
//template<size_t M, size_t N, typename T>
//constexpr DivisionResult<big_int<M, T>, big_int<N, T>> div(big_int<M, T> u,
//                                                           big_int<N, T> v) {
//	// Knuth's "Algorithm D" for multiprecision division as described in TAOCP
//	// Volume 2: Seminumerical Algorithms
//	// combined with short division
//
//	//
//	// input:
//	// u  big_int<M>,      M>=N
//	// v  big_int<N>
//	//
//	// computes:
//	// quotient = floor[ u/v ]
//	// rem = u % v
//	//
//	// returns:
//	// std::pair<big_int<N+M>, big_int<N>>(quotient, rem)
//
//	using TT = typename dbl_bitlen<T>::type;
//	size_t tight_N = N;
//	while (tight_N > 0 && v[tight_N - 1] == 0)
//		--tight_N;
//
//	if (tight_N == 0)
//		return {};// division by zero
//
//	big_int<M, T> q{};
//
//	if (tight_N == 1) {// short division
//		TT r{};
//		for (int i = M - 1; i >= 0; --i) {
//			TT w = (r << std::numeric_limits<T>::digits) + u[i];
//			q[i] = w / v[0];
//			r = w % v[0];
//		}
//		return {q, {static_cast<T>(r)}};
//	}
//
//	uint8_t k = 0;
//	while (v[tight_N - 1] <
//	       (static_cast<T>(1) << (std::numeric_limits<T>::digits - 1))) {
//		++k;
//		v = cryptanalysislib::first<N>(shift_left(v, 1));
//	}
//	auto us = shift_left(u, k);
//
//	for (int j = M - tight_N; j >= 0; --j) {
//		TT tmp = us[j + tight_N - 1];
//		TT tmp2 = us[j + tight_N];
//		tmp += (tmp2 << std::numeric_limits<T>::digits);
//		TT qhat = tmp / v[tight_N - 1];
//		TT rhat = tmp % v[tight_N - 1];
//
//		auto b = static_cast<TT>(1) << std::numeric_limits<T>::digits;
//		while (qhat == b ||
//		       (qhat * v[tight_N - 2] >
//		        (rhat << std::numeric_limits<T>::digits) + us[j + tight_N - 2])) {
//			qhat -= 1;
//			rhat += v[tight_N - 1];
//			if (rhat >= b)
//				break;
//		}
//		auto true_value = subtract(take<N + 1>(us, j, j + tight_N + 1),
//		                           mul(v, big_int<1, T>{{static_cast<T>(qhat)}}));
//		if (true_value[tight_N]) {
//			auto corrected =
//			        add_ignore_carry(true_value, cryptanalysislib::unary_encoding<N + 2, T>(tight_N + 1));
//			auto new_us_part = add_ignore_carry(corrected, pad<2>(v));
//			for (size_t i = 0; i <= tight_N; ++i)
//				us[j + i] = new_us_part[i];
//			--qhat;
//		} else {
//			for (size_t i = 0; i <= tight_N; ++i)
//				us[j + i] = true_value[i];
//		}
//		q[j] = qhat;
//	}
//	return {q, shift_right(cryptanalysislib::first<N>(us), k)};
//}
//
///// NOTE: this function will use a sginaficant amount of compiler ressources.
///// you may need to increase the constexpr steps via:
/////    -fconstexpr-steps=99999999
///// \tparam N
///// \tparam T
///// \param n
///// \param k
///// \return
//template<size_t N, typename T, const T n, const T k>
//constexpr big_int<N, T> binomial() {
//	static_assert(n * k < 1u<<20); // we need to enforce some limit
//	static_assert(k <= n);
//
//	const big_int<N, T> one = big_int<N, T>(1u);
//
//	std::array<std::array<big_int<N, T>, k+1>, n+1> tmp;
//	for (T i = 0; i <= n; ++i) {
//		for (T j = 0; j <= std::min(i, k); ++j) {
//			if ((j == 0) || (j == i)) {
//				tmp[i][j] = one;
//			} else {
//				tmp[i][j] = big_int<N, T>{tmp[i-1][j-1] + tmp[i-1][j]};
//			}
//		}
//	}
//
//	return tmp[n][k];
//}

//template<typename T, size_t N1, size_t N2>
//constexpr auto operator/(const big_int<N1, T> &a,
//                         const big_int<N2, T> &b) noexcept {
//	return div(a, b).quotient;
//}
//
//template<typename T, size_t N1, size_t N2>
//constexpr auto operator%(const big_int<N1, T> &a,
//                         const big_int<N2, T> &b) noexcept {
//	return div(a, b).remainder;
//}
//
//template<typename T, size_t N1, size_t N2>
//constexpr auto operator*(const big_int<N1, T> &a,
//                         const big_int<N2, T> &b) noexcept {
//	return mul(a, b);
//}

template<typename T, size_t N1, size_t N2>
constexpr auto operator+(const big_int<N1, T> &a,
                         const big_int<N2, T> &b) noexcept {
	return big_int<N2, T>::add(a, b);
}

//template<typename T, size_t N1, size_t N2>
//constexpr auto operator-(const big_int<N1, T> &a,
//                         const big_int<N2, T> &b) noexcept {
//	return sub(a, b);
//}
#endif
