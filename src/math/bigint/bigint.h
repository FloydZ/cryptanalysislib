#ifndef CRYPTANALYSISLIB_BIGINT_H
#define CRYPTANALYSISLIB_BIGINT_H

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

///
/// \tparam N
/// \tparam T
template<size_t N,
        std::unsigned_integral T = uint64_t>
struct big_int : std::array<T, N> {
	///
	static_assert(N > 0);

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
	constexpr big_int(const big_int<M> &a) {
		for (size_t i = 0; i < std::min(N, M); ++i) {
			this->operator[](i) = a[i];
		}
	}

	/// NOTE thats cheating
	template<size_t M, typename TT>
	bool operator==(big_int<M, TT> const &tc) const {
		for (size_t i = 0; i < std::min(N, M); i++) {
			if (this->operator[](i) != tc[i]) {
				return false;
			}
		}

		return true;
	}

	template<size_t M, typename TT>
	bool operator<(big_int<M, TT> const &tc) const {
		return std::lexicographical_compare(this->begin(), this->end(),
		                                    tc.begin(), tc.end());
	}
	friend std::ostream& operator<<(std::ostream& os, big_int const &tc) {
		for (size_t i = 0; i < N; i++) {
			os << tc[i] << " ";
		}

		return os << ", size: " << N << std::endl;
	}
};

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


namespace cryptanalysislib {
	template <size_t Begin, size_t End, size_t Padding=0,
	         typename T, size_t N1>
	constexpr auto take(big_int< N1, T> t) {
		static_assert(End >= Begin, "invalid range");
		static_assert(End - Begin <= N1, "invalid range");

		big_int< End - Begin + Padding, T> res{};
		for (auto i = Begin; i < End; ++i) {
			res[i-Begin] = t[i];
		}

		return res;
	}

	///
	/// \tparam ResultLength
	/// \tparam T
	/// \tparam N1
	template<size_t ResultLength, typename T, size_t N1>
	constexpr auto take(big_int<N1, T> t,
	                    const size_t Begin,
	                    const size_t End,
	                    const size_t Offset = 0) {
		big_int<ResultLength, T> res{};
		for (auto i = Begin; i < End; ++i) {
			res[i - Begin + Offset] = t[i];
		}

		return res;
	}

	///
	/// \tparam N
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \return
	template<size_t N, typename T, size_t N1>
	constexpr auto pad(big_int<N1, T> t) {
		// add N extra limbs (at msb side)
		return take<0, N1, N>(t);
	}

	///
	/// \tparam N
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \return
	template<size_t N, typename T, size_t N1>
	constexpr auto first(big_int<N1, T> t) {
		// take first N limbs
		// first<N>(x) corresponds with x modulo (2^64)^N
		return take<0, N>(t);
	}

	///
	/// \tparam T
	/// \tparam N1
	/// \param t
	/// \param N
	/// \return
	template<typename T, size_t N1>
	constexpr auto first(big_int<N1, T> t, size_t N) {
		// take first N limbs, runtime version
		// first(x,N) corresponds with x modulo (2^64)^N
		return take<N1>(t, 0, N);
	}

	///
	/// \tparam K
	/// \tparam N
	/// \tparam T
	/// \return
	template<size_t K, size_t N, typename T = uint64_t>
	constexpr auto unary_encoding() {
		// N limbs, Kth limb set to one
		big_int<N, T> res{};
		res[K] = 1;
		return res;
	}

	///
	/// \tparam N
	/// \tparam T
	/// \param K
	/// \return
	template<size_t N, typename T = uint64_t>
	constexpr auto unary_encoding(size_t K) {
		big_int<N, T> res{};
		res[K] = 1;
		return res;
	}
}// namespace cryptanalysislib

/// addition function for inputs of the same length
/// \tparam T
/// \tparam N
/// \param a
/// \param b
/// \return
template<typename T, size_t N>
constexpr inline auto add_same(big_int<N, T> a, big_int<N, T> b) {
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
template<typename T, size_t M, size_t N>
constexpr inline auto add(big_int<M, T> a, big_int<N, T> b) {
	constexpr auto L = std::max(M, N);
	return add_same(cryptanalysislib::pad<L - M>(a), cryptanalysislib::pad<L - N>(b));
}

///
/// \tparam T
/// \tparam M
/// \tparam N
/// \param a
/// \param b
/// \return
template<typename T, size_t M, size_t N>
constexpr auto sub(big_int<M, T> a, big_int<N, T> b) {
	constexpr auto L = std::max(M, N);
	return subtract_same(cryptanalysislib::pad<L - M>(a), cryptanalysislib::pad<L - N>(b));
}

///
/// \tparam padding_limbs
/// \tparam M
/// \tparam N
/// \tparam T
/// \param u
/// \param v
/// \return
template<size_t padding_limbs = 0U, size_t M, size_t N, typename T>
constexpr inline auto mul(big_int<M, T> u, big_int<N, T> v) {
	using TT = typename dbl_bitlen<T>::type;
	constexpr uint32_t digits = std::numeric_limits<T>::digits;

	big_int<M + N + padding_limbs, T> w{};
	for (auto j = 0U; j < N; ++j) {
		T k = 0U;
		for (auto i = 0U; i < M; ++i) {
			TT t = static_cast<TT>(u[i]) * static_cast<TT>(v[j]) + w[i + j] + k;
			w[i + j] = static_cast<T>(t);
			k = t >> digits;
		}
		w[j + M] = k;
	}
	return w;
}

template<typename Q, typename R>
struct DivisionResult {
	Q quotient;
	R remainder;
};

///
/// \tparam M
/// \tparam N
/// \tparam T
/// \param u
/// \param v
/// \return
template<size_t M, size_t N, typename T>
constexpr DivisionResult<big_int<M, T>, big_int<N, T>> div(big_int<M, T> u,
                                                           big_int<N, T> v) {
	// Knuth's "Algorithm D" for multiprecision division as described in TAOCP
	// Volume 2: Seminumerical Algorithms
	// combined with short division

	//
	// input:
	// u  big_int<M>,      M>=N
	// v  big_int<N>
	//
	// computes:
	// quotient = floor[ u/v ]
	// rem = u % v
	//
	// returns:
	// std::pair<big_int<N+M>, big_int<N>>(quotient, rem)

	using TT = typename dbl_bitlen<T>::type;
	size_t tight_N = N;
	while (tight_N > 0 && v[tight_N - 1] == 0)
		--tight_N;

	if (tight_N == 0)
		return {};// division by zero

	big_int<M, T> q{};

	if (tight_N == 1) {// short division
		TT r{};
		for (int i = M - 1; i >= 0; --i) {
			TT w = (r << std::numeric_limits<T>::digits) + u[i];
			q[i] = w / v[0];
			r = w % v[0];
		}
		return {q, {static_cast<T>(r)}};
	}

	uint8_t k = 0;
	while (v[tight_N - 1] <
	       (static_cast<T>(1) << (std::numeric_limits<T>::digits - 1))) {
		++k;
		v = cryptanalysislib::first<N>(shift_left(v, 1));
	}
	auto us = shift_left(u, k);

	for (int j = M - tight_N; j >= 0; --j) {
		TT tmp = us[j + tight_N - 1];
		TT tmp2 = us[j + tight_N];
		tmp += (tmp2 << std::numeric_limits<T>::digits);
		TT qhat = tmp / v[tight_N - 1];
		TT rhat = tmp % v[tight_N - 1];

		auto b = static_cast<TT>(1) << std::numeric_limits<T>::digits;
		while (qhat == b ||
		       (qhat * v[tight_N - 2] >
		        (rhat << std::numeric_limits<T>::digits) + us[j + tight_N - 2])) {
			qhat -= 1;
			rhat += v[tight_N - 1];
			if (rhat >= b)
				break;
		}
		auto true_value = subtract(take<N + 1>(us, j, j + tight_N + 1),
		                           mul(v, big_int<1, T>{{static_cast<T>(qhat)}}));
		if (true_value[tight_N]) {
			auto corrected =
			        add_ignore_carry(true_value, cryptanalysislib::unary_encoding<N + 2, T>(tight_N + 1));
			auto new_us_part = add_ignore_carry(corrected, pad<2>(v));
			for (size_t i = 0; i <= tight_N; ++i)
				us[j + i] = new_us_part[i];
			--qhat;
		} else {
			for (size_t i = 0; i <= tight_N; ++i)
				us[j + i] = true_value[i];
		}
		q[j] = qhat;
	}
	return {q, shift_right(cryptanalysislib::first<N>(us), k)};
}

/// NOTE: this function will use a sginaficant amount of compiler ressources.
/// you may need to increase the constexpr steps via:
///    -fconstexpr-steps=99999999
/// \tparam N
/// \tparam T
/// \param n
/// \param k
/// \return
template<size_t N, typename T, const T n, const T k>
constexpr big_int<N, T> binomial() {
	static_assert(n * k < 1u<<20); // we need to enforce some limit
	static_assert(k <= n);

	const big_int<N, T> one = big_int<N, T>(1u);

	std::array<std::array<big_int<N, T>, k+1>, n+1> tmp;
	for (T i = 0; i <= n; ++i) {
		for (T j = 0; j <= std::min(i, k); ++j) {
			if ((j == 0) || (j == i)) {
				tmp[i][j] = one;
			} else {
				tmp[i][j] = big_int<N, T>{tmp[i-1][j-1] + tmp[i-1][j]};
			}
		}
	}

	return tmp[n][k];
}

template<typename T, size_t N1, size_t N2>
constexpr auto operator/(big_int<N1, T> a, big_int<N2, T> b) {
	return div(a, b).quotient;
}

template<typename T, size_t N1, size_t N2>
constexpr auto operator%(big_int<N1, T> a, big_int<N2, T> b) {
	return div(a, b).remainder;
}

template<typename T, size_t N1, size_t N2>
constexpr auto operator*(big_int<N1, T> a, big_int<N2, T> b) {
	return mul(a, b);
}

template<typename T, size_t N1, size_t N2>
constexpr auto operator+(big_int<N1, T> a, big_int<N2, T> b) {
	return add(a, b);
}

template<typename T, size_t N1, size_t N2>
constexpr auto operator-(big_int<N1, T> a, big_int<N2, T> b) {
	return sub(a, b);
}
#endif
