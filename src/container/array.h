#ifndef SMALLSECRETLWE_ARRAY_H
#define SMALLSECRETLWE_ARRAY_H

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <assert.h>

#include "helper.h"


namespace cryptanalysislib {
	struct config_array {
		// force the container pointer to be aligned
		const uint32_t base_alignment = 16;

		// force each element in the container to be aligned.
		const uint32_t element_alignment = 1;

		constexpr config_array() {};
	};

	constexpr config_array std_config_array = config_array();


	template<typename T, size_t N, const config_array &config = std_config_array>
	class array {
	public:
		using const_iterator = const T *const;

		/// default constructor
		constexpr array() {}

		/// aggregate constructor
		template<typename ...Es>
		requires (sizeof...(Es) <= N)
		constexpr array(Es &&... e) : m_data{std::forward<Es>(e)...} {}

		/// array constructor
		template<size_t M>
		requires (M <= N)
		constexpr array(const T(&a)[M]) : array{a, std::make_index_sequence<M>()} {}

		/// construct from pointer and index sequence
		template<size_t ...Is>
		constexpr array(const T *p, std::index_sequence<Is...>) : m_data{p[Is]...} {}

		/// size, element access, begin, end
		[[nodiscard]] constexpr size_t size() const noexcept { return N; }

		/// access operator
		///		this functions perform a boundary check.
		///		throws an exception on bad boundary check
		/// \param n position to access.
		/// \return the element.
		constexpr T operator[](const size_t n) const {
			if (unlikely(n >= N)) {
				throw std::invalid_argument("invalid size");
			}

			return m_data[n];
		}

		/// same as the operator[]
		/// \param n index to accdess
		/// \return the element
		constexpr T at(const size_t n) const noexcept {
			return this->operator[](n);
		}

		constexpr const_iterator begin() const noexcept { return &m_data[0]; }

		constexpr const_iterator cbegin() const noexcept { return &m_data[0]; }

		constexpr const_iterator end() const noexcept { return &m_data[N]; }

		constexpr const_iterator cend() const noexcept { return &m_data[N]; }

		// map a function over an array (or two)
		template<typename F>
		constexpr auto map(F &&f) const -> array<decltype(f(T{})), N> {
			return map(std::forward<F>(f), std::make_index_sequence<N>());
		}

		template<typename F, typename U, size_t M>
		constexpr auto map(F &&f, const array<U, M> &rhs) const -> array<decltype(f(T{}, U{})), (N > M ? M : N)> {
			return map(std::forward<F>(f), rhs, std::make_index_sequence<(N > M ? M : N)>());
		}

		// array comparison
		template<size_t M>
		constexpr bool less(const array<T, M> &rhs) const {
			return less_r(rhs.begin(), rhs.end(), 0);
		}

		// push_back, push_front
		constexpr array<T, N + 1> push_back(const T &t) const {
			return push_back(t, std::make_index_sequence<N>());
		}

		constexpr array<T, N + 1> push_front(const T &t) const {
			return push_front(t, std::make_index_sequence<N>());
		}

		// concatenate two arrays
		template<size_t M>
		constexpr array<T, (M + N)> concat(const array<T, M> &a) const {
			return concat(a, std::make_index_sequence<N>(), std::make_index_sequence<M>());
		}

		///
		/// \tparam A
		/// \tparam B
		/// \return
		template<size_t A, size_t B>
		requires (A < B) && (B <= N)
		constexpr array<T, (B - A)> slice() const {
			return {&m_data[A], std::make_index_sequence<(B - A)>()};
		}

		// tail (omit first M elements) or init (omit last M elements)
		template<size_t M = 1>
		requires (M < N)
		constexpr array<T, (N - M)> tail() const {
			return slice<M, N>();
		}

		template<size_t M = 1>
		requires (M < N)
		constexpr array<T, (N - M)> init() const {
			return slice<0, N - M>();
		}

		// insert element at position
		template<size_t I, typename = void>
		struct inserter;

		template<size_t I>
		constexpr array<T, N + 1> insert(const T &t) const {
			return inserter<I>()(*this, t);
		}

		// mergesort
		template<size_t I, typename = void>
		struct sorter;

		template<size_t I, size_t J, typename = void>
		struct merger;

		///
		/// \tparam F
		/// \param f
		/// \return
		template<typename F>
		constexpr array<T, N> mergesort(F &&f) const {
			return sorter<N>::sort(*this, std::forward<F>(f));
		}

		///
		/// \tparam P
		/// \param p
		/// \return
		template<typename P>
		constexpr array<T, N> partition(P &&p) const {
			return mergesort(pred_to_less_t<P>(p));
		}

		///
		/// \tparam F
		/// \tparam Is
		/// \param f
		/// \return
		template<typename F, size_t ...Is>
		constexpr auto map(F &&f, std::index_sequence<Is...>) const
		-> array<decltype(f(T{})), N> {
			return array<decltype(f(T{})), N>{f(m_data[Is])...};
		}

		///
		/// \tparam F
		/// \tparam U
		/// \tparam M
		/// \tparam Is
		/// \param f
		/// \param rhs
		/// \return
		template<typename F, typename U, size_t M, size_t ...Is>
		constexpr auto map(F &&f, const array<U, M> &rhs, std::index_sequence<Is...>) const
		-> array<decltype(f(T{}, U{})), sizeof...(Is)> {
			return array<decltype(f(T{}, U{})), sizeof...(Is)>
					{f(m_data[Is], rhs.m_data[Is])...};
		}

		constexpr bool less_r(const T *b, const T *e, size_t i) const {
			return b == e ? false :         // other has run out
			       i == N ? true :          // this has run out
			       m_data[i] < *b ? true :  // elementwise less
			       less_r(b + 1, e, i + 1); // recurse
		}

		template<size_t ...Is>
		constexpr array<T, N + 1> push_back(const T &t, std::index_sequence<Is...>) const {
			return {m_data[Is]..., t};
		}

		template<size_t ...Is>
		constexpr array<T, N + 1> push_front(const T &t, std::index_sequence<Is...>) const {
			return {t, m_data[Is]...};
		}

		template<size_t ...Is, size_t ...Js>
		constexpr array<T, (sizeof...(Is) + sizeof...(Js))>
		concat(const array<T, sizeof...(Js)> &a,
		       std::index_sequence<Is...>, std::index_sequence<Js...>) const {
			return {m_data[Is]..., a[Js]...};
		}

		template<size_t ...Is>
		constexpr array<T, sizeof...(Is)> tail(std::index_sequence<Is...>) const {
			return {m_data[Is + N - sizeof...(Is)]...};
		}

		template<size_t ...Is>
		constexpr array<T, sizeof...(Is)> init(std::index_sequence<Is...>) const {
			return {m_data[Is]...};
		}

		// inserter for at front, in the middle somewhere, at end
		template<size_t I>
		struct inserter<I, typename std::enable_if<(I == 0)>::type> {
			constexpr array<T, N + 1> operator()(const array<T, N> &a, const T &t) const {
				return a.push_front(t, std::make_index_sequence<N>());
			}
		};
		
		// TODO not compiling under gcc
		// template<size_t I>
		// struct inserter<I, typename std::enable_if<(I > 0 && I < N)>::type> {
		// 	constexpr array<T, N + 1> operator()(const array<T, N> &a, const T &t) const {
		// 		return a.slice<0, I>()
		// 				.concat(a.slice<I, N>()
		// 				.push_front(t));
		// 	}
		// };

		template<size_t I>
		struct inserter<I, typename std::enable_if<(I == N)>::type> {
			constexpr array<T, N + 1> operator()(const array<T, N> &a, const T &t) const {
				return a.push_back(t, std::make_index_sequence<N>());
			}
		};

		// sorter: a 1-element array is sorted
		template<size_t I>
		struct sorter<I, typename std::enable_if<(I == 1)>::type> {
			template<typename F>
			constexpr static array<T, I> sort(const array<T, I> &a, F &&) {
				return a;
			}
		};

		// otherwise proceed by sorting each half and merging them
		template<size_t I>
		struct sorter<I, typename std::enable_if<(I > 1)>::type> {
			template<typename F>
			constexpr static array<T, I> sort(const array<T, I> &a, const F &f) {
				return merger<I / 2, I - I / 2>::merge(
						a.init(std::make_index_sequence<I / 2>()).mergesort(f),
						a.tail(std::make_index_sequence<I - I / 2>()).mergesort(f),
						f);
			}
		};

		// merger: zero-length arrays aren't a thing, so allow for each or both to
		// be of size 1
		template<size_t I, size_t J>
		struct merger<I, J,
				typename std::enable_if<(I == 1 && J == 1)>::type> {
			template<typename F>
			constexpr static array<T, I + J> merge(const array<T, I> &a, const array<T, J> &b,
			                                       const F &f) {
				return f(b[0], a[0]) ?
				       array<T, I + J>{b[0], a[0]} :
				       array<T, I + J>{a[0], b[0]};
			}
		};

		template<size_t I, size_t J>
		struct merger<I, J,
				typename std::enable_if<(I == 1 && J > 1)>::type> {
			template<typename F>
			constexpr static array<T, I + J> merge(const array<T, I> &a, const array<T, J> &b,
			                                       const F &f) {
				return f(b[0], a[0]) ?
				       merger<I, J - 1>::merge(a, b.tail(), f).push_front(b[0]) :
				       b.push_front(a[0]);
			}
		};

		template<size_t I, size_t J>
		struct merger<I, J,
				typename std::enable_if<(I > 1 && J == 1)>::type> {
			template<typename F>
			constexpr static array<T, I + J> merge(const array<T, I> &a, const array<T, J> &b,
			                                       const F &f) {
				return f(b[0], a[0]) ?
				       a.push_front(b[0]) :
				       merger<I - 1, J>::merge(a.tail(), b, f).push_front(a[0]);
			}
		};

		template<size_t I, size_t J>
		struct merger<I, J,
				typename std::enable_if<(I > 1 && J > 1)>::type> {
			template<typename F>
			constexpr static array<T, I + J> merge(const array<T, I> &a, const array<T, J> &b,
			                                       const F &f) {
				return f(b[0], a[0]) ?
				       merger<I, J - 1>::merge(a, b.tail(), f).push_front(b[0]) :
				       merger<I - 1, J>::merge(a.tail(), b, f).push_front(a[0]);
			}
		};

		// make a predicate into a comparison function suitable for sort
		template<typename P>
		struct pred_to_less_t {
			constexpr pred_to_less_t(P &&p) : m_p(std::forward<P>(p)) {}

			constexpr bool operator()(const T &a, const T &b) const {
				return m_p(b) ? false : m_p(a);
			}

			P m_p;
		};

	private:
		T m_data[N] = {};
	};

//// make an array from e.g. a string literal
//template<typename T, size_t N>
//constexpr auto make_array(const T(&a)[N]) -> array<T, N> {
//	return array<T, N>(a);
//}
//
//// make an array from some values: decay them so that we can easily have
//// arrays of string literals
//template<typename E, typename ...Es>
//constexpr auto make_array(E &&e, Es &&... es)
//-> array<std::decay_t<E>, 1 + sizeof...(Es)> {
//	return true ? array<std::decay_t<E>, 1 + sizeof...(Es)>(
//			std::forward<std::decay_t<E>>(e),
//			std::forward<std::decay_t<Es>>(es)...) :
//	       throw err::array_runtime_error;
//}
//
//// array equality
//template<typename T, size_t N>
//constexpr bool operator==(const array<T, N> &a, const array<T, N> &b) {
//	return equal(a.cbegin(), a.cend(), b.cbegin());
//}
//
//template<typename T, size_t N>
//constexpr bool operator!=(const array<T, N> &a, const array<T, N> &b) {
//	return !(a == b);
//}
//
//// array comparison
//template<typename T, size_t N, size_t M>
//constexpr bool operator<(const array<T, N> &a, const array<T, M> &b) {
//	return true ? a.less(b) :
//	       throw err::array_runtime_error;
//}
//
// transform: 1-arg (map) and 2-arg (zip) variants
	template<typename F, typename T, size_t N>
	constexpr auto transform(const array<T, N> &a, F &&f) -> decltype(a.map(f)) {
		return a.map(std::forward<F>(f));
	}

	template<typename F, typename T, size_t N, typename U, size_t M>
	constexpr auto transform(const array<T, N> &a, const array<U, M> &b, F &&f)
	-> decltype(a.map(f, b)) {
		return a.map(std::forward<F>(f), b);
	}

//// sort (mergesort)
//template<typename F, typename T, size_t N>
//constexpr array<T, N> sort(const array<T, N> &a, F &&lessFn) {
//	return true ? a.mergesort(std::forward<F>(lessFn)) :
//	       throw err::sort_runtime_error;
//}
//
//// partition
//template<typename P, typename T, size_t N>
//constexpr array<T, N> partition(const array<T, N> &a, P &&pred) {
//	return true ? a.partition(std::forward<P>(pred)) :
//	       throw err::partition_runtime_error;
//}
//
//// reverse
//namespace detail {
//	template<typename T, int ...Is>
//	constexpr array<T, sizeof...(Is)> reverse(
//			const array<T, sizeof...(Is)> &a, std::integer_sequence<int, Is...>) {
//		return array<T, sizeof...(Is)>{a.end()[-(Is + 1)]...};
//	}
//}
//
//template<typename T, size_t N>
//constexpr array<T, N> reverse(const array<T, N> &a) {
//	return true ? detail::reverse(a, std::make_integer_sequence<int, N>()) :
//	       throw err::reverse_runtime_error;
//}
}

namespace std {
	template<size_t n, typename T, size_t N>
	constexpr T get(const cryptanalysislib::array<T, N> &a) noexcept {
		static_assert(n < N);
		return a[n];
	}
}

#endif //SMALLSECRETLWE_ARRAY_H