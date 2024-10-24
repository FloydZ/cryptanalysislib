#ifndef CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
#define CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H

#include <cstdint>
#include <cstdlib>

class Combinations_Fq_Lexicographic {
	/// length to enumerate
	const uint32_t n;

	/// max value to enumerate
	const uint32_t q;


	Combinations_Fq_Lexicographic(const uint32_t n, const uint32_t q) :
	n(n), q(q) {}

	/// \param stack array of size
	/// \param sp current stack pointer. Do not think about this
	/// \param k max weight to enumerate
	/// \return the number of subsequent numbers which have hamming weight < k.
	uint64_t restricted_zeta_update_stack(uint32_t *stack, uint32_t *sp, const uint64_t k) {
		if (stack[*sp] == k) {
			uint32_t i = *sp + 1;

			// walk up
			while (stack[i] == k)
				i += 1;

			// update
			stack[i] += 1;
			const uint32_t val = stack[i];
			const uint32_t altered_sp = i;

			// walk down
			while (i > 0) {
				stack[i - 1] = val;
				i -= 1;
			}

			// fix up stack pointer
			*sp = 0;
			return (1ull << (altered_sp + 1)) - 1;
		} else {
			stack[*sp] += 1;
			return 1ull;
		}
	}
public:
};

/// lexicographic enumeration of p error position <= n
/// so it can be that certain posiiton occur twice within the error
/// \tparam n max size of each index
/// \tparam p number of error position
/// \tparam q (unused, needed for api compability)
template<const uint32_t n,
         const uint32_t p,
         const uint32_t q = 2>
class enumerate_t {
	using T = uint16_t;
	T idx[16] = {0};

	static_assert(q>=2);
	static_assert(p<=4);
	static_assert(n > p);

public:
	[[nodiscard]] constexpr size_t list_size() const noexcept {
		size_t ret = 1;
		for (uint32_t i = 0; i < p; i++) {
			ret	*= n;
		}
		return ret;
	}

	template<typename F>
	constexpr inline void enumerate(F &&f) noexcept {
		if constexpr (p == 0) {
			// catch for prange
			return;
		} else if constexpr (p == 1) {
			return enumerate1(idx, f);
		} else if constexpr (p == 2) {
			return enumerate2(idx, f);
		} else if constexpr (p == 3) {
			return enumerate3(idx, f);
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate1(T *idx,
										    F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			f(idx);
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate2(T *idx, F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				f(idx);
			}
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate3(T *idx,
											F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				for (idx[2] = idx[1] + 1; idx[2] < n; ++idx[2]) {
					f(idx);
				}
			}
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate4(T *idx,
											F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				for (idx[2] = idx[1] + 1; idx[2] < n; ++idx[2]) {
					for (idx[3] = idx[2] + 1; idx[3] < n; ++idx[3]) {
						f(idx);
					}
				}
			}
		}
	}

	/// enumerating <= p
	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumeratep(T *idx,
											F &&f) noexcept {
		for (uint32_t i = 0; i < p; i++) { idx[i] = 0; }
		while (true) {
			for (idx[0] = 0; idx[0] < n; idx[0]++) {
				f(idx);
			}

			uint32_t nsp = 0;
			while(idx[nsp] == (n-1)) {
				if (nsp == p-1) { goto __exit; }
				idx[nsp++] = 0;
				idx[nsp]++;
				f(idx);
			}
		}

		__exit:
		return;
	}
};
#endif//CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
