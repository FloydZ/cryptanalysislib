#ifndef CRYPTANALYSISLIB_COMBINATION_CHASE_H
#define CRYPTANALYSISLIB_COMBINATION_CHASE_H

#include <array>
#include <cstdint>
#include <vector>

#include "helper.h"
#include "math/math.h"

#if __cplusplus > 201709L
/// functions/fields an enumerator must implement
/// \tparam Enumerator
template<typename Enumerator>
concept EnumeratorAble = requires(Enumerator e) {
	e.step();
};
#endif

/// \return  number of elements in the gray code
template<const uint32_t w, const uint32_t q>
constexpr static size_t compute_gray_size() noexcept {
	uint64_t sum = 1;
	for (uint64_t i = 0; i < w; i++) { sum *= q - 1; }

	// just make sure that we do not return zero.
	return std::max(size_t(sum), size_t(1ul));
}

/// needed to compute the list size before initializing the enumerator
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate: NOTE: can be zero, for Prange
/// \return list size
template<const uint32_t n, const uint32_t q, const uint32_t w>
constexpr size_t compute_combinations_fq_chase_list_size() {
	static_assert(n > w);
	static_assert(q > 1);
	size_t size = compute_gray_size<w, q>();

	// just make sure that we do not return zero.
	return std::max(size, size_t(1ul)) * bc(n, w);
}

/// \tparam T base limb type of the input const_array.
/// 			Most likely always `uint64_t`
/// \tparam n number of positions to enumerate
/// \tparam w weight to enumerate
/// \tparam start offset to start the enumeration process (in bits)
template<typename T,
         const uint32_t n,
         const uint32_t w,
         const uint32_t start = 0>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
class Combinations_Binary_Chase {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	idea taken from https://stackoverflow.com/questions/36451090/permutations-of-binary-number-by-swapping-two-bits-not-lexicographically
	 * 	algorithm 3
	 *
	 *  This is the data struct which is used in the BJMM/MMT algorithm.
	 */
	// data for the two functions: 'two_changes_binary_left_init', 'two_changes_binary_left_init'
	std::array<uint64_t, n> two_changes_binary_o;// offset from the left most position
	std::array<int64_t, n> two_changes_binary_d; // direction set bit is moving
	std::array<uint64_t, n> two_changes_binary_n;// length of current part of the sequence
	std::array<uint64_t, n> two_changes_binary_p;// current position of the bit in the current part
	uint64_t two_changes_binary_b = 0;           // how many permutations already processed
	bool init = true;                            // if set to true the next call to `left_step`
	                                             // will init the given pointer

	// number of bits in one limb
	constexpr static uint32_t RADIX = sizeof(T) * 8;


	/// computes a single round of the chase sequence
	/// \param b
	inline void left_round(const uint64_t b) noexcept {
		ASSERT(b < two_changes_binary_o.size());

		two_changes_binary_o[b] = two_changes_binary_o[b - 1] + two_changes_binary_d[b - 1] *
		                                                                (two_changes_binary_p[b - 1] % 2 ? two_changes_binary_n[b - 1] - 1 : two_changes_binary_p[b - 1] + 1);
		two_changes_binary_d[b] = two_changes_binary_d[b - 1] * (two_changes_binary_p[b - 1] % 2 ? -1 : 1);
		two_changes_binary_n[b] = two_changes_binary_n[b - 1] - two_changes_binary_p[b - 1] - 1;
		two_changes_binary_p[b] = 0;
	}

	/// \return
	template<const bool write = true>
	constexpr inline uint64_t left_write(T *A,
	                                     const uint32_t b,
	                                     const int bit) noexcept {
		ASSERT(b < two_changes_binary_o.size());
		uint64_t ret = start + two_changes_binary_o[b] + two_changes_binary_p[b] * two_changes_binary_d[b];
		if constexpr (write) { WRITE_BIT(A, ret, bit); }
		return ret;
	}


	// TODO rename
	// we need these little helpers, because M4RI does not implement any row
	// access functions, only ones for matrices.
	constexpr static inline void WRITE_BIT(T *v,
	                                       const size_t i,
	                                       const uint64_t b) noexcept {
		v[i / RADIX] = ((v[i / RADIX] & ~(1ull << (i % RADIX))) | (T(b) << (i % RADIX)));
	}

public:
	/// max length of the sequence
	constexpr static size_t chase_size = bc(n, w);

	///
	constexpr Combinations_Binary_Chase() noexcept {
		static_assert(n > w);
		reset();
	};

	/// resets the
	constexpr void reset() noexcept {
		two_changes_binary_o.fill(0);
		two_changes_binary_d.fill(0);
		two_changes_binary_n.fill(0);
		two_changes_binary_p.fill(0);

		two_changes_binary_b = 0;

		two_changes_binary_d[0] = 1;
		two_changes_binary_n[0] = n;

		init = true;
	}

	/// computes one step in the cache sequence
	/// \param A  input const_array = limbs needed to represent 'n' bits.
	/// \param init if True: if the first element in the cache sequence is going to be generated,
	/// 			  False: else
	/// \return false if the end of the sequence is reached
	///         true if the end of the sequence is NOT reached
	template<bool write = true>
	bool left_step(T *A,
	               uint16_t *pos1,
	               uint16_t *pos2) noexcept {
		uint16_t pos = 0;
		if (!init) {
			// cleanup of the previous round
			do {
				pos = left_write<write>(A, two_changes_binary_b, 0);
			} while (++two_changes_binary_p[two_changes_binary_b] > (two_changes_binary_n[two_changes_binary_b] + two_changes_binary_b - w) && two_changes_binary_b--);
		}

		init = false;
		*pos1 = pos;
		if (two_changes_binary_p[0] > n - w) {
			return false;
		}

		// this is the bit which will be set to one.
		pos = left_write<write>(A, two_changes_binary_b, 1);
		while (++two_changes_binary_b < w) {
			left_round(two_changes_binary_b);
			left_write<write>(A, two_changes_binary_b, 1);
			pos += two_changes_binary_d[two_changes_binary_b];
			if (pos == *pos1)
				pos += -two_changes_binary_d[two_changes_binary_b];
		}

		*pos2 = pos;
		if (two_changes_binary_p[0] > n - w) {
			return false;
		}

		two_changes_binary_b = w - 1ul;
		return true;
	}

	/// TODO make static
	/// \tparam write
	/// \param ret input/output const_array containing ``
	/// \param listsize
	/// \return
	template<bool write = true>
	constexpr void changelist(std::pair<uint16_t, uint16_t> *ret,// TODO change to reference
	                          const size_t listsize = 0) {
		const size_t size = listsize == 0 ? chase_size : listsize;

		left_step<write>(nullptr, &ret[0].first, &ret[0].second);
		for (size_t i = 0; i < size; ++i) {
			bool c = left_step<write>(nullptr, &ret[i].first, &ret[i].second);
			ASSERT(c == (i != size - 1u));
			ASSERT(ret[i].first < n);
			ASSERT(ret[i].second < n);
		}
	}

	/// NOTE: old function, only for testing.
	/// NOTE: normally you shouldnt use this function.
	/// This functions simply xors together two given rows `p` and `p_old` and finds the two positions where they differ
	/// \param p newly generated chase element. Output from `left_step`
	/// \param p_old  last generated element from the chase sequence.
	/// \param limbs Number of limbs of type T needed to represent a element of the chase sequence
	/// \param pos1 output: first bit position where 'p` and `p_old` differs
	/// \param pos2 output: second bit position
	static void __diff(const T *p,
	                   const T *p_old,
	                   const uint32_t limbs,
	                   uint16_t *pos1,
	                   uint16_t *pos2) noexcept {
		uint8_t sols = 0;                   // solution counter. Should be at most 2 if Chase generation is used.
		uint32_t sol;                       // bit position of the change
		uint16_t *sol_ptr[2] = {pos1, pos2};// easy access to the solution const_array

		for (uint32_t i = 0; i < limbs; ++i) {
			// get the diff of the current limb
			T x = p[i] ^ p_old[i];
			// loop as long we found ones in the limb. (Maybe we have two ones in on limb)
			while (x != 0 && sols < 2) {
				sol = ffsll(x) - 1;
				// clear the bit
				x ^= (uint64_t(1) << sol);

				// now check if the bit was already set in p_old. If so we know the new zero pos (pos1).
				// if not we know that the bit was set in p. So we know pos2. Na we ignore this now. Doesnt matter
				// where the zero or the one is.
				const uint64_t pos = i * RADIX + sol;
				*(sol_ptr[sols]) = pos;
				sols += 1;
			}

			// early exit.
			if (sols == 2) {
				break;
			}
		}
	}
};

/// This class enumerates a normal chase sequence of weight w. This means
/// all elements are enumerated which only differ in at most two positions.
/// On top of this sequence, a grey code will enumerated to enumerated over
/// all elementes in Fq
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate
template<uint32_t n, const uint32_t q, const uint32_t w>
class Combinations_Fq_Chase {
	/// max value to enumerate
	constexpr static uint32_t qm1 = q - 1;

	/// NOTE: this is the output container for `mixed_radix_grey`
	std::array<uint32_t, n> a;

	/// stuff for the mixed radix enumeration
	std::array<uint32_t, n + 1> f;
	std::array<uint32_t, n> s;// sentinel

	/// stuff for the chase sequence
	Combinations_Binary_Chase<uint64_t, n, w> chase;

public:
	constexpr static size_t chase_size = bc(n, w);

	// the -1u is needed, because we already enumerated the 1...10...0 vector
	// which is the first vector
	constexpr static size_t gray_size = compute_gray_size<w, q>() - 1u;

	// So here things become tricky. As we substracted one, we need to add one
	constexpr static size_t LIST_SIZE = chase_size * (gray_size + 1u);

	/// NOTE: this enumerates on a length w NOT on length n
	/// NOTE: you have to free this stuff yourself
	/// NOTE: the input must be of size sum_bc(w, w)*q**w
	/// NOTE: returns 0 on success
	/// NOTE: only enumerates to q-1
	constexpr int changelist_mixed_radix_grey(uint16_t *ret) noexcept {
		uint32_t j;
		size_t ctr = 0;

		while (ctr < gray_size) {
			j = f[0];
			f[0] = 0;
			ret[ctr++] = j;
			if (j == w) {
				return 0;
			}

			a[j] = (a[j] + 1) % (qm1);

			if (a[j] == s[j]) {
				s[j] = (s[j] - 1 + qm1) % qm1;
				f[j] = f[j + 1];
				f[j + 1] = j + 1;
			}
		}

		return 0;
	}

	/// \param ret input/output const_array containing ``
	/// \return nothing
	template<bool write = true>
	constexpr void changelist_chase(std::pair<uint16_t, uint16_t> *ret) {
		chase.template changelist<write>(ret);
	}

	///
	/// \param n length
	/// \param q field size
	/// \param w max hamming weight to enumerate
	constexpr Combinations_Fq_Chase() noexcept {
		/// init the restricted gray code
		for (uint32_t i = 0; i < n; ++i) {
			a[i] = 0;
			f[i] = i;
			s[i] = qm1 - 1;
		}
		f[n] = n;
	}
};

// TODO move somewhere usefull
// TODO absolutly not enumerating only weight p
// lexicographic enumeration
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
		return compute_combinations_fq_chase_list_size<n, q, p>();
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

	template<typename F>
	constexpr static inline void enumerate1(T *idx, F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			f(idx);
		}
	}

	template<typename F>
	constexpr static inline void enumerate2(T *idx, F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				f(idx);
			}
		}
	}

	template<typename F>
	constexpr static inline void enumerate3(T *idx, F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				for (idx[2] = idx[1] + 1; idx[2] < n; ++idx[2]) {
					f(idx);
				}
			}
		}
	}

	template<typename F>
	constexpr static inline void enumerate4(T *idx, F &&f) noexcept {
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
};

template<const uint32_t n,
         const uint32_t p,
         const uint32_t q = 2>
class chase_t {
	using T = uint16_t;
	// TODO reset und wie machen wir das wenn wir mehrerer solcher fks hintereinander callen
	T idx[16] = {0};

	static_assert(q >= 2);
	static_assert(p <= 3);

public:
	constexpr size_t list_size() const noexcept {
		return compute_combinations_fq_chase_list_size<n, q, p>();
	}


	///
	/// \tparam F
	/// \param idx
	/// \param f
	/// \return
	template<typename F>
	constexpr static inline void enumerate1(T *idx, F &&f) noexcept {
		for (idx[0] = p; idx[0] < n; ++idx[0]) {
			f(idx[0], idx[0] - 1);
		}
	}

	///
	/// \tparam F
	/// \param idx
	/// \param f
	/// \param off0
	/// \return
	template<typename F>
	constexpr static inline void enumerate2(T *idx, F &&f, const uint32_t off0 = 0) noexcept {
		for (idx[0] = off0; idx[0] < n;) {
			const uint32_t start = idx[0] == 0 ? p : idx[0] + 2;
			for (idx[1] = start; idx[1] < n; idx[1] += 1) {
				f(idx[1], idx[1] - 1);
			}

			idx[0] += 1;
			if (idx[0] >= n - 1) { break; }

			f(idx[0], idx[0] - 1);
			idx[1] -= 1;


			for (; idx[1] > idx[0] + 1; idx[1] -= 1) {
				f(idx[1], idx[1] - 1);
			}

			idx[0] += 1;
			if (idx[0] >= n - 1) { break; }
			idx[1] += 1;

			f(idx[0] + 1, idx[0] - 1);
		}
	}

	///
	/// \tparam F
	/// \param idx
	/// \param f
	/// \return
	template<typename F>
	constexpr static inline void enumerate3(T *idx, F &&f) noexcept {
		idx[1] = 1;
		idx[2] = 2;
		for (idx[0] = 0; idx[0] < n;) {
			enumerate2(idx + 1, f, idx[0] + 1);
			idx[0] += 1;
			if (idx[0] >= n - 1) { break; }
			f(idx[0], idx[0] - 1);
			idx[1] -= 1;

			for (; idx[2] > idx[0] + 2;) {
				for (; idx[1] > idx[0] + 1; idx[1] -= 1) {
					f(idx[1], idx[1] - 1);
				}

				f(idx[2], idx[2] - 1);
				idx[2] -= 1;
				idx[1] += 1;

				for (; idx[1] < idx[2]; idx[1] += 1) {
					f(idx[1], idx[1] - 1);
				}

				idx[1] -= 1;
				f(idx[1] - 1, idx[1] + 1);
				idx[2] -= 1;
				idx[1] -= 1;
			}

			f(idx[0], idx[0] + 3);
			idx[0] += 1;
			idx[1] += 1;
			idx[2] += 1;
		}
	}

	template<typename F>
	constexpr inline void enumerate(F &&f) noexcept {
		static_assert(n > p);
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


	inline void biject1(uint64_t a, uint16_t rows[1]) noexcept {
		ASSERT(a < n);
		rows[0] = a;
	}

	template<const uint32_t nn, const uint32_t pp>
	inline void biject2(uint64_t a, uint16_t rows[2]) noexcept {
		constexpr uint64_t n2 = 2 * nn;
		constexpr uint64_t n22 = 2 * nn - 2;
		constexpr uint64_t np = nn * pp;
		constexpr int64_t nnn = (1 - n2) * (1 - n2);

		/// https://www.wolframalpha.com/input?i=a+%3D%3D+%28sum+%28n+-+1-+k%29%2C+k%3D0+to+x%29%2C+solve+for+x
		const double pos1 = __builtin_sqrt(nnn - double(a << 3));
		const double pos = (pos1 + n2 - 3.) / 2.;
		const uint32_t t = np - __builtin_ceil(pos) - pp;
		rows[0] = t;

		const uint32_t t2 = (-1 * (t + 1) * (t - n22)) >> 1;

		const bool dir = t & 1ul;
		uint32_t t3 = !dir ? nn - (t2 - a) : t2 - a + t;
		rows[1] = t3;
	}

	template<const uint32_t nn, const uint32_t pp>
	inline void biject2_down(uint64_t a, uint16_t rows[2]) noexcept {
		constexpr uint64_t n2 = 2 * nn;
		constexpr uint64_t n22 = 2 * nn - 2;
		constexpr uint64_t np = nn * pp;
		constexpr int64_t nnn = (1 - n2) * (1 - n2);

		/// https://www.wolframalpha.com/input?i=a+%3D%3D+%28sum+%28n+-+1-+k%29%2C+k%3D0+to+x%29%2C+solve+for+x
		const double pos1 = __builtin_sqrt(nnn - double(a << 3));
		const double pos = (pos1 + n2 - 3.) / 2.;
		const uint32_t t = np - __builtin_ceil(pos) - pp;
		rows[0] = n - 2 - t;

		const uint32_t t2 = (-1 * (t + 1) * (t - n22)) >> 1;

		const bool dir = t & 1ul;
		uint32_t t3 = !dir ? nn - (t2 - a) : t2 - a + t;
		rows[1] = n - 2 - t3;
	}

	inline void biject(uint64_t a, uint16_t rows[p]) noexcept {
		static_assert(p <= 3, "not implemented");
		if constexpr (p == 1) {
			return biject1(a, rows);
		}

		if constexpr (p == 2) {
			biject2<n, p>(a, rows);
		}

		if constexpr (p == 3) {
			constexpr uint64_t np = bc(n - 1, p - 1);
			constexpr uint64_t n2 = 2 * np;
			// constexpr uint64_t n22 = 2 * np - 2;
			constexpr int64_t nnn = (1 - n2) * (1 - n2);
			const double pos1 = __builtin_sqrt(nnn - double(a << 3));
			const double pos = (pos1 + n2 - 3.) / 2.;
			const uint32_t t = 2 * np - __builtin_ceil(pos) - 2;
			rows[0] = t;

			// TODO not finished const uint32_t t2 = (-1 * (t + 1) * (t - n22)) >> 1;

			if (a % (np - 1) > 0 || a == 0) {
				if (t & 1u) {
					biject2_down<n - 1, p - 1>(a % np, rows + 1);
				} else {
					biject2<n - 1, p - 1>(a % np, rows + 1);
				}

				for (uint32_t i = 0; i < 2; ++i) {
					rows[i + 1] += 1;
				}
			}
		}
	}
};

/// based on: https://github.com/vvasseur/isd/blob/master/src/dumer.c
template<const uint32_t n, const uint32_t t>
class BinaryChaseEnumerator {
	static_assert(n > t);
	static_assert(t > 0);

	/// max length of the sequence
	/// The thing is: we will not create the first element in this list.
	constexpr static size_t chase_size = bc(n, t) - 1ull;

public:
	static constexpr inline size_t size() noexcept {
		return chase_size;
	}

	/*
	 * List all combinations of 't' elements of a set of 'n' elements.
	 *
	 * Generate a Chase's sequence: the binary representation of a combination and
	 * its successor only differ by two bits that are either consecutive of
	 * separated by only one position.
	 *
	 * See exercise 45 of Knuth's The art of computer programming volume 4A.
	 */
	static constexpr void changelist(std::vector<std::pair<uint16_t, uint16_t>> &ret,
									 const size_t listsize = 0) {
		const size_t size = listsize == 0 ? chase_size : listsize;
		ret.resize(size);
		if constexpr (t==1) {
			for (size_t i = 1; i <= size; ++i) {
				std::pair<uint16_t, uint16_t> tmp{i-1,i};
				ret[i-1] = tmp;
			}
			return;
		}

		size_t N = 0;
		uint16_t diff_pos = 0;
		uint16_t diff_len = 0;
		int32_t x;
		uint16_t c[t + 2];
		uint16_t z[t + 2];
		for (size_t j = 1; j <= t + 1; ++j) {
			z[j] = 0;
		}
		for (size_t j = 1; j <= t + 1; ++j) {
			c[j] = n - t - 1 + j;
		}
		/* r is the least subscript with c[r] >= r. */
		size_t r = 1;
		size_t j=0;

		uint16_t old_val = t-1u;
		uint16_t cur_val = t;
		uint16_t old_c[t + 2];

		while (true) {
			if ((N-1ul) == size) { return; }
			// for (size_t i = 1; i <= t; ++i) {
			// 	combinations[i - 1 + N * t] = c[i];
			// }
			// diff[N] = diff_pos + (diff_len - 1) * (n - 1);
			(void)diff_pos;
			(void)diff_len;
			if (j <= t) { cur_val = n-1-c[j]; }
			if (j <= t) { old_val = n-1-old_c[j]; }
			if (N > 0) {
				 std::pair<uint16_t, uint16_t> tmp{old_val, cur_val};
				 ret[N-1] = tmp;
			}
			memcpy(old_c, c, (t+2) * sizeof(uint16_t));

			++N;
			j = r;

		novisit:
			if (z[j]) {
				x = c[j] + 2;
				if (x < z[j]) {
					diff_pos = c[j];
					diff_len = 2;
					c[j] = x;
				} else if (x == z[j] && z[j + 1]) {
					diff_pos = c[j];
					diff_len = 2 - (c[j + 1] % 2);
					c[j] = x - (c[j + 1] % 2);
				} else {
					z[j] = 0;
					++j;
					if (j <= t)
						goto novisit;
					else
						goto exit;
				}
				if (c[1] > 0) {
					r = 1;
				} else {
					r = j - 1;
				}
			} else {
				x = c[j] + (c[j] % 2) - 2;
				if (x >= (int32_t) j) {
					diff_pos = x;
					diff_len = 2 - (c[j] % 2);
					c[j] = x;
					r = 1;
				} else if (c[j] == j) {
					diff_pos = j - 1;
					diff_len = 1;
					c[j] = j - 1;
					z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
					r = j;
				} else if (c[j] < j) {
					diff_pos = c[j];
					diff_len = j - c[j];
					c[j] = j;
					z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
					r = (j > 2) ? j - 1 : 1;
				} else {
					diff_pos = x;
					diff_len = 2 - (c[j] % 2);
					c[j] = x;
					r = j;
				}
			}
		}
	exit:
		return;
	}

};

#endif//CRYPTANALYSISLIB_CHASE_H
