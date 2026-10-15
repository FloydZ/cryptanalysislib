#ifndef CRYPTANALYSISLIB_NN_LIMB_H
#define CRYPTANALYSISLIB_NN_LIMB_H
#include <cstddef>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "alloc/alloc.h"
#include "simd/simd.h"
#include "algorithm/bits/popcount.h"
#include "algorithm/countr.h"

/// nearest neighbour search on single limb elements: finds all pairs
/// (i, j) with `popcount(L[i] ^ R[j]) == w`.
/// \tparam T limb type
/// \tparam w weight to match
template<typename T,
 		 const uint32_t w>
class NNLimb {
public:
	/// simple nearest neighbour function
	/// This returns the indices of solutions
	/// NOTE: Binary only
	/// \param out[out]: out[k] = {i, j} for the k-th solution
	/// \param L left list
	/// \param R right list
	/// \param el size of the left list
	/// \param er size of the right list
	/// \param max maximal number of solution to found
	/// \return number of solutions written to `out`
	template<typename Index=uint32_t>
	static size_t nn(Index (*out)[2],
	 			     const T *L,
	 			     const T *R,
				     const size_t el,
				     const size_t er,
				     const size_t max) noexcept {
		size_t ctr = 0;
		if (max == 0) { return 0; }
		for (size_t i = 0; i < el; i++) {
			for (size_t j = 0; j < er; j++) {
				if (cryptanalysislib::popcount::popcount(T(L[i] ^ R[j])) == w) {
					out[ctr][0] = i;
					out[ctr][1] = j;
					ctr += 1;
					if (ctr == max) { return ctr; }
				}
			}
		}

		return ctr;
	}

	/// same as `nn`, but compares `u` left elements against `8*v` right
	/// elements with `uint32x8_t` (32 bit limbs only).
	/// NOTE: the order of the solutions differs from `nn`: for each block of
	/// 	`u` left elements, the right list is processed in blocks of `8*v`.
	/// NOTE: a version matching on several limbs at once (like the
	/// 	multi limb kernels in `nn.h`) is not implemented.
	/// \tparam u number of left elements per block
	/// \tparam v number of SIMD registers of right elements per block
	/// \return number of solutions written to `out`
	template<typename Index=uint32_t,
	 		 const uint32_t u=4,
	 		 const uint32_t v=4>
	static size_t nn_bruteforce_simd_F2_32(Index (*out)[2],
	 									   const T *L, const T *R,
										   const size_t el,
										   const size_t er,
										   const size_t max) noexcept {
		static_assert(sizeof(T) == 4u);
		static_assert(u > 0 && v > 0);
		using S = uint32x8_t;
		using L32 = typename S::limb_type;

		size_t ctr = 0;
		if (max == 0) { return 0; }

		const S wv = S::set1(w);
		size_t i = 0;
		for (; i + u <= el; i += u) {
			S li[u];
			for (uint32_t ui = 0; ui < u; ++ui) { li[ui] = S::set1(L32(L[i + ui])); }

			size_t j = 0;
			for (; j + 8u*v <= er; j += 8u*v) {
				S r[v];
				for (uint32_t vi = 0; vi < v; ++vi) {
					r[vi] = S::unaligned_load((const L32 *)(R + j + 8u*vi));
				}

				for (uint32_t ui = 0; ui < u; ++ui) {
					for (uint32_t vi = 0; vi < v; ++vi) {
						uint32_t m = S::eq(S::popcnt(S::xor_(li[ui], r[vi])), wv);
						while (m) {
							const uint32_t b = cryptanalysislib::countr_zero(m);
							out[ctr][0] = i + ui;
							out[ctr][1] = j + 8u*vi + b;
							ctr += 1;
							if (ctr == max) { return ctr; }
							m &= m - 1u;
						}
					}
				}
			}

			// tail of the right list
			for (uint32_t ui = 0; ui < u; ++ui) {
				for (size_t jj = j; jj < er; ++jj) {
					if (cryptanalysislib::popcount::popcount(T(L[i + ui] ^ R[jj])) == w) {
						out[ctr][0] = i + ui;
						out[ctr][1] = jj;
						ctr += 1;
						if (ctr == max) { return ctr; }
					}
				}
			}
		}

		// tail of the left list
		for (; i < el; ++i) {
			for (size_t j = 0; j < er; ++j) {
				if (cryptanalysislib::popcount::popcount(T(L[i] ^ R[j])) == w) {
					out[ctr][0] = i;
					out[ctr][1] = j;
					ctr += 1;
					if (ctr == max) { return ctr; }
				}
			}
		}

		return ctr;
	}
};

#endif // CRYPTANALYSISLIB_NN_LIMB_H
