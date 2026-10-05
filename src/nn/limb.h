#ifndef CRYPTANALYSISLIB_NN_LIMB_H
#define CRYPTANALYSISLIB_NN_LIMB_H
#include <cstddef>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "alloc/alloc.h"
#include "simd/simd.h"
#include "algorithm/bits/popcount.h"

using namespace cryptanalysislib;

///
template<typename T,
 		 const uint32_t w>
class NNLimb {
	
	/// simple nearest neighbour function
	/// This returns the indices of solutions
	/// NOTE: Binary only
	/// \param out
	/// \param L left list
	/// \param R right list
	/// \param el size of the left list
	/// \param er size of the right list
	/// \param max maximal number of solution to found
	template<typename Index=uint32_t>
	static void nn(const Index *out[2],
	 			   const T *L,
	 			   const T *R,
				   const size_t el,
				   const size_t er,
				   const size_t max) noexcept {
		size_t ctr = 0;
		for (size_t i = 0; i < el; i++) {
			for (size_t j = 0; j < er; j++) {
				if (popcount::popcount(L[i] ^ R[i]) == w) {
					out[ctr][0] = i; 
					out[ctr][1] = j;
					ctr += 1;
					if (ctr == max) { return; }
				}
			}
		}
	}

	///
	template<typename Index=uint32_t,
	 		 const uint32_t u=4,
	 		 const uint32_t v=4>
	static void nn_bruteforce_simd_F2_32(const Index *out[2],
	 									 const T *L, const T *R,
										 const size_t el,
										 const size_t er,
										 const size_t max) noexcept {
		size_t ctr = 0;
		// TODO code von bruteforce_simd_256_32_uxv klauen
		// TODO: BM like nn version der gleichzeit auf verschieden limbs matched
	}

};

#endif // CRYPTANALYSISLIB_NN_LIMB_H
