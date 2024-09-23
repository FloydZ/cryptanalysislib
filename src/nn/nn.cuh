#include <cassert>
#include <curand_kernel.h>

#include "helper.cuh"
#include "warp_scan.cuh"

constexpr size_t BRUTEFORCE_SWITCH = 1024;
constexpr bool NN_EQUAL = false;
constexpr bool NN_LOWER = true;

template<typename T, const uint32_t d> 
__device__ __always_inline
uint32_t compare(const T v) {
	static_assert((NN_EQUAL + NN_LOWER) == 1);
	
	if constexpr (NN_EQUAL) {
		return __popc(v) == d;
	}

	if constexpr (NN_LOWER) {
		return __popc(v) < (d+1u);
	}

	/// to make the compiler happy
	return 0;
}


///
/// initializes the cuRand states
__global__ void initRNG(curandStateXORWOW_t *randState) {
	const uint32_t bid = blockDim.x;
	if (threadIdx.x == 0)
		curand_init(bid, 1, 0, &randState[bid]);
}



/// NOTE: e1,e2 is the size of the lists 
/// NOTE: the kernel only works if in expection at most only a single solution
/// 		can be found.
/// NOTE: nr threads cannot be bigger than e1 and e2
template<typename T, const uint32_t nr_limbs,
	const uint32_t d>
__device__ void dbruteforce(T *L, T *R,
						   const size_t e1, const size_t e2,
						   size_t *f1, size_t *f2) {

	const uint32_t dim = blockDim.x*gridDim.x;
	uint32_t tid = idx;
	ASSERT(dim <= e1);
	ASSERT(dim <= e2);
	ASSERT(blockDim.x % 32u == 0u);
	static_assert(nr_limbs > 0);


	uint32_t tmpL[nr_limbs];
	while (tid < e1) {
		// first load the current element into the register
		for (uint32_t j = 0; j < nr_limbs; j++) {
			tmpL[j] = L[tid*nr_limbs + j];
		}

		// loop over each element in the second list
		for (size_t i = 0; i < e2; i++) {
			// next compute the weight
			uint32_t weight = 0;
			for (uint32_t j = 0; j < nr_limbs; j++) {
				// TODO popc unabhängig machen von T
				weight += __popc(tmpL[j] ^ R[i*nr_limbs + j]);
			}
			
			// highly unlikely
			if (weight == d) {
				(*f1) = tid;
				(*f2) = i;
			}
		}

		tid += dim;

		// TODO check if important
		//__syncthreads();
		__syncwarp();
	}
}

///
/// just a global wrapper around the bruteforce function
template<typename T, const uint32_t nr_limbs,
	const uint32_t d>
__global__ void bruteforce(T *L, T *R,
						   const size_t e1, const size_t e2,
						   size_t *f1, size_t *f2) {
	dbruteforce
		<T, nr_limbs, d>
		(L, R, e1, e2, f1, f2);
}

///
///
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__device__ void dnn_shared_inner(T *L, T *R,
				   const size_t e1, const size_t e2,
				   size_t *f1, size_t *f2) {
	static_assert(lvl <= nr_limbs);
	static_assert(nr_limbs > 0);
	ASSERT(*f1 == size_t(-1));
	ASSERT(*f2 == size_t(-1));

	/// break the recursion and just bruteforce if there are no more limbs
	/// to apply the NN to.
	if constexpr (lvl == 0) {
		dbruteforce
			<T, nr_limbs, d>
			(L, R, e1, e2, f1, f2);
		return;
	} else {

	}
}

/// 
/// this outer function is needed, because otherwise the compiler would 
/// allocate in each recurion step the needed shared memory.
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__device__ void dnn_shared_outer(T *L, T *R,
				   				 const size_t e1, const size_t e2,
				   				 size_t *f1, size_t *f2) {
	static_assert(lvl <= nr_limbs);
	static_assert(nr_limbs > 0);
	static_assert(lvl > 0);
	ASSERT(*f1 == size_t(-1));
	ASSERT(*f2 == size_t(-1));
	
	/// NOTE: 1024*32 is the maximal number of bytes we can statically allocate
	/// as shared memory per block. One need to divide this by 4 to get the 
	/// maximal number of uint32_t and by 2 for each list.
	/// Thus 1024*32/4/2 is the maximal number of uint32_t to allocate for 
	/// each list.
	__shared__ uint32_t Ls[1024*32/4/2];
	__shared__ uint32_t Rs[1024*32/4/2];
	__shared__ uint8_t block_adder[32];
	
	// TODO make this block wide accessable
	size_t new_e1=0, new_e2=0;
	uint32_t tid = idx;
	const uint32_t dim = blockDim.x*gridDim.x;
	constexpr uint32_t limb = nr_limbs-lvl;
	const T z = 98273727; // TODO

	while (tid < e1) {
		uint32_t predL = compare<T, dk>(L[tid*nr_limbs + limb] ^ z);
		uint32_t predR = compare<T, dk>(R[tid*nr_limbs + limb] ^ z);
			
		// popcount should now yield the number of solutions within a warp
		// DEBUG
		//const uint32_t pred = compare<T, dk>(L[tid] ^ R[tid]);
		

		const uint32_t pred_excL = warp_scan_exc_up_add_shuffled(predL);
		const uint32_t nr_solsL = __popc(__ballot_sync(0xffffffff, predL));
		// printf("%02d %d %d %d\n", tid, pred, pred2, nr_sols);
		
		// after the exclusive sum within the warp, we need to compute
		// the exclusive sum within the block.
		const uint8_t warpId = threadIdx.x/32;
		if (threadIdx.x % 32 == 0) {
			block_adder[warpId] = nr_solsL;
		}

		// __syncthreads();
		if (threadIdx.x < 32) {
			 block_adder[threadIdx.x] = warp_scan_exc_up_add_shuffled(block_adder[threadIdx.x]);
			 // printf("%02d %d\n", threadIdx.x,block_adder[threadIdx.x]);
		}

		// this is important
		__syncthreads();
		
		if (predL) {
			// write back the partial solutions
			for (uint32_t i = 0; i < nr_limbs; i++) {
				Ls[(block_adder[warpId] + pred_excL)*nr_limbs + i] = L[tid*nr_limbs + limb];
			}
		}

		// right side
		const uint32_t pred_excR = warp_scan_exc_up_add_shuffled(predR);
		const uint32_t nr_solsR = __popc(__ballot_sync(0xffffffff, predR));
		if (threadIdx.x % 32 == 0) {
			block_adder[warpId] = nr_solsR;
		}

		if (threadIdx.x < 32) {
			 block_adder[threadIdx.x] = warp_scan_exc_up_add_shuffled(block_adder[threadIdx.x]);
		}

		// this is important
		__syncthreads();
		
		if (predR) {
			for (uint32_t i = 0; i < nr_limbs; i++) {
				Rs[(block_adder[warpId] + pred_excR)*nr_limbs + i] = R[tid*nr_limbs + limb];
			}
		}

		tid += dim;
	}

	/// in the case where not enough elements survived, just do the bruteforce.
	if (new_e1 < BRUTEFORCE_SWITCH && new_e2 < BRUTEFORCE_SWITCH) {
		dbruteforce
			<T, nr_limbs, d>
			(Ls, Rs, new_e1, new_e2, f1, f2);

		return;
	}

	for (uint32_t i = 0; i < N; i++) {
		/// in the case where enough elements survived, go to the next limb
		dnn_shared_inner
			<T, nr_limbs, d, dk, N, lvl-1>
			(Ls, Rs, new_e1, new_e2, f1, f2);
	}
}


///
///
/// \tparam T base limb type. Always either uint32_t or uint64_t
/// \tparam nr_limbs number of limbs representing one element in the lists
/// \tparam d weight to match on for the final element
/// \tparam dk weight to matcah on during the nn step on each limb
/// \tparam N number of leaves in the algorithm
/// \tparam lvl current limb/lvl the algorithm is.
/// \param L 
/// \param R 
/// \param e1 
/// \param e2
/// \param f1 position of the final element in the left list. Must be init with -1
/// \param f2 position of the final element in the right list. Must be init with -1
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__device__ void dnn(T *L, T *R,
				   const size_t e1, const size_t e2,
				   size_t *f1, size_t *f2) {
	static_assert(lvl <= nr_limbs);
	static_assert(nr_limbs > 0);
	ASSERT(*f1 == size_t(-1));
	ASSERT(*f2 == size_t(-1));

	/// break the recursion
	if constexpr (lvl == 0) {
		dbruteforce
			<T, nr_limbs, d>
			(L, R, e1, e2, f1, f2);
		return;
	} else {
		/// TODO here the selection process for the correct nn algorithm
		/// 	- irgendwie hier die estimated list size berechnen und abhängig davon
		/// 		dann zwischen dierser implementierung mit shared mem und einer
		/// 		anderen implementierung die auf den eigentlichen listen arbeitet
		/// 		switchen.
		dnn_shared_outer
			<T, nr_limbs, d, dk, N, lvl>
			(L, R, e1, e2, f1, f2);
	}
}

///
/// just a wrapper around the device function
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__global__ void nn(T *L, T *R,
				   const size_t e1, const size_t e2,
				   size_t *f1, size_t *f2) {
	dnn
		<T, nr_limbs, d, dk, N, lvl>
		(L, R, e1, e2, f1, f2);
}
