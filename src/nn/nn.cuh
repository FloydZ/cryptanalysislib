#include <cassert>
#include <cstdlib>
#include <curand_kernel.h>

#include "helper.cuh"
#include "warp_scan.cuh"

constexpr bool NN_EQUAL = false;
constexpr bool NN_LOWER = true;

/// popcount for 32 and 64 bit limbs
template<typename T>
__device__ __always_inline
uint32_t popc(const T v) {
	static_assert((sizeof(T) == 4) || (sizeof(T) == 8));
	if constexpr (sizeof(T) == 8) {
		return __popcll(v);
	} else {
		return __popc(v);
	}
}

template<typename T, const uint32_t d>
__device__ __always_inline
uint32_t compare(const T v) {
	static_assert((NN_EQUAL + NN_LOWER) == 1);

	if constexpr (NN_EQUAL) {
		return popc<T>(v) == d;
	}

	if constexpr (NN_LOWER) {
		return popc<T>(v) < (d+1u);
	}

	/// to make the compiler happy
	return 0;
}


///
/// initializes the cuRand states, one per block
__global__ void initRNG(curandStateXORWOW_t *randState) {
	const uint32_t bid = blockIdx.x;
	if (threadIdx.x == 0)
		curand_init(bid, 1, 0, &randState[bid]);
}



/// NOTE: e1,e2 is the size of the lists
/// NOTE: the current thread starts with element `start` of `L` and continues
/// 		with every `stride`-th element.
/// NOTE: if `iL` and `iR` are given, the positions written into `f1` and `f2`
/// 		are translated through them.
/// NOTE: if more than one solution exists, only the first one found is written
/// 		into `f1` and `f2`. Both must be init with -1.
template<typename T, const uint32_t nr_limbs,
	const uint32_t d>
__device__ void dbruteforce(T *L, T *R,
						   const size_t e1, const size_t e2,
						   size_t *f1, size_t *f2,
						   const size_t start, const size_t stride,
						   const uint32_t *iL=nullptr, const uint32_t *iR=nullptr) {
	static_assert(nr_limbs > 0);
	static_assert(sizeof(size_t) == sizeof(unsigned long long));

	size_t tid = start;
	T tmpL[nr_limbs];
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
				weight += popc<T>(tmpL[j] ^ R[i*nr_limbs + j]);
			}

			// highly unlikely
			if (weight == d) {
				const size_t pos1 = iL ? iL[tid] : tid;
				const size_t pos2 = iR ? iR[i] : i;
				if (atomicCAS((unsigned long long *)f1, (unsigned long long)-1, pos1) == (unsigned long long)-1) {
					(*f2) = pos2;
				}
			}
		}

		tid += stride;

		// NOTE: no synchronization needed: the threads do not share any
		// 	(shared) memory.
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
		(L, R, e1, e2, f1, f2, idx, blockDim.x*gridDim.x);
}

/// filters the `e` elements of `X`: an element survives if its limbs
/// `nr_limbs-lvl, ..., nr_limbs-lvl+depth-1` are each within weight `dk` of
/// the corresponding center in `z`. The first `size` survivors are written
/// together with their positions in `X` into `Xs` and `Xi`.
/// NOTE: must be called by all threads of the block.
/// \return the number of survivors, which may be bigger than `size`.
template<typename T, const uint32_t nr_limbs, const uint32_t dk,
		 const uint32_t lvl, const uint32_t size>
__device__ uint32_t dnn_filter(const T *X, const size_t e,
							   T *Xs, uint32_t *Xi,
							   const T *z, const uint32_t depth,
							   uint32_t *cnt) {
	constexpr uint32_t limb = nr_limbs-lvl;

	// all threads must have read the previous value of `cnt`
	__syncthreads();
	if (threadIdx.x == 0) {
		*cnt = 0;
	}
	__syncthreads();

	/// NOTE: all threads of a warp run the same number of iterations, so the
	/// 		warp wide functions below see the full warp.
	for (size_t base = 0; base < e; base += blockDim.x) {
		const size_t tid = base + threadIdx.x;
		uint32_t pred = tid < e;
		for (uint32_t j = 0; pred && (j < depth); j++) {
			pred = compare<T, dk>(X[tid*nr_limbs + limb + j] ^ z[j]);
		}

		// popcount yields the number of solutions within a warp
		const uint32_t pred_exc = warp_scan_exc_up_add_shuffled(pred);
		const uint32_t nr_sols = __popc(__ballot_sync(0xffffffff, pred));

		// reserve the space for the solutions of the whole warp at once
		uint32_t offset = 0;
		if ((threadIdx.x % 32u == 0u) && (nr_sols > 0)) {
			offset = atomicAdd(cnt, nr_sols);
		}
		offset = __shfl_sync(0xffffffff, offset, 0) + pred_exc;

		if (pred && (offset < size)) {
			// write back the partial solutions
			for (uint32_t i = 0; i < nr_limbs; i++) {
				Xs[offset*nr_limbs + i] = X[tid*nr_limbs + i];
			}
			Xi[offset] = tid;
		}
	}

	__syncthreads();
	return *cnt;
}

///
/// each block computes `N` leaves. For each leaf it draws random centers,
/// filters both lists into shared memory and bruteforces the survivors.
/// If the survivors do not fit into shared memory, the filter is applied on
/// one more limb, up to `lvl` limbs. Survivors which still do not fit are
/// dropped.
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__device__ void dnn_shared(T *L, T *R,
						   const size_t e1, const size_t e2,
						   size_t *f1, size_t *f2,
						   curandStateXORWOW_t *randState) {
	static_assert(lvl <= nr_limbs);
	static_assert(nr_limbs > 0);
	static_assert(lvl > 0);
	assert(blockDim.x % 32u == 0u);
	assert(e1 <= uint32_t(-1));
	assert(e2 <= uint32_t(-1));

	/// NOTE: 1024*32 is the maximal number of bytes we statically allocate
	/// as shared memory per block. Half of it is used for each list, and each
	/// surviving element is stored together with its position in the list.
	constexpr uint32_t size = (1024*32/2) / (nr_limbs*sizeof(T) + sizeof(uint32_t));
	__shared__ T Ls[size*nr_limbs];
	__shared__ T Rs[size*nr_limbs];
	__shared__ uint32_t Li[size];
	__shared__ uint32_t Ri[size];
	__shared__ T z[lvl];
	__shared__ uint32_t cnt;

	for (uint32_t n = 0; n < N; n++) {
		// all threads must be finished with the previous leaf
		__syncthreads();
		if (threadIdx.x == 0) {
			curandStateXORWOW_t state = randState[blockIdx.x];
			for (uint32_t j = 0; j < lvl; j++) {
				if constexpr (sizeof(T) == 8) {
					z[j] = (T(curand(&state)) << 32u) | curand(&state);
				} else {
					z[j] = curand(&state);
				}
			}
			randState[blockIdx.x] = state;
		}
		__syncthreads();

		uint32_t depth = 0, new_e1, new_e2;
		do {
			depth += 1;
			new_e1 = dnn_filter<T, nr_limbs, dk, lvl, size>(L, e1, Ls, Li, z, depth, &cnt);
			new_e2 = dnn_filter<T, nr_limbs, dk, lvl, size>(R, e2, Rs, Ri, z, depth, &cnt);
		} while (((new_e1 > size) || (new_e2 > size)) && (depth < lvl));

		dbruteforce
			<T, nr_limbs, d>
			(Ls, Rs, min(new_e1, size), min(new_e2, size), f1, f2,
			 threadIdx.x, blockDim.x, Li, Ri);
	}
}

/// \tparam T base limb type. Always either uint32_t or uint64_t
/// \tparam nr_limbs number of limbs representing one element in the lists
/// \tparam d weight to match on for the final element
/// \tparam dk weight to matcah on during the nn step on each limb
/// \tparam N number of leaves each block computes
/// \tparam lvl maximal number of limbs the nn step is applied on. 0 means
/// 		bruteforce.
/// \param L
/// \param R
/// \param e1
/// \param e2
/// \param f1 position of the final element in the left list. Must be init with -1
/// \param f2 position of the final element in the right list. Must be init with -1
/// \param randState one cuRand state per block, see `initRNG`
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__device__ void dnn(T *L, T *R,
				   const size_t e1, const size_t e2,
				   size_t *f1, size_t *f2,
				   curandStateXORWOW_t *randState) {
	static_assert(lvl <= nr_limbs);
	static_assert(nr_limbs > 0);

	/// break the recursion
	if constexpr (lvl == 0) {
		dbruteforce
			<T, nr_limbs, d>
			(L, R, e1, e2, f1, f2, idx, blockDim.x*gridDim.x);
		return;
	} else {
		/// TODO here the selection process for the correct nn algorithm
		/// 	- irgendwie hier die estimated list size berechnen und abhängig davon
		/// 		dann zwischen dierser implementierung mit shared mem und einer
		/// 		anderen implementierung die auf den eigentlichen listen arbeitet
		/// 		switchen.
		dnn_shared
			<T, nr_limbs, d, dk, N, lvl>
			(L, R, e1, e2, f1, f2, randState);
	}
}

/// global wrapper around the device function `dnn`, see there for the
/// template parameters and arguments.
template<typename T, const uint32_t nr_limbs,
	     const uint32_t d, const uint32_t dk, const uint32_t N,
		 const uint32_t lvl>
__global__ void nn(T *L, T *R,
				   const size_t e1, const size_t e2,
				   size_t *f1, size_t *f2,
				   curandStateXORWOW_t *randState) {
	dnn
		<T, nr_limbs, d, dk, N, lvl>
		(L, R, e1, e2, f1, f2, randState);
}
