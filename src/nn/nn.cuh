#include <cassert>
#include <cstdlib>
#include <curand_kernel.h>

#include "helper.cuh"
#include "warp_scan.cuh"
// NOTE: the 1 bit tensor core operations need sm_75+
#ifdef NN_CUDA_TENSOR
#include "nn/nn_tensor.cuh"
#endif

/// GPU implementation of the closest pair algorithm of Esser, Kübler and
/// Zweydinger, https://arxiv.org/abs/2102.02597, Algorithm 1 (Closest-Pair),
/// with the "at most δk" bucket criterion of its Section 5 (`NN_LOWER`, as in
/// `NN<config>` of nn/nn.h; `NN_EQUAL` is the exact criterion of Algorithm 1):
///
///   Closest-Pair(L, R):  stack := [(L, R, level r)]
///     pop (A, B, level):
///       level > 0: N times: z <- rng, A' := {x in A : wt((x_limb ^ z) & mask_k) <= dk}
///                                      B' := {y in B : wt((y_limb ^ z) & mask_k) <= dk}
///                                      push (A', B', level - 1)
///       level = 0: bruteforce: all (x, y) in A x B with wt(x ^ y) <= d
///
/// `limb` is limb `r - level` of the element (limb 0 at the root), `mask_k`
/// keeps its low `k` bits (one block B_{i,r} of the paper is one limb). The
/// tree has branching factor N and depth r:
///
///            (L, R)                               level r, limb 0
///         z_1 /  | ... \ z_N
///     (L', R') (L', R') ... (L', R')              level r-1, limb 1
///       / | \     / | \      / | \
///      ...                                        ...
///     leaves: bruteforce A x B                    level 0
///
/// Two implementations:
///   1. `nn` (flat): each CTA computes `N` leaves on its own, each with
///      fresh centers for all limbs: filter both complete lists into shared
///      memory, bruteforce there (`dnn_shared`).
///   2. `nn_tree_root` + `nn_tree_child` (tree, as `NN::simd_nn_internal`):
///      level r as one wide kernel, each of the N*N (root, child) subtrees as a
///      work item of a persistent CTA. With `NN_CUDA_TENSOR`, the roots below
///      the bruteforce threshold are bruteforced on the tensor cores by
///      `nn_tree_gather`, `nn_tree_tensor` and `nn_tree_tensor_result`.
///
/// Both implementations stop at the first pair found; the paper collects
/// all pairs. The lists are `nr_limbs` limbs of type `T` per element,
/// stored contiguously (element i at L[i*nr_limbs .. (i+1)*nr_limbs)).
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


/// initializes the cuRand states of the `nn` kernel, one per CTA: state `b`
/// is seeded with `b`, so every launch of `nn` after the same `initRNG` draws
/// the same centers.
/// \param randState `gridDim.x` states in device memory
///
/// grid/block: gridDim.x = number of CTAs the following `nn` launch uses,
/// blockDim.x arbitrary (only thread 0 of each CTA works), e.g.
///
///   curandStateXORWOW_t *randState;
///   cudaMalloc(&randState, sizeof(curandStateXORWOW_t) * blocks);
///   initRNG<<<blocks, 1>>>(randState);
///   nn<T, nr_limbs, d, dk, N, lvl><<<blocks, 256>>>(L, R, e1, e2, f1, f2, randState);
__global__ void initRNG(curandStateXORWOW_t *randState) {
	const uint32_t bid = blockIdx.x;
	if (threadIdx.x == 0)
		curand_init(bid, 1, 0, &randState[bid]);
}



/// naive search (level 0 of the tree): thread `start` compares the rows
/// `start`, `start + stride`, ... of `L` against all of `R`:
///
///                 R:  0   1   2   3  ...  e2-1
///   L[start]         x   x   x   x  ...   x      weight = sum_j popc(L[i][j] ^ R[.][j])
///   L[start+stride]  x   x   x   x  ...   x      weight <= d  ->  (f1, f2) once
///   ...
///
/// The row of `L` is kept in registers, `R` is streamed; if `R` is in shared
/// memory all threads read the same element at once (broadcast).
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
			// NOTE: `<= d` as `NN::FINAL_SOL_WEIGHT`, was `== d`
			if (weight <= d) {
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

/// global wrapper around `dbruteforce`: every thread of the grid takes the
/// rows `tid`, `tid + gridDim.x*blockDim.x`, ... of `L`.
/// \tparam T limb type, uint32_t or uint64_t
/// \tparam nr_limbs limbs per element
/// \tparam d maximal weight of a solution: wt(L[i] ^ R[j]) <= d
/// \param L, R the lists in device (or managed) memory
/// \param e1, e2 number of elements of `L`, `R`
/// \param f1, f2 output: positions of the first solution found, must be init
/// 		with -1 (e.g. `cudaMemset(f1, 0xff, sizeof(size_t))`)
///
/// grid/block: blockDim.x arbitrary (a multiple of 32), gridDim.x =
/// ceil(e1 / blockDim.x) gives one row of `L` per thread; fewer CTAs
/// grid-stride. E.g. for 2^20 x 2^20 elements of 256 bits:
///
///   bruteforce<uint32_t, 8, 20><<<(e1 + 1023) / 1024, 1024>>>(L, R, e1, e2, f1, f2);
///
/// NOTE: ~930 Gpairs/s on a RTX 5090 for n=128, |L| = |R| = 2^17 (the tensor
/// 		core kernels of nn/nn_tensor.cuh are faster and need less rows).
template<typename T, const uint32_t nr_limbs,
	const uint32_t d>
__global__ void bruteforce(T *L, T *R,
						   const size_t e1, const size_t e2,
						   size_t *f1, size_t *f2) {
	dbruteforce
		<T, nr_limbs, d>
		(L, R, e1, e2, f1, f2, threadIdx.x + blockIdx.x*blockDim.x, blockDim.x*gridDim.x);
}

/// filters the `e` elements of `X`: an element survives if its limbs
/// `nr_limbs-lvl, ..., nr_limbs-lvl+depth-1` are each within weight `dk` of
/// the corresponding center in `z`. The first `size` survivors are written
/// together with their positions in `X` into `Xs` and `Xi`.
///
///   X (global):   x0   x1   x2   x3   x4   x5  ...  x_{e-1}
///                  |    |    |    |    |    |
///   thread:        0    1    2    3    4    5  ...  (one element per thread
///   pred:          0    1    0    0    1    0  ...   and iteration)
///                       \______________/
///                        warp: ballot -> nr_sols, exclusive scan -> rank
///                        lane 0: offset = atomicAdd(cnt, nr_sols)
///                                 |
///   Xs (shared):  [ x1 | x4 | ... ]    Xi: [ 1 | 4 | ... ]    cnt = #survivors
///
/// pred for `depth` limbs:  wt(x[limb+j] ^ z[j]) <= dk  for all j < depth
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

/// flat variant of the tree: each CTA computes `N` leaves. For each leaf it
/// draws random centers, filters both lists into shared memory and
/// bruteforces the survivors. If the survivors do not fit into shared
/// memory, the filter is applied on one more limb, up to `lvl` limbs.
/// Survivors which still do not fit are dropped.
///
///   per CTA, N times:
///     z[0..lvl) <- rng                       (thread 0, shared)
///     depth = 1:  Ls, Li <- filter(L, z[0])          Rs, Ri <- filter(R, z[0])
///     too many?   depth = 2:  filter(L, z[0], z[1])  filter(R, z[0], z[1])
///     ...         (up to depth = lvl, i.e. limbs nr_limbs-lvl .. nr_limbs-1)
///     bruteforce  Ls x Rs in shared memory, positions via Li, Ri
///
/// Compared to the tree, a leaf of depth `depth` costs `depth` scans of the
/// complete lists instead of one scan of the parent's lists, but needs no
/// buffers in global memory: a leaf is a path root -> leaf with fresh centers.
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
			(L, R, e1, e2, f1, f2, threadIdx.x + blockIdx.x*blockDim.x, blockDim.x*gridDim.x);
		return;
	} else {
		/// selection: the bruteforce on the lists costs `e1*e2` comparisons,
		/// the nn filters both lists for each of the `N` leaves of each block.
		/// The bruteforce is exact, so it is used whenever it is not more
		/// expensive. Uniform over all blocks, i.e. all blocks take the same branch.
		const uint64_t bruteforce_cost = uint64_t(e1) * uint64_t(e2);
		const uint64_t nn_cost = uint64_t(gridDim.x) * N * (uint64_t(e1) + uint64_t(e2));
		if (bruteforce_cost <= nn_cost) {
			dbruteforce
				<T, nr_limbs, d>
				(L, R, e1, e2, f1, f2, threadIdx.x + blockIdx.x*blockDim.x, blockDim.x*gridDim.x);
			return;
		}

		dnn_shared
			<T, nr_limbs, d, dk, N, lvl>
			(L, R, e1, e2, f1, f2, randState);
	}
}

/// global wrapper around the device function `dnn`, see there for the
/// template parameters and arguments: the flat variant, every CTA computes
/// `N` leaves (or the whole grid bruteforces, if that is cheaper).
///
/// grid/block: blockDim.x a multiple of 32 (asserted), e.g. 256; gridDim.x
/// CTAs compute gridDim.x * N leaves in total, `randState` must hold
/// gridDim.x states (see `initRNG`). Shared memory: 32 KiB static per CTA.
/// E.g. 128 CTAs x N = 4 leaves on 64 bit elements (2 limbs), dk = 11 on
/// each limb, final weight d = 4:
///
///   initRNG<<<128, 1>>>(randState);
///   nn<uint32_t, 2, 4, 11, 4, 2><<<128, 256>>>(L, R, e1, e2, f1, f2, randState);
///   cudaDeviceSynchronize();      // *f1 != -1 <=> a pair was found
///
/// See tests/nn/cuda.cu.
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

/// filters the `e` elements `X[I[i]]` (`X[i]` if `I == nullptr`): an element
/// survives if the lower `k` bits of its `limb`-th `LT` limb are within
/// weight `dk` of `z`. The positions in `X` of the survivors are written into `O`.
/// Same as `NN::simd_sort_nn_on32/64`.
///
///   I (positions):  [ 17 | 3 | 42 | 8 | ... ]      one per thread
///                      |    |    |    |
///   X[I[i]], limb:   x17  x3  x42  x8           (gather of one LT limb)
///   pred:             1    0    1    0           wt((x ^ z) & mask) <= dk
///                      \__________/
///                       warp: ballot + scan, one atomicAdd(cnt) per warp
///   O (positions):  [ 17 | 42 | ... ]            cnt = #survivors
///
/// Only positions are moved, never elements: the next level gathers again.
/// NOTE: must be called by all threads of the block.
/// \return the number of survivors
template<typename T, typename LT, const uint32_t nr_limbs,
		 const uint32_t k, const uint32_t dk, const uint32_t limb>
__device__ uint32_t dnn_tree_filter(const T *X, const uint32_t *I, const uint32_t e,
									uint32_t *O, const LT z, uint32_t *cnt) {
	constexpr uint32_t bits = sizeof(LT) * 8u;
	constexpr LT mask = k >= bits ? LT(-1) : LT((LT(1) << k) - 1u);
	static_assert(limb < (nr_limbs * sizeof(T) / sizeof(LT)));

	// all threads must have read the previous value of `cnt`
	__syncthreads();
	if (threadIdx.x == 0) {
		*cnt = 0;
	}
	__syncthreads();

	/// NOTE: all threads of a warp run the same number of iterations
	for (uint32_t base = 0; base < e; base += blockDim.x) {
		const uint32_t tid = base + threadIdx.x;
		uint32_t pos = 0, pred = 0;
		if (tid < e) {
			pos = I ? I[tid] : tid;
			const LT x = ((const LT *)(X + size_t(pos) * nr_limbs))[limb];
			pred = compare<LT, dk>((x ^ z) & mask);
		}

		const uint32_t pred_exc = warp_scan_exc_up_add_shuffled(pred);
		const uint32_t nr_sols = __popc(__ballot_sync(0xffffffff, pred));

		// reserve the space for the survivors of the whole warp at once
		uint32_t offset = 0;
		if ((threadIdx.x % 32u == 0u) && (nr_sols > 0)) {
			offset = atomicAdd(cnt, nr_sols);
		}
		offset = __shfl_sync(0xffffffff, offset, 0) + pred_exc;

		if (pred) {
			O[offset] = pos;
		}
	}

	__syncthreads();
	return *cnt;
}

/// compares all pairs `(L[IL[i]], R[IR[j]])`. The positions of the first pair
/// with weight <= `d` are written into `f1` and `f2`, both must be init with -1.
/// Both lists are loaded in tiles of `tile` elements into shared memory, each
/// pair of tiles is compared by `dbruteforce` (same as `dnn_shared`).
///
///                 R[IR]:  tile 0    tile 1    tile 2  ...
///   L[IL]: tile 0        [Ls x Rs] [Ls x Rs] [Ls x Rs]      Ls loaded once per
///          tile 1        [Ls x Rs] [Ls x Rs] [Ls x Rs]      row of tiles, Rs per
///          ...                                              tile pair
///
/// In a tile pair thread `t` holds row `t` of `Ls` and loops over `Rs`.
/// NOTE: must be called by all threads of the block.
/// NOTE: only `t1 <= tile` threads work per tile pair: leaves with few
/// 		elements use the CTA badly, see the NOTE at `nn_tree_child`.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t tile = 256>
__device__ void dnn_tree_bruteforce(const T *L, const T *R,
									const uint32_t *IL, const uint32_t *IR,
									const uint32_t e1, const uint32_t e2,
									size_t *f1, size_t *f2) {
	__shared__ T Ls[tile * nr_limbs], Rs[tile * nr_limbs];
	__shared__ uint32_t Li[tile], Ri[tile];

	for (uint32_t i = 0; i < e1; i += tile) {
		const uint32_t t1 = (e1 - i) < tile ? (e1 - i) : tile;
		for (uint32_t j = 0; j < e2; j += tile) {
			const uint32_t t2 = (e2 - j) < tile ? (e2 - j) : tile;

			// all threads must be finished with the previous tiles
			__syncthreads();
			for (uint32_t a = threadIdx.x; a < tile; a += blockDim.x) {
				// NOTE: the left tile is the same for all right tiles
				if ((j == 0) && (a < t1)) {
					Li[a] = IL[i + a];
					for (uint32_t l = 0; l < nr_limbs; l++) {
						Ls[a*nr_limbs + l] = L[size_t(Li[a])*nr_limbs + l];
					}
				}
				if (a < t2) {
					Ri[a] = IR[j + a];
					for (uint32_t l = 0; l < nr_limbs; l++) {
						Rs[a*nr_limbs + l] = R[size_t(Ri[a])*nr_limbs + l];
					}
				}
			}
			__syncthreads();

			dbruteforce<T, nr_limbs, d>(Ls, Rs, t1, t2, f1, f2, threadIdx.x, blockDim.x, Li, Ri);
		}
	}
}

/// one node on level `level` of the NN tree. Same as `NN::simd_nn_internal`:
/// filters both lists with a new random center on limb `r - level`, and
/// either bruteforces the survivors or recurses `N` times one level down.
///
///   level    limb      input lists        output lists (positions)
///   r        0         L, R               RL[root], RR[root]     nn_tree_root
///   r-1      1         IL, IR             BL[r-2], BR[r-2]       dnn_tree<r-1>
///   ...
///   1        r-1       BL[1], BR[1]       BL[0], BR[0]           dnn_tree<1>
///   0        -         bruteforce IL x IR                        dnn_tree<0>
///
///   dnn_tree<level>(IL, IR):
///     z <- rng; OL <- filter(IL), OR <- filter(IR)
///     |OL| = 0 or |OR| = 0           -> return           (dead subtree)
///     |OL| < bf or |OR| < bf         -> bruteforce(OL x OR)
///     else N times: dnn_tree<level-1>(OL, OR), stop as soon as f1 != -1
///
/// \param IL, IR positions of the elements of this node, nullptr for all
/// \param BL, BR `r` buffers of size `list_size`, one per level
/// NOTE: must be called by all threads of the block.
template<typename T, typename LT, const uint32_t nr_limbs,
		 const uint32_t d, const uint32_t dk, const uint32_t k,
		 const uint32_t N, const uint32_t r, const uint32_t level>
__device__ void dnn_tree(const T *L, const T *R,
						 const uint32_t *IL, const uint32_t *IR,
						 const uint32_t e1, const uint32_t e2,
						 uint32_t *BL, uint32_t *BR, const size_t list_size,
						 const uint32_t bf, size_t *f1, size_t *f2,
						 curandStatePhilox4_32_10_t *state, LT *z,
						 uint32_t *cnt, uint32_t *stop) {
	if constexpr (level == 0) {
		dnn_tree_bruteforce<T, nr_limbs, d>(L, R, IL, IR, e1, e2, f1, f2);
	} else {
		// all threads must have read the previous center
		__syncthreads();
		if (threadIdx.x == 0) {
			if constexpr (sizeof(LT) == 8) {
				*z = (LT(curand(state)) << 32u) | curand(state);
			} else {
				*z = curand(state);
			}
		}
		__syncthreads();

		uint32_t *OL = BL + (level - 1u) * list_size;
		uint32_t *OR = BR + (level - 1u) * list_size;
		const uint32_t new_e1 = dnn_tree_filter<T, LT, nr_limbs, k, dk, r - level>(L, IL, e1, OL, *z, cnt);
		const uint32_t new_e2 = dnn_tree_filter<T, LT, nr_limbs, k, dk, r - level>(R, IR, e2, OR, *z, cnt);

		/// early exit if we filtered everything out
		if ((new_e1 == 0) || (new_e2 == 0)) { return; }

		if ((new_e1 < bf) || (new_e2 < bf)) {
			dnn_tree_bruteforce<T, nr_limbs, d>(L, R, OL, OR, new_e1, new_e2, f1, f2);
			return;
		}

		for (uint32_t i = 0; i < N; i++) {
			dnn_tree<T, LT, nr_limbs, d, dk, k, N, r, level - 1u>
				(L, R, OL, OR, new_e1, new_e2, BL, BR, list_size, bf, f1, f2, state, z, cnt, stop);

			// NOTE: `f1` is read by a single thread, s.t. all threads break
			__syncthreads();
			if (threadIdx.x == 0) {
				*stop = (*(volatile size_t *)f1) != size_t(-1);
			}
			__syncthreads();
			if (*stop) { break; }
		}
	}
}

/// Esser, Kuebler, Zweydinger NN on the GPU, step 1 of 2: level `r` of the
/// NN tree, i.e. the filter of the `N` roots of `NN::run`. Block `(x, y)`
/// filters the slice `x` of both lists with the center of root `y` and
/// appends the positions of the survivors to `RL + y*list_size` and
/// `RR + y*list_size`, their numbers to `cnt[2*y]` and `cnt[2*y + 1]`.
///
///   grid (x, y): y = root, x = slice of both lists (grid-stride)
///
///   L: [ slice 0 | slice 1 | slice 2 | ... ]   every thread tests one element
///   R: [ slice 0 | slice 1 | slice 2 | ... ]   per iteration against z_y
///            |         |
///            v         v     (one atomicAdd on cnt[2y] / cnt[2y+1] per warp)
///   RL + y*list_size: [ positions of the survivors of L, in warp order ]
///   RR + y*list_size: [ positions of the survivors of R, in warp order ]
///
/// \tparam T limb type of the lists
/// \tparam LT limb type to filter on: uint32_t if `k <= 32`, else uint64_t
/// \tparam nr_limbs limbs of type `T` per element
/// \tparam dk weight bound of the filter
/// \tparam k number of bits of the limb the filter uses (low bits)
/// \param L, R the lists in device memory
/// \param e1, e2 number of elements, < 2^32
/// \param RL, RR output: `N` index buffers of `list_size` entries each
/// \param cnt 2 counters per root, must be zero
/// \param list_size distance between the index buffers of two roots, >= e1, e2
/// \param seed the center of root `y` is drawn from the Philox subsequence `y`
///
/// grid/block: blockDim.x a multiple of 32 (asserted), e.g. 256; gridDim.y = N
/// (the roots), gridDim.x = number of slices, e.g. ~4 elements per thread:
///
///   cudaMemsetAsync(cnt, 0, (2 * N + 1) * sizeof(uint32_t));
///   const size_t e = max(e1, e2);
///   const dim3 grid((e + 4 * 256 - 1) / (4 * 256), N);
///   nn_tree_root<uint32_t, uint32_t, 8, dk, 32><<<grid, 256>>>
///       (L, R, e1, e2, RL, RR, cnt, list_size, seed);
///
/// See `NN::cuda_launch` in nn/nn.h for the complete sequence of launches.
template<typename T, typename LT, const uint32_t nr_limbs,
		 const uint32_t dk, const uint32_t k>
__global__ void nn_tree_root(const T *L, const T *R,
							 const uint32_t e1, const uint32_t e2,
							 uint32_t *RL, uint32_t *RR, uint32_t *cnt,
							 const size_t list_size, const uint64_t seed) {
	constexpr uint32_t bits = sizeof(LT) * 8u;
	constexpr LT mask = k >= bits ? LT(-1) : LT((LT(1) << k) - 1u);
	assert(blockDim.x % 32u == 0u);

	// NOTE: every thread draws the same center of its root
	const uint32_t root = blockIdx.y;
	curandStatePhilox4_32_10_t state;
	curand_init(seed, root, 0, &state);
	LT z;
	if constexpr (sizeof(LT) == 8) {
		z = (LT(curand(&state)) << 32u) | curand(&state);
	} else {
		z = curand(&state);
	}

	const uint32_t lane = threadIdx.x % 32u;
	for (uint32_t list = 0; list < 2; list++) {
		const T *X = list ? R : L;
		const uint32_t e = list ? e2 : e1;
		uint32_t *O = (list ? RR : RL) + root * list_size;

		/// NOTE: all threads of a warp run the same number of iterations
		const uint32_t stride = gridDim.x * blockDim.x;
		for (uint32_t base = blockIdx.x * blockDim.x; base < e; base += stride) {
			const uint32_t tid = base + threadIdx.x;
			uint32_t pred = 0;
			if (tid < e) {
				const LT x = ((const LT *)(X + size_t(tid) * nr_limbs))[0];
				pred = compare<LT, dk>((x ^ z) & mask);
			}

			// reserve the space for the survivors of the whole warp at once
			const uint32_t ballot = __ballot_sync(0xffffffff, pred);
			uint32_t offset = 0;
			if ((lane == 0) && ballot) {
				offset = atomicAdd(cnt + 2*root + list, __popc(ballot));
			}
			offset = __shfl_sync(0xffffffff, offset, 0) + __popc(ballot & ((1u << lane) - 1u));

			if (pred) {
				O[offset] = tid;
			}
		}
	}
}

/// Esser, Kuebler, Zweydinger NN on the GPU, step 2 of 2: the `N` children of
/// each of the `N` roots computed by `nn_tree_root`. Same as the loop over
/// `i` in `NN::simd_nn_internal<r>`, but each (root, child) pair is a work
/// item, which the (persistent) blocks fetch from `work`.
///
///   work items:  item = root * N + child,  item < N*N
///
///   CTA b:  loop:  item <- atomicAdd(work);  stop if f1 != -1 or item >= N*N
///                  IL = RL[root], IR = RR[root]  (e1 = cnt[2 root], e2 = cnt[2 root + 1])
///                  e1 = 0 or e2 = 0          -> next item
///                  e1 < bf or e2 < bf or r=1 -> bruteforce slice `child` of IL x IR
///                                               (skipped with `tensor`)
///                  else                      -> dnn_tree<r-1>(IL, IR) with the
///                                               buffers BL, BR of this CTA
///
/// \param L, R the lists in device memory
/// \param RL, RR, cnt the survivors of the roots, see `nn_tree_root`
/// \param B `2 * (r - 1) * list_size` index buffers per block
/// \param list_size size of each index buffer, >= e1, e2
/// \param bf bruteforce threshold: a node with fewer elements in one of its
/// 		lists is bruteforced (`NN_Config::BRUTEFORCE_THRESHOLD`)
/// \param work work counter, must be zero
/// \param f1 position of the final element in the left list. Must be init with -1
/// \param f2 position of the final element in the right list. Must be init with -1
/// \param seed the centers of work item `i` are drawn from the Philox subsequence `N + i`
/// \tparam tensor if true, the roots with `e1 < bf` or `e2 < bf` were already
/// 		bruteforced by `nn_tree_tensor` and are skipped
///
/// grid/block: blockDim.x a multiple of 32 (asserted), 256 recommended (the
/// bruteforce tiles are 256 rows); gridDim.x = number of persistent CTAs,
/// i.e. SMs * CTAs per SM, `B` must hold gridDim.x * 2 * (r-1) * list_size
/// entries. Shared memory: 2 * 256 * (nr_limbs * sizeof(T) + 4) bytes.
///
///   int sms, occ;
///   cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
///   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,
///       nn_tree_child<uint32_t, uint32_t, 8, d, dk, 32, N, r>, 256, 0);
///   nn_tree_child<uint32_t, uint32_t, 8, d, dk, 32, N, r><<<sms * occ, 256>>>
///       (L, R, RL, RR, cnt, B, list_size, bf, f1, f2, cnt + 2 * N, seed);
///
/// NOTE: one CTA works on one subtree at a time; the parallelism is the N*N
/// 		work items. Leaves (lists of ~bf elements) use the 256 threads
/// 		badly, see `dnn_tree_bruteforce`.
template<typename T, typename LT, const uint32_t nr_limbs,
		 const uint32_t d, const uint32_t dk, const uint32_t k,
		 const uint32_t N, const uint32_t r, const bool tensor = false>
__global__ void nn_tree_child(const T *L, const T *R,
							  const uint32_t *RL, const uint32_t *RR, const uint32_t *cnt,
							  uint32_t *B, const size_t list_size, const uint32_t bf,
							  size_t *f1, size_t *f2, uint32_t *work, const uint64_t seed) {
	static_assert(r > 0);
	assert(blockDim.x % 32u == 0u);

	__shared__ curandStatePhilox4_32_10_t state;
	__shared__ LT z;
	__shared__ uint32_t c, stop, item;

	uint32_t *BL = B + size_t(blockIdx.x) * 2u * (r - 1u) * list_size;
	uint32_t *BR = BL + (r - 1u) * list_size;
	while (true) {
		// all threads must have read the previous item
		__syncthreads();
		if (threadIdx.x == 0) {
			item = atomicAdd(work, 1u);
			stop = (*(volatile size_t *)f1) != size_t(-1);
		}
		__syncthreads();
		if (stop || (item >= N * N)) { return; }

		const uint32_t root = item / N, child = item % N;
		const uint32_t e1 = cnt[2*root], e2 = cnt[2*root + 1];
		const uint32_t *IL = RL + root * list_size, *IR = RR + root * list_size;

		/// early exit if we filtered everything out
		if ((e1 == 0) || (e2 == 0)) { continue; }

		// NOTE: the root is bruteforced only once: each of its `N` children
		// 	takes a slice of the left list
		if ((e1 < bf) || (e2 < bf) || (r == 1)) {
			if constexpr (tensor) { continue; }
			const uint32_t s = (e1 + N - 1u) / N;
			const uint32_t b = child * s;
			if (b < e1) {
				dnn_tree_bruteforce<T, nr_limbs, d>(L, R, IL + b, IR, (e1 - b) < s ? (e1 - b) : s, e2, f1, f2);
			}
			continue;
		}

		if constexpr (r > 1) {
			if (threadIdx.x == 0) {
				curand_init(seed, N + item, 0, &state);
			}
			dnn_tree<T, LT, nr_limbs, d, dk, k, N, r, r - 1u>
				(L, R, IL, IR, e1, e2, BL, BR, list_size, bf, f1, f2, &state, &z, &c, &stop);
		}
	}
}

#ifdef NN_CUDA_TENSOR
/// tensor core bruteforce of the roots, step 1 of 3: copies the survivors of
/// root `blockIdx.y` (see `nn_tree_root`) into the dense lists `DL`, `DR` of
/// `TL` limbs per element (zero padded to a multiple of 128 bits), if the root
/// is bruteforced, i.e. `e1 < bf` or `e2 < bf` (as in `nn_tree_child`). The
/// sizes of the dense lists are written into `tcnt`, 0 if not bruteforced.
///
///   RL[root]: [ 17 | 3 | 42 | ... ]        cnt[2 root] = e1 < bf
///                |    |    |
///   DL + root*list_size*TL:
///             [ L[17] 0 | L[3] 0 | L[42] 0 | ... ]   TL limbs each, zero padded
///
/// \param L, R the lists
/// \param RL, RR, cnt the survivors of the roots, see `nn_tree_root`
/// \param DL, DR output: `N` dense lists of `list_size * TL` limbs each
/// \param tcnt output: 2 sizes per root
/// \param list_size size of each index buffer / dense list
/// \param bf bruteforce threshold, as in `nn_tree_child`
///
/// grid/block: gridDim.y = N (one root per y), gridDim.x and blockDim.x
/// arbitrary (grid-stride over the elements), e.g. the grid of `nn_tree_root`:
///
///   nn_tree_gather<uint32_t, 8, 8, r><<<grid, 256>>>
///       (L, R, RL, RR, cnt, DL, DR, tcnt, list_size, bf);
template<typename T, const uint32_t nr_limbs, const uint32_t TL, const uint32_t r>
__global__ void nn_tree_gather(const T *L, const T *R,
							   const uint32_t *RL, const uint32_t *RR, const uint32_t *cnt,
							   T *DL, T *DR, uint32_t *tcnt,
							   const size_t list_size, const uint32_t bf) {
	static_assert(TL >= nr_limbs);
	const uint32_t root = blockIdx.y;
	const uint32_t e1 = cnt[2*root], e2 = cnt[2*root + 1];
	const bool bruteforce = (e1 < bf) || (e2 < bf) || (r == 1);
	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
		tcnt[2*root] = bruteforce ? e1 : 0;
		tcnt[2*root + 1] = bruteforce ? e2 : 0;
	}
	if (!bruteforce) { return; }

	const size_t o = size_t(root) * list_size;
	const uint32_t e = e1 > e2 ? e1 : e2;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < e; i += gridDim.x * blockDim.x) {
		for (uint32_t l = 0; l < TL; l++) {
			if (i < e1) { DL[(o + i)*TL + l] = l < nr_limbs ? L[size_t(RL[o + i])*nr_limbs + l] : T(0); }
			if (i < e2) { DR[(o + i)*TL + l] = l < nr_limbs ? R[size_t(RR[o + i])*nr_limbs + l] : T(0); }
		}
	}
}

/// tensor core bruteforce of the roots, step 2 of 3: `tensor_bruteforce_mxf4`
/// on sm_120a, else `tensor_bruteforce_hybrid` (see `nn/nn_tensor.cuh`) on
/// the dense lists of root `blockIdx.z`. The positions in the dense lists of
/// a solution are written into `F[2*root]` and `F[2*root + 1]`, which must be
/// init with -1.
/// \param DL, DR, tcnt the dense lists, see `nn_tree_gather`
/// \param list_size size of each dense list in elements
/// \param F output: 2 positions per root, must be init with -1
///
/// grid/block: blockDim.x = (8 + 4)*32 = 384 (12 warps: on sm_120a all of
/// them on the tensor cores, else 8 tensor core + 4 popc warps, the
/// parameters of the hybrid kernel); grid (x, 1, N): gridDim.z = N roots,
/// gridDim.x CTAs per root (grid-stride over the rows of the dense left
/// list), gridDim.y = 1 (no split of the right list):
///
///   cudaMemsetAsync(F, 0xff, 2 * N * sizeof(size_t));
///   nn_tree_tensor<uint32_t, 8, d><<<dim3(32, 1, N), (8 + 4) * 32>>>(DL, DR, tcnt, list_size, F);
///   nn_tree_tensor_result<<<1, 1>>>(RL, RR, F, N, list_size, f1, f2);
///
/// NOTE: `blockDim.x` must be (8 + 4)*32.
template<typename T, const uint32_t TL, const uint32_t d>
__global__ void __launch_bounds__((8+4)*32)
nn_tree_tensor(const T *DL, const T *DR, const uint32_t *tcnt,
			   const size_t list_size, size_t *F) {
	const uint32_t root = blockIdx.z;
	const size_t o = size_t(root) * list_size * TL;
#if defined(NN_TENSOR_MXF4)
	// variant G with 12 warps, as the launch bounds do not limit the registers
	cryptanalysislib::nn::tensor::dtensor_bruteforce_mxf4
		<T, TL, d, 0, 8 + 4, 256, 0, 2>
		(DL + o, DR + o, tcnt[2*root], tcnt[2*root + 1], F + 2*root, F + 2*root + 1);
#else
	cryptanalysislib::nn::tensor::dtensor_bruteforce_and
		<T, TL, d, 2, 8, 256, 4, 112>
		(DL + o, DR + o, tcnt[2*root], tcnt[2*root + 1], F + 2*root, F + 2*root + 1);
#endif
}

/// tensor core bruteforce of the roots, step 3 of 3: translates the first
/// solution in `F` (positions in the dense lists of its root) into positions
/// in `L`, `R` and writes them into `f1`, `f2`.
///
///   F[2 root] = i, F[2 root + 1] = j   -->   f1 = RL[root*list_size + i]
///                                            f2 = RR[root*list_size + j]
///
/// \param RL, RR the survivors of the roots, see `nn_tree_root`
/// \param F 2 positions per root, see `nn_tree_tensor`
/// \param N number of roots
/// \param list_size size of each index buffer
/// \param f1, f2 output, must be init with -1
///
/// grid/block: <<<1, 1>>>, a single thread (`N` is small). Launch it after
/// `nn_tree_tensor` and before `nn_tree_child` on the same stream, so that
/// the children stop early if the roots already found a pair.
/// NOTE: a single thread.
__global__ void nn_tree_tensor_result(const uint32_t *RL, const uint32_t *RR,
									  const size_t *F, const uint32_t N,
									  const size_t list_size,
									  size_t *f1, size_t *f2) {
	for (uint32_t root = 0; root < N; root++) {
		if (F[2*root] != size_t(-1)) {
			const size_t o = size_t(root) * list_size;
			if (atomicCAS((unsigned long long *)f1, (unsigned long long)-1, RL[o + F[2*root]]) == (unsigned long long)-1) {
				(*f2) = RR[o + F[2*root + 1]];
			}
			return;
		}
	}
}
#endif
