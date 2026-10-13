#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_pipeline.h>
#include <mma.h>

#include "helper.cuh"

/// Bruteforce step of the NN algorithm on the tensor cores. Searches for a pair
/// (i, j) with wt(L[i] ^ R[j]) <= d, using the 1-bit WMMA operations
/// (CUDA C Programming Guide, "Sub-byte Operations"):
///
///   C(8x8 s32) = bmma_sync(A, B, C, bmmaBitOp{XOR,AND}, bmmaAccumulateOpPOPC)
///   C[r][c]    = popc(L[i+r] {^,&} R[j+c])
///
/// where A are 8 consecutive elements of L (row_major) and B are 8 consecutive
/// elements of R (col_major). Elements are `n = K*128` bits; the K chunks of
/// 128 bits are accumulated into the same C. Hence one `bmma_sync` compares
/// 64 pairs on 128 bits.
///
/// Three variants are implemented:
/// 	- `tensor_bruteforce`        (C): bmmaBitOpXOR, the distance directly.
/// 	- `tensor_bruteforce_and`    (D): bmmaBitOpAND, the distance is recovered
/// 	                                  from the weights of the elements.
/// 	- `tensor_bruteforce_hybrid` (F): D, plus warps which compute on the CUDA
/// 	                                  cores (__popc) in parallel.
/// Each kernel has a host side launcher `launch_*`, which chooses the grid.
///
/// Common structure of all variants:
///
///   global R: [ chunk0 ][ chunk1 ][ chunk2 ] ...      chunk = RC elements
///                 | cp.async      | cp.async
///                 v               v
///   smem:      [ Rs[0] ]  <-->  [ Rs[1] ]             double buffered
///                 |
///       +---------+---------+---- ... ----+
///       v         v         v             v
///     warp0     warp1     warp2   ...   warpW-1       A tiles in registers
///    TM tiles  TM tiles  TM tiles       TM tiles      (all K chunks)
///
/// 	- each CTA owns CTA_ROWS = WARPS*TM*8 rows of L (a "super-tile"). They
/// 	  are staged once into shared memory, and each warp keeps its TM A-tiles
/// 	  in registers.
/// 	- R is streamed in chunks of RC elements through shared memory, double
/// 	  buffered via `cp.async`, such that the next chunk is loaded while the
/// 	  current one is processed. All warps of the CTA share each chunk.
/// 	- early abort: first only the first 128 bits are multiplied. Only if a
/// 	  lane of the warp sees a distance <= d on them, the remaining chunks
/// 	  are added. For random elements and small `d` this is very unlikely,
/// 	  hence the cost is ~1 bmma per tile instead of K.
/// 	- the mapping of the accumulator elements to (row, col) is not specified
/// 	  for WMMA. It is learned at runtime by loading the accumulator fragment
/// 	  coord[r][c] = r*8 + c from shared memory ("coord trick").
/// 	- tails: elements beyond `e1`/`e2` are zero in shared memory, and
/// 	  solutions found on them are dropped.
///
/// Shared memory layout: both L and R are stored "k-planar":
/// 	plane k = 128 bit chunk k of all elements, i.e.
/// 	X[k*stride + e] = bits [128k, 128k+128) of element e.
/// Hence every fragment starts at a 256 bit aligned address (as required by
/// `load_matrix_sync`) and `ldm = 128`. The natural layout with an offset of
/// 16k bytes would violate the alignment for odd k.
///
/// Throughput on a RTX 5090 (sm_120), n=128, d=20, |L|=|R|=2^17 with the
/// default parameters: C ~600, D ~1000, F ~1200 Gpairs/s; plain __popc
/// (`bruteforce` in nn.cuh) ~930 Gpairs/s, which however needs |L| >= 2^17 to
/// fill the GPU (one thread per element of L). The XOR is emulated on sm_90+
/// (half rate of the AND). See `DISABLED_bench` and `DISABLED_tune` in
/// tests/nn/tensor.cu.
namespace cryptanalysislib::nn::tensor {

using namespace nvcuda;

/// the only shape supported for 1-bit WMMA
constexpr uint32_t WMMA_M = 8, WMMA_N = 8, WMMA_K = 128;

using b1 = wmma::experimental::precision::b1;
/// 8 elements of L, 128 bits each
using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, b1, wmma::row_major>;
/// 8 elements of R, 128 bits each
using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, b1, wmma::col_major>;
/// 8x8 popcounts, `num_elements` of them per thread
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>;

/// checks the accumulator of an XOR bmma against the distance bound.
/// \tparam d maximal hamming distance of a solution. A pair is a candidate if
/// 		its (partial) distance is <= d. As the distance only grows with
/// 		each further 128 bit chunk, a pair whose partial distance is > d can
/// 		never become a solution, which is what the early abort relies on.
/// \param c accumulator fragment of a `bmma_sync` with bmmaBitOpXOR. Each
/// 		thread holds `c.num_elements` of the 64 entries of the 8x8 tile;
/// 		entry (r, col) contains popc(L[r] ^ R[col]) summed over all 128 bit
/// 		chunks multiplied so far. Which entries a thread holds is not
/// 		specified, see the coord trick at the top of this file.
/// \return true if one of the accumulator elements of this thread is <= d.
/// 		NOTE: only the elements of the calling thread are checked, use
/// 		`__any_sync` for the whole tile.
template<const uint32_t d>
__device__ __forceinline__ bool any_leq(const FragC &c) {
	bool ret = false;
	#pragma unroll
	for (uint32_t s = 0; s < c.num_elements; s++) {
		ret |= uint32_t(c.x[s]) <= d;
	}
	return ret;
}

/// hamming distance of two 128 bit chunks on the CUDA cores (4 LOP3 + 4 POPC).
/// Used by the popc warps of the hybrid variant.
/// \param a first 128 bit chunk, e.g. chunk k of an element of L. The 4 words
/// 		are treated as one 128 bit vector, the bit order does not matter.
/// \param b second 128 bit chunk, e.g. chunk k of an element of R
/// \return popc(a ^ b) in [0, 128], i.e. the hamming distance between `a` and `b`
__device__ __forceinline__ uint32_t popc_xor(const uint4 a, const uint4 b) {
	return __popc(a.x ^ b.x) + __popc(a.y ^ b.y) + __popc(a.z ^ b.z) + __popc(a.w ^ b.w);
}

/// hamming weight of a 128 bit chunk on the CUDA cores.
/// \param a 128 bit chunk, the 4 words are treated as one 128 bit vector
/// \return popc(a) in [0, 128], i.e. the hamming weight of `a`
__device__ __forceinline__ uint32_t popc(const uint4 a) {
	return __popc(a.x) + __popc(a.y) + __popc(a.z) + __popc(a.w);
}

/// computes the hamming weights of a k-planar stored element, as needed by the
/// AND variants to recover the distance from popc(a & b).
/// \tparam K number of 128 bit chunks per element, i.e. n / 128. Must be > 0.
/// \param X pointer to chunk 0 of the element in shared memory. Chunk k of the
/// 		same element is at X[k * stride] (k-planar layout, see the top of
/// 		this file).
/// \param stride distance in uint4 between chunk k and chunk k+1 of the same
/// 		element, i.e. the number of elements per plane: CTA_ROWS for the L
/// 		super-tile `Ls`, RC for an R stage `Rs[b]`.
/// \return packed weights: (wt(element) << 16) | wt(chunk 0 of element).
/// 		The low half is used for the early abort on the first 128 bits, the
/// 		high half for the final check. Both fit into 16 bits, as
/// 		wt <= K*128 < 2^16 for all supported K.
template<const uint32_t K>
__device__ __forceinline__ uint32_t weights(const uint4 *X, const uint32_t stride) {
	const uint32_t w0 = popc(X[0]);
	uint32_t w = w0;
	#pragma unroll
	for (uint32_t k = 1; k < K; k++) {
		w += popc(X[k * stride]);
	}
	return (w << 16u) | w0;
}

/// writes the solution (row, col) into (f1, f2), if no other solution was
/// written before. Same protocol as `dbruteforce` in `nn.cuh`: the first
/// thread which swaps `f1` from -1 to `row` also writes `f2`. All other
/// solutions (of this or any other thread) are dropped.
/// \param row absolute position of the solution in L, i.e. index into the
/// 		list passed to the kernel (not into the shared memory tile).
/// 		Must be < e1; the caller filters the zero padding.
/// \param col absolute position of the solution in R. Must be < e2.
/// \param f1 global (device or managed) memory holding the position in L.
/// 		Must be init with -1 before the kernel is launched; -1 means "no
/// 		solution found yet".
/// \param f2 global (device or managed) memory holding the position in R.
/// 		Must be init with -1 before the kernel is launched.
/// NOTE: `f2` is only valid after the kernel finished, as it is written after
/// 		the atomic swap of `f1`.
__device__ __forceinline__ void report_solution(const size_t row, const size_t col,
												size_t *f1, size_t *f2) {
	if (atomicCAS((unsigned long long *)f1, (unsigned long long)-1, row) == (unsigned long long)-1) {
		(*f2) = col;
	}
}

/// copies the R chunk `c` (`RC` elements of K*128 bits) k-planar into the
/// shared memory stage `Rs` via asynchronous copies (`cp.async`), and commits
/// them as one batch. Wait for it with `__pipeline_wait_prior` followed by
/// `__syncthreads`.
/// Consecutive threads read consecutive 16 bytes in global memory (coalesced).
/// Elements beyond `e2` are written as zero.
/// NOTE: must be called by all threads of the CTA, as each thread commits its
/// 		own batch of async copies and the work is strided over `blockDim.x`.
/// \tparam K number of 128 bit chunks per element, i.e. n / 128
/// \tparam RC number of elements per R chunk, which is also the number of
/// 		elements per plane in `Rs` (chunk k of element e is at Rs[k*RC + e])
/// \param Rs destination: one shared memory stage of K*RC uint4, 16 byte
/// 		aligned (128 byte aligned for conflict free fragment loads).
/// 		Must not be read by any warp until the copies are waited for.
/// \param R4 source: the list R in global memory, viewed as 128 bit words, i.e.
/// 		element g, chunk k is at R4[g*K + k]. Must be 16 byte aligned.
/// \param c index of the chunk: elements [c*RC, c*RC + RC) of R are copied
/// \param e2 number of valid elements in R. Elements with index >= e2 are not
/// 		read, their slots in `Rs` are set to zero.
template<const uint32_t K, const uint32_t RC>
__device__ __forceinline__ void load_chunk(uint4 *Rs, const uint4 *R4,
										   const size_t c, const size_t e2) {
	for (uint32_t p = threadIdx.x; p < K * RC; p += blockDim.x) {
		const uint32_t e = p / K, k = p % K;
		const size_t g = c * RC + e;
		uint4 *dst = Rs + k * RC + e;
		if (g < e2) {
			__pipeline_memcpy_async(dst, R4 + g * K + k, sizeof(uint4));
		} else {
			*dst = make_uint4(0, 0, 0, 0);
		}
	}
	__pipeline_commit();
}

/// Variant C: searches a pair (i, j), i < e1, j < e2 with
/// wt(L[i] ^ R[j]) <= d with the tensor cores via bmmaBitOpXOR.
///
/// Each warp keeps TM tiles of 8 rows of L in registers. For each 8-element
/// tile of R in shared memory, it computes TM bmma on the first 128 bits. Only
/// if any lane sees a distance <= d (`__any_sync`), the next 128 bits are
/// accumulated, and so on, until either no distance <= d is left, or all K
/// chunks are processed. Then all accumulator elements <= d are reported.
/// See the top of this file for the overall structure.
///
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t. Only
/// 		used to compute the element size; the kernel internally reads the
/// 		lists as 128 bit words, so the limb type does not change the bit
/// 		order of the distance computation.
/// \tparam nr_limbs number of limbs of type `T` per element. The element size
/// 		n = nr_limbs*sizeof(T)*8 must be a multiple of 128, as the tensor
/// 		cores multiply 128 bit chunks (K = n/128 of them). Pad elements with
/// 		zero limbs otherwise (zeros do not change the distance).
/// \tparam d maximal hamming distance: (i, j) is a solution if
/// 		wt(L[i] ^ R[j]) <= d. Must be smaller than n. The smaller `d`
/// 		compared to 64 (the expected distance on the first 128 bits), the
/// 		more effective the early abort for K > 1.
/// \tparam TM number of 8-row L tiles each warp keeps in registers (TM*K
/// 		registers for A, 2*TM for the accumulators). Each B fragment
/// 		loaded from shared memory is used for TM bmma, so larger values
/// 		reduce the shared memory traffic, but increase the register
/// 		pressure and the size of each L super-tile (worse for small e1).
/// 		Default 2 (best on sm_120, see `DISABLED_tune` in
/// 		tests/nn/tensor.cu).
/// \tparam WARPS number of warps per CTA; all of them compute on the tensor
/// 		cores and share each R chunk in shared memory. `blockDim.x` must be
/// 		WARPS*32. Together with TM it defines the rows of L per CTA:
/// 		CTA_ROWS = WARPS*TM*8.
/// \tparam RC number of elements of R per shared memory stage (two stages,
/// 		double buffered). Must be a multiple of 8. Larger values mean fewer
/// 		`__syncthreads` per pair and more copies in flight, but more shared
/// 		memory and coarser work splitting along R.
/// \param L first list: `e1` elements of n bits each, stored contiguously
/// 		(element i occupies limbs [i*nr_limbs, (i+1)*nr_limbs)), in global
/// 		memory, 16 byte aligned.
/// \param R second list, same layout as `L`, `e2` elements, 16 byte aligned.
/// \param e1 number of elements of L to search, i.e. L[0..e1). Need not be a
/// 		multiple of any tile size.
/// \param e2 number of elements of R to search, i.e. R[0..e2). Need not be a
/// 		multiple of any tile size.
/// \param f1 output in global memory: position i of the solution in L. Must be
/// 		init with -1; stays -1 if no solution exists.
/// \param f2 output in global memory: position j of the solution in R. Must be
/// 		init with -1; stays -1 if no solution exists.
/// NOTE: if more than one solution exists, only the first one found is written
/// 		into `f1` and `f2`. Which one is the first is not deterministic.
/// NOTE: gridDim.x is a grid-stride over the L super-tiles (CTA_ROWS rows
/// 		each), gridDim.y splits R into gridDim.y contiguous parts. Use
/// 		`launch_tensor_bruteforce` to choose the grid.
/// NOTE: shared memory: (WARPS*TM*8 + 2*RC) * K * 16 bytes + 256 bytes, which
/// 		must not exceed 48 KiB (checked at compile time).
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=128>
__global__ void __launch_bounds__(WARPS*32)
tensor_bruteforce(const T *L, const T *R,
				  const size_t e1, const size_t e2,
				  size_t *f1, size_t *f2) {
	static_assert(sizeof(size_t) == sizeof(unsigned long long));
	static_assert((nr_limbs * sizeof(T) * 8u) % WMMA_K == 0, "element size must be a multiple of 128 bit");
	static_assert(RC % WMMA_N == 0);
	constexpr uint32_t K = (nr_limbs * sizeof(T) * 8u) / WMMA_K;
	static_assert(d < K * WMMA_K);
	constexpr uint32_t CTA_ROWS = WARPS * TM * WMMA_M;
	static_assert((CTA_ROWS + 2u*RC) * K * sizeof(uint4) + 64u*sizeof(int) <= 48u*1024u,
				  "too much static shared memory");

	__shared__ alignas(128) uint4 Ls[K * CTA_ROWS];
	__shared__ alignas(128) uint4 Rs[2][K * RC];
	__shared__ alignas(32) int coords[WMMA_M * WMMA_N];

	assert(blockDim.x == WARPS * 32u);
	const uint32_t tid  = threadIdx.x;
	const uint32_t warp = tid / 32u;
	const uint4 *L4 = (const uint4 *)L;
	const uint4 *R4 = (const uint4 *)R;

	// learn the (row, col) of each accumulator element of this thread
	for (uint32_t p = tid; p < WMMA_M * WMMA_N; p += blockDim.x) {
		coords[p] = p;
	}
	__syncthreads();
	FragC coord;
	wmma::load_matrix_sync(coord, coords, WMMA_N, wmma::mem_row_major);

	// the part of R this CTA works on, in chunks of RC elements
	const size_t nr_chunks = (e2 + RC - 1u) / RC;
	const size_t c_begin = (nr_chunks * blockIdx.y) / gridDim.y;
	const size_t c_end = (nr_chunks * (blockIdx.y + 1u)) / gridDim.y;
	if (c_begin == c_end) {
		return;
	}

	for (size_t lt = blockIdx.x; lt * CTA_ROWS < e1; lt += gridDim.x) {
		const size_t l_base = lt * CTA_ROWS;

		// all warps must be finished with the previous L tile and R stages
		__syncthreads();
		load_chunk<K, RC>(Rs[0], R4, c_begin, e2);

		// stage the L super-tile; out of range rows are zero
		for (uint32_t p = tid; p < K * CTA_ROWS; p += blockDim.x) {
			const uint32_t e = p / K, k = p % K;
			const size_t g = l_base + e;
			Ls[k * CTA_ROWS + e] = g < e1 ? L4[g * K + k] : make_uint4(0, 0, 0, 0);
		}
		__syncthreads();

		const size_t w_base = l_base + warp * TM * WMMA_M;
		// warp-uniform: warps without any valid L row only help with the copies
		const bool active = w_base < e1;
		FragA a[TM][K];
		#pragma unroll
		for (uint32_t i = 0; i < TM; i++) {
			#pragma unroll
			for (uint32_t k = 0; k < K; k++) {
				wmma::load_matrix_sync(a[i][k], Ls + k * CTA_ROWS + (warp * TM + i) * WMMA_M, WMMA_K);
			}
		}

		for (size_t c = c_begin; c < c_end; c++) {
			const uint32_t b = (c - c_begin) & 1u;
			if (c + 1u < c_end) {
				load_chunk<K, RC>(Rs[b ^ 1u], R4, c + 1u, e2);
				__pipeline_wait_prior(1);
			} else {
				__pipeline_wait_prior(0);
			}
			__syncthreads();

			const size_t r_base = c * RC;
			for (uint32_t t = 0; active && (t < RC / WMMA_N) && (r_base + t * WMMA_N < e2); t++) {
				const uint4 *Rt = Rs[b] + t * WMMA_N;

				// first 128 bits only
				FragB fb;
				FragC acc[TM];
				wmma::load_matrix_sync(fb, Rt, WMMA_K);
				bool cand = false;
				#pragma unroll
				for (uint32_t i = 0; i < TM; i++) {
					wmma::fill_fragment(acc[i], 0);
					wmma::bmma_sync(acc[i], a[i][0], fb, acc[i],
									wmma::experimental::bmmaBitOpXOR,
									wmma::experimental::bmmaAccumulateOpPOPC);
					cand |= any_leq<d>(acc[i]);
				}

				// highly unlikely: some partial distance is <= d
				if (__any_sync(0xffffffffu, cand)) {
					#pragma unroll
					for (uint32_t k = 1; k < K; k++) {
						wmma::load_matrix_sync(fb, Rt + k * RC, WMMA_K);
						cand = false;
						#pragma unroll
						for (uint32_t i = 0; i < TM; i++) {
							wmma::bmma_sync(acc[i], a[i][k], fb, acc[i],
											wmma::experimental::bmmaBitOpXOR,
											wmma::experimental::bmmaAccumulateOpPOPC);
							cand |= any_leq<d>(acc[i]);
						}

						if (!__any_sync(0xffffffffu, cand)) {
							break;
						}
					}

					// NOTE: if the loop above was left early, no element is <= d
					#pragma unroll
					for (uint32_t i = 0; i < TM; i++) {
						#pragma unroll
						for (uint32_t s = 0; s < acc[i].num_elements; s++) {
							const size_t row = w_base + i * WMMA_M + coord.x[s] / WMMA_N;
							const size_t col = r_base + t * WMMA_N + coord.x[s] % WMMA_N;
							// the zero padding can produce wrong solutions
							if ((uint32_t(acc[i].x[s]) <= d) && (row < e1) && (col < e2)) {
								report_solution(row, col, f1, f2);
							}
						}
					}
				}
			}

			// all warps must be finished with stage `b` before it is overwritten
			__syncthreads();
		}
	}
}

/// Variants D and F: searches a pair (i, j), i < e1, j < e2 with
/// wt(L[i] ^ R[j]) <= d. The tensor cores compute popc(a & b) via
/// bmmaBitOpAND, which runs at twice the rate of bmmaBitOpXOR on sm_90+ (the
/// XOR is emulated there). The distance is recovered via
///
///   wt(a ^ b) = wt(a) + wt(b) - 2*popc(a & b) <= d
///   <=>  2*popc(a & b) - wt(b) >= wt(a) - d
///
/// The weights of the L super-tile are computed once into shared memory
/// (`Lw`), and each tensor core thread keeps the thresholds wt(a) - d of the
/// rows of its accumulator elements in registers (`tl0`, `tl`). The weights of
/// each R chunk are computed once per CTA into shared memory (`Rw`), which
/// costs one additional `__syncthreads` per chunk. For the early abort the
/// weights of the first 128 bits are stored as well, see `weights`.
/// In contrast to variant C, the early abort only checks the first 128 bits:
/// a candidate tile is directly multiplied on all remaining chunks.
///
/// Variant F (POPC_WARPS > 0): additionally `POPC_WARPS` warps compute on the
/// CUDA cores (`__popc` of the XOR) in parallel to the `WARPS` tensor core
/// warps. As both use different execution units, both run (mostly) in
/// parallel. Each R chunk is split:
///
///   Rs[b]: [ 0 ............ RC-RC_POPC ............ RC )
///           \__ tensor core warps __/ \__ popc warps __/
///             all CTA_ROWS rows         all CTA_ROWS rows
///
/// Each popc thread keeps CTA_ROWS / (POPC_WARPS*32) rows of L in registers and
/// iterates over its columns of the chunk; all lanes of a warp read the same
/// R element (shared memory broadcast). The ratio RC_POPC/RC must match the
/// throughput ratio of both paths; it is sensitive (±10%), so tune it.
///
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t. Only
/// 		used to compute the element size; the lists are read as 128 bit
/// 		words.
/// \tparam nr_limbs number of limbs of type `T` per element. The element size
/// 		n = nr_limbs*sizeof(T)*8 must be a multiple of 128 (K = n/128
/// 		chunks). Pad elements with zero limbs otherwise.
/// \tparam d maximal hamming distance: (i, j) is a solution if
/// 		wt(L[i] ^ R[j]) <= d. Must be smaller than n. Enters the thresholds
/// 		wt(a) - d, which each tensor core thread precomputes per L row.
/// \tparam TM number of 8-row L tiles each tensor core warp keeps in registers
/// 		(TM*K for A, 2*TM*2 for the thresholds, 2*TM for the
/// 		accumulators). Each B fragment is used for TM bmma. Default 2.
/// \tparam WARPS number of tensor core warps per CTA. Together with TM it
/// 		defines the rows of L per CTA: CTA_ROWS = WARPS*TM*8. The popc warps
/// 		work on the same CTA_ROWS rows.
/// \tparam RC number of elements of R per shared memory stage (two stages,
/// 		double buffered). Must be a multiple of 8. Per stage the weights
/// 		of the RC - RC_POPC tensor core elements are computed once.
/// \tparam POPC_WARPS number of additional CUDA core warps per CTA (variant
/// 		F). Zero for variant D. CTA_ROWS must be a multiple of
/// 		POPC_WARPS*32, as each popc thread keeps exactly
/// 		CTA_ROWS / (POPC_WARPS*32) rows of L (each K uint4) in registers.
/// \tparam RC_POPC number of elements at the end of each R chunk, which are
/// 		processed by the CUDA core warps instead of the tensor cores.
/// 		RC - RC_POPC must be a multiple of 8 (whole B tiles). Must be zero
/// 		iff `POPC_WARPS` is zero. RC_POPC == RC is allowed, then only the
/// 		CUDA cores compute. The ratio RC_POPC / RC should match the share
/// 		of the CUDA cores in the total throughput (~0.44 on sm_120).
/// \param L first list: `e1` elements of n bits each, stored contiguously
/// 		(element i occupies limbs [i*nr_limbs, (i+1)*nr_limbs)), in global
/// 		memory, 16 byte aligned.
/// \param R second list, same layout as `L`, `e2` elements, 16 byte aligned.
/// \param e1 number of elements of L to search, i.e. L[0..e1). Need not be a
/// 		multiple of any tile size.
/// \param e2 number of elements of R to search, i.e. R[0..e2). Need not be a
/// 		multiple of any tile size.
/// \param f1 output in global memory: position i of the solution in L. Must be
/// 		init with -1; stays -1 if no solution exists.
/// \param f2 output in global memory: position j of the solution in R. Must be
/// 		init with -1; stays -1 if no solution exists.
/// NOTE: `blockDim.x` must be (WARPS + POPC_WARPS)*32. Warps [0, WARPS) are
/// 		the tensor core warps, warps [WARPS, WARPS+POPC_WARPS) the popc warps.
/// NOTE: if more than one solution exists, only the first one found is written
/// 		into `f1` and `f2`. Which one is the first is not deterministic.
/// NOTE: grid as in `tensor_bruteforce`.
/// NOTE: shared memory: (CTA_ROWS + 2*RC) * K * 16 + (CTA_ROWS + RC) * 4 +
/// 		256 bytes, which must not exceed 48 KiB (checked at compile time).
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM, const uint32_t WARPS, const uint32_t RC,
		 const uint32_t POPC_WARPS, const uint32_t RC_POPC>
__device__ void dtensor_bruteforce_and(const T *L, const T *R,
									   const size_t e1, const size_t e2,
									   size_t *f1, size_t *f2) {
	static_assert(sizeof(size_t) == sizeof(unsigned long long));
	static_assert((nr_limbs * sizeof(T) * 8u) % WMMA_K == 0, "element size must be a multiple of 128 bit");
	constexpr uint32_t K = (nr_limbs * sizeof(T) * 8u) / WMMA_K;
	static_assert(d < K * WMMA_K);
	constexpr uint32_t CTA_ROWS = WARPS * TM * WMMA_M;
	constexpr uint32_t RC_TENSOR = RC - RC_POPC;
	static_assert(RC_POPC <= RC);
	static_assert(RC_TENSOR % WMMA_N == 0);
	static_assert((POPC_WARPS == 0) == (RC_POPC == 0));
	constexpr uint32_t ROWS_POPC = POPC_WARPS ? CTA_ROWS / (POPC_WARPS * 32u) : 0u;
	static_assert((POPC_WARPS == 0) || (ROWS_POPC * POPC_WARPS * 32u == CTA_ROWS),
				  "the rows of a CTA must be distributable on the popc threads");
	static_assert((CTA_ROWS + 2u*RC) * K * sizeof(uint4) + (CTA_ROWS + RC) * sizeof(uint32_t) +
				  64u*sizeof(int) <= 48u*1024u, "too much static shared memory");

	__shared__ alignas(128) uint4 Ls[K * CTA_ROWS];
	__shared__ alignas(128) uint4 Rs[2][K * RC];
	__shared__ uint32_t Lw[CTA_ROWS];
	__shared__ uint32_t Rw[RC_TENSOR ? RC_TENSOR : 1u];
	__shared__ alignas(32) int coords[WMMA_M * WMMA_N];

	assert(blockDim.x == (WARPS + POPC_WARPS) * 32u);
	const uint32_t tid  = threadIdx.x;
	const uint32_t warp = tid / 32u;
	const uint4 *L4 = (const uint4 *)L;
	const uint4 *R4 = (const uint4 *)R;

	// learn the (row, col) of each accumulator element of this thread
	for (uint32_t p = tid; p < WMMA_M * WMMA_N; p += blockDim.x) {
		coords[p] = p;
	}
	__syncthreads();
	FragC coord;
	wmma::load_matrix_sync(coord, coords, WMMA_N, wmma::mem_row_major);

	// the part of R this CTA works on, in chunks of RC elements
	const size_t nr_chunks = (e2 + RC - 1u) / RC;
	const size_t c_begin = (nr_chunks * blockIdx.y) / gridDim.y;
	const size_t c_end = (nr_chunks * (blockIdx.y + 1u)) / gridDim.y;
	if (c_begin == c_end) {
		return;
	}

	for (size_t lt = blockIdx.x; lt * CTA_ROWS < e1; lt += gridDim.x) {
		const size_t l_base = lt * CTA_ROWS;

		// all warps must be finished with the previous L tile and R stages
		__syncthreads();
		load_chunk<K, RC>(Rs[0], R4, c_begin, e2);

		// stage the L super-tile; out of range rows are zero
		for (uint32_t p = tid; p < K * CTA_ROWS; p += blockDim.x) {
			const uint32_t e = p / K, k = p % K;
			const size_t g = l_base + e;
			Ls[k * CTA_ROWS + e] = g < e1 ? L4[g * K + k] : make_uint4(0, 0, 0, 0);
		}
		__syncthreads();
		for (uint32_t e = tid; e < CTA_ROWS; e += blockDim.x) {
			Lw[e] = weights<K>(Ls + e, CTA_ROWS);
		}
		__syncthreads();

		// tensor core warps: A tiles and the thresholds wt(a) - d of the
		// accumulator elements of this thread (first 128 bits and full).
		const bool tensor_warp = warp < WARPS;
		const size_t w_base = l_base + warp * TM * WMMA_M;
		const bool active = tensor_warp && (w_base < e1);
		FragA a[TM][K];
		int tl0[TM][FragC::num_elements], tl[TM][FragC::num_elements];
		// popc warps: their rows of L
		uint4 Lr[ROWS_POPC ? ROWS_POPC : 1u][K];
		const uint32_t popc_tid = tid - WARPS * 32u;
		if (tensor_warp) {
			#pragma unroll
			for (uint32_t i = 0; i < TM; i++) {
				#pragma unroll
				for (uint32_t k = 0; k < K; k++) {
					wmma::load_matrix_sync(a[i][k], Ls + k * CTA_ROWS + (warp * TM + i) * WMMA_M, WMMA_K);
				}
				#pragma unroll
				for (uint32_t s = 0; s < FragC::num_elements; s++) {
					const uint32_t w = Lw[(warp * TM + i) * WMMA_M + coord.x[s] / WMMA_N];
					tl0[i][s] = int(w & 0xffffu) - int(d);
					tl[i][s] = int(w >> 16u) - int(d);
				}
			}
		} else if constexpr (POPC_WARPS > 0) {
			#pragma unroll
			for (uint32_t q = 0; q < ROWS_POPC; q++) {
				#pragma unroll
				for (uint32_t k = 0; k < K; k++) {
					Lr[q][k] = Ls[k * CTA_ROWS + q * POPC_WARPS * 32u + popc_tid];
				}
			}
		}

		for (size_t c = c_begin; c < c_end; c++) {
			const uint32_t b = (c - c_begin) & 1u;
			if (c + 1u < c_end) {
				load_chunk<K, RC>(Rs[b ^ 1u], R4, c + 1u, e2);
				__pipeline_wait_prior(1);
			} else {
				__pipeline_wait_prior(0);
			}
			__syncthreads();

			const size_t r_base = c * RC;
			if constexpr (RC_TENSOR > 0) {
				for (uint32_t e = tid; e < RC_TENSOR; e += blockDim.x) {
					Rw[e] = weights<K>(Rs[b] + e, RC);
				}
				__syncthreads();

				for (uint32_t t = 0; active && (t < RC_TENSOR / WMMA_N) && (r_base + t * WMMA_N < e2); t++) {
					const uint4 *Rt = Rs[b] + t * WMMA_N;
					uint32_t rw[FragC::num_elements];
					#pragma unroll
					for (uint32_t s = 0; s < FragC::num_elements; s++) {
						rw[s] = Rw[t * WMMA_N + coord.x[s] % WMMA_N];
					}

					// first 128 bits only
					FragB fb;
					FragC acc[TM];
					wmma::load_matrix_sync(fb, Rt, WMMA_K);
					bool cand = false;
					#pragma unroll
					for (uint32_t i = 0; i < TM; i++) {
						wmma::fill_fragment(acc[i], 0);
						wmma::bmma_sync(acc[i], a[i][0], fb, acc[i],
										wmma::experimental::bmmaBitOpAND,
										wmma::experimental::bmmaAccumulateOpPOPC);
						#pragma unroll
						for (uint32_t s = 0; s < FragC::num_elements; s++) {
							cand |= 2 * acc[i].x[s] - int(rw[s] & 0xffffu) >= tl0[i][s];
						}
					}

					// highly unlikely: some distance on the first 128 bits is <= d
					if (__any_sync(0xffffffffu, cand)) {
						#pragma unroll
						for (uint32_t k = 1; k < K; k++) {
							wmma::load_matrix_sync(fb, Rt + k * RC, WMMA_K);
							#pragma unroll
							for (uint32_t i = 0; i < TM; i++) {
								wmma::bmma_sync(acc[i], a[i][k], fb, acc[i],
												wmma::experimental::bmmaBitOpAND,
												wmma::experimental::bmmaAccumulateOpPOPC);
							}
						}

						#pragma unroll
						for (uint32_t i = 0; i < TM; i++) {
							#pragma unroll
							for (uint32_t s = 0; s < FragC::num_elements; s++) {
								const size_t row = w_base + i * WMMA_M + coord.x[s] / WMMA_N;
								const size_t col = r_base + t * WMMA_N + coord.x[s] % WMMA_N;
								// the zero padding can produce wrong solutions
								if ((2 * acc[i].x[s] - int(rw[s] >> 16u) >= tl[i][s]) && (row < e1) && (col < e2)) {
									report_solution(row, col, f1, f2);
								}
							}
						}
					}
				}
			}

			if constexpr (POPC_WARPS > 0) {
				// R elements are read by all lanes of a warp: broadcast
				for (uint32_t j = RC_TENSOR; (!tensor_warp) && (j < RC) && (r_base + j < e2); j++) {
					const uint4 r0 = Rs[b][j];
					#pragma unroll
					for (uint32_t q = 0; q < ROWS_POPC; q++) {
						uint32_t w = popc_xor(Lr[q][0], r0);
						// highly unlikely
						if (w <= d) {
							#pragma unroll
							for (uint32_t k = 1; k < K; k++) {
								w += popc_xor(Lr[q][k], Rs[b][k * RC + j]);
							}

							const size_t row = l_base + q * POPC_WARPS * 32u + popc_tid;
							if ((w <= d) && (row < e1)) {
								report_solution(row, r_base + j, f1, f2);
							}
						}
					}
				}
			}

			// all warps must be finished with stage `b` before it is overwritten
			__syncthreads();
		}
	}
}

/// Variant D: global wrapper around `dtensor_bruteforce_and` without CUDA core
/// warps (POPC_WARPS = RC_POPC = 0). Use `launch_tensor_bruteforce_and` to
/// choose the grid.
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per element; n =
/// 		nr_limbs*sizeof(T)*8 must be a multiple of 128.
/// \tparam d maximal hamming distance of a solution, d < n
/// \tparam TM number of 8-row L tiles per warp in registers (default 2)
/// \tparam WARPS number of tensor core warps per CTA; `blockDim.x` must be
/// 		WARPS*32 (default 8)
/// \tparam RC number of elements of R per shared memory stage, multiple of 8
/// 		(default 128)
/// \param L first list, `e1` contiguous elements, 16 byte aligned
/// \param R second list, `e2` contiguous elements, 16 byte aligned
/// \param e1 number of elements of L to search
/// \param e2 number of elements of R to search
/// \param f1 output: position of the solution in L, must be init with -1
/// \param f2 output: position of the solution in R, must be init with -1
/// See `dtensor_bruteforce_and` for the details of all parameters.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=128>
__global__ void __launch_bounds__(WARPS*32)
tensor_bruteforce_and(const T *L, const T *R,
					  const size_t e1, const size_t e2,
					  size_t *f1, size_t *f2) {
	dtensor_bruteforce_and
		<T, nr_limbs, d, TM, WARPS, RC, 0, 0>
		(L, R, e1, e2, f1, f2);
}

/// Variant F: global wrapper around `dtensor_bruteforce_and` with
/// `POPC_WARPS` CUDA core warps, which process the last `RC_POPC` elements of
/// each R chunk. Use `launch_tensor_bruteforce_hybrid` to choose the grid.
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per element; n =
/// 		nr_limbs*sizeof(T)*8 must be a multiple of 128.
/// \tparam d maximal hamming distance of a solution, d < n
/// \tparam TM number of 8-row L tiles per tensor core warp in registers
/// 		(default 2)
/// \tparam WARPS number of tensor core warps per CTA (default 8)
/// \tparam RC number of elements of R per shared memory stage, multiple of 8
/// 		(default 256)
/// \tparam POPC_WARPS number of CUDA core warps per CTA, > 0; WARPS*TM*8 must
/// 		be a multiple of POPC_WARPS*32 (default 4)
/// \tparam RC_POPC number of elements at the end of each R chunk processed by
/// 		the CUDA core warps, 0 < RC_POPC <= RC, RC - RC_POPC multiple of 8
/// 		(default 112, i.e. ~44% of each chunk)
/// \param L first list, `e1` contiguous elements, 16 byte aligned
/// \param R second list, `e2` contiguous elements, 16 byte aligned
/// \param e1 number of elements of L to search
/// \param e2 number of elements of R to search
/// \param f1 output: position of the solution in L, must be init with -1
/// \param f2 output: position of the solution in R, must be init with -1
/// NOTE: `blockDim.x` must be (WARPS + POPC_WARPS)*32.
/// See `dtensor_bruteforce_and` for the details of all parameters.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=256,
		 const uint32_t POPC_WARPS=4, const uint32_t RC_POPC=112>
__global__ void __launch_bounds__((WARPS+POPC_WARPS)*32)
tensor_bruteforce_hybrid(const T *L, const T *R,
						 const size_t e1, const size_t e2,
						 size_t *f1, size_t *f2) {
	dtensor_bruteforce_and
		<T, nr_limbs, d, TM, WARPS, RC, POPC_WARPS, RC_POPC>
		(L, R, e1, e2, f1, f2);
}

/// computes the grid for the kernels in this file:
/// 	- x: one CTA per L super-tile of `CTA_ROWS` rows,
/// 	- y: R is split into contiguous parts (multiples of `RC` elements),
/// 	  such that there are about 8 CTAs per SM. This is needed if L is small,
/// 	  otherwise only a few SMs would be used.
/// The SM count is read from the current device (`cudaGetDevice`).
/// \param e1 number of elements in L; determines gridDim.x =
/// 		ceil(e1 / CTA_ROWS) (capped at 2^31-1, the kernels grid-stride).
/// \param e2 number of elements in R; gridDim.y is at most the number of R
/// 		chunks ceil(e2 / RC) (and at most 65535), so that every CTA gets at
/// 		least one chunk.
/// \param CTA_ROWS number of rows of L each CTA processes at once, i.e.
/// 		WARPS*TM*8 of the kernel to launch
/// \param RC number of elements of R per shared memory stage of the kernel
/// 		to launch
/// \return the grid (gridDim.x, gridDim.y, 1)
inline dim3 tensor_grid(const size_t e1, const size_t e2,
						const uint32_t CTA_ROWS, const uint32_t RC) {
	int dev = 0, sms = 0;
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

	const size_t gx = (e1 + CTA_ROWS - 1u) / CTA_ROWS;
	const size_t nr_chunks = (e2 + RC - 1u) / RC;
	const size_t target = 8u * size_t(sms);
	size_t gy = (target + gx - 1u) / gx;
	gy = gy > nr_chunks ? nr_chunks : gy;
	gy = gy > 65535u ? 65535u : gy;

	return dim3(uint32_t(gx > 0x7fffffffu ? 0x7fffffffu : gx), uint32_t(gy));
}

/// launches `tensor_bruteforce` (variant C, XOR) on `stream` with the grid
/// of `tensor_grid` and WARPS*32 threads per CTA. Does not synchronize.
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per element; n =
/// 		nr_limbs*sizeof(T)*8 must be a multiple of 128.
/// \tparam d maximal hamming distance of a solution, d < n
/// \tparam TM number of 8-row L tiles per warp in registers (default 2)
/// \tparam WARPS number of warps per CTA (default 8)
/// \tparam RC number of elements of R per shared memory stage, multiple of 8
/// 		(default 128)
/// \param L first list, `e1` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param R second list, `e2` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param e1 number of elements of L to search
/// \param e2 number of elements of R to search
/// \param f1 output: position of the solution in L. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param f2 output: position of the solution in R. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param stream CUDA stream to launch on (default: the legacy default stream)
/// NOTE: nothing is launched if one of the lists is empty.
/// NOTE: launch errors exit the program via `checkCudaErrors`.
/// See `tensor_bruteforce` for the details of all parameters.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=128>
void launch_tensor_bruteforce(const T *L, const T *R,
							  const size_t e1, const size_t e2,
							  size_t *f1, size_t *f2,
							  cudaStream_t stream = 0) {
	if ((e1 == 0) || (e2 == 0)) {
		return;
	}

	tensor_bruteforce
		<T, nr_limbs, d, TM, WARPS, RC>
		<<<tensor_grid(e1, e2, WARPS * TM * WMMA_M, RC), WARPS * 32u, 0, stream>>>
		(L, R, e1, e2, f1, f2);
	checkCudaErrors(cudaGetLastError());
}

/// launches `tensor_bruteforce_and` (variant D, AND + weights) on `stream`
/// with the grid of `tensor_grid` and WARPS*32 threads per CTA. Does not
/// synchronize.
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per element; n =
/// 		nr_limbs*sizeof(T)*8 must be a multiple of 128.
/// \tparam d maximal hamming distance of a solution, d < n
/// \tparam TM number of 8-row L tiles per warp in registers (default 2)
/// \tparam WARPS number of tensor core warps per CTA (default 8)
/// \tparam RC number of elements of R per shared memory stage, multiple of 8
/// 		(default 128)
/// \param L first list, `e1` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param R second list, `e2` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param e1 number of elements of L to search
/// \param e2 number of elements of R to search
/// \param f1 output: position of the solution in L. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param f2 output: position of the solution in R. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param stream CUDA stream to launch on (default: the legacy default stream)
/// NOTE: nothing is launched if one of the lists is empty.
/// NOTE: launch errors exit the program via `checkCudaErrors`.
/// See `dtensor_bruteforce_and` for the details of all parameters.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=128>
void launch_tensor_bruteforce_and(const T *L, const T *R,
								  const size_t e1, const size_t e2,
								  size_t *f1, size_t *f2,
								  cudaStream_t stream = 0) {
	if ((e1 == 0) || (e2 == 0)) {
		return;
	}

	tensor_bruteforce_and
		<T, nr_limbs, d, TM, WARPS, RC>
		<<<tensor_grid(e1, e2, WARPS * TM * WMMA_M, RC), WARPS * 32u, 0, stream>>>
		(L, R, e1, e2, f1, f2);
	checkCudaErrors(cudaGetLastError());
}

/// launches `tensor_bruteforce_hybrid` (variant F, AND + CUDA core warps) on
/// `stream` with the grid of `tensor_grid` and (WARPS + POPC_WARPS)*32
/// threads per CTA. Does not synchronize.
/// \tparam T base limb type of the lists, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per element; n =
/// 		nr_limbs*sizeof(T)*8 must be a multiple of 128.
/// \tparam d maximal hamming distance of a solution, d < n
/// \tparam TM number of 8-row L tiles per tensor core warp in registers
/// 		(default 2)
/// \tparam WARPS number of tensor core warps per CTA (default 8)
/// \tparam RC number of elements of R per shared memory stage, multiple of 8
/// 		(default 256)
/// \tparam POPC_WARPS number of CUDA core warps per CTA, > 0; WARPS*TM*8 must
/// 		be a multiple of POPC_WARPS*32 (default 4)
/// \tparam RC_POPC number of elements at the end of each R chunk processed by
/// 		the CUDA core warps, 0 < RC_POPC <= RC, RC - RC_POPC multiple of 8
/// 		(default 112). Tune it per GPU, the optimum is sharp.
/// \param L first list, `e1` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param R second list, `e2` contiguous elements in device or managed
/// 		memory, 16 byte aligned
/// \param e1 number of elements of L to search
/// \param e2 number of elements of R to search
/// \param f1 output: position of the solution in L. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param f2 output: position of the solution in R. Must be init with -1 (device
/// 		or managed memory); only valid after the stream is synchronized.
/// \param stream CUDA stream to launch on (default: the legacy default stream)
/// NOTE: nothing is launched if one of the lists is empty.
/// NOTE: launch errors exit the program via `checkCudaErrors`.
/// See `dtensor_bruteforce_and` for the details of all parameters.
template<typename T, const uint32_t nr_limbs, const uint32_t d,
		 const uint32_t TM=2, const uint32_t WARPS=8, const uint32_t RC=256,
		 const uint32_t POPC_WARPS=4, const uint32_t RC_POPC=112>
void launch_tensor_bruteforce_hybrid(const T *L, const T *R,
									 const size_t e1, const size_t e2,
									 size_t *f1, size_t *f2,
									 cudaStream_t stream = 0) {
	if ((e1 == 0) || (e2 == 0)) {
		return;
	}

	tensor_bruteforce_hybrid
		<T, nr_limbs, d, TM, WARPS, RC, POPC_WARPS, RC_POPC>
		<<<tensor_grid(e1, e2, WARPS * TM * WMMA_M, RC), (WARPS + POPC_WARPS) * 32u, 0, stream>>>
		(L, R, e1, e2, f1, f2);
	checkCudaErrors(cudaGetLastError());
}

} // namespace cryptanalysislib::nn::tensor
