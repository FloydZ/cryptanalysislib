#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "helper.cuh"

/// Locality sensitive filtering on full length elements:
///
///   f_i(x) := wt(x ^ c_i) <= t,      i = 0, ..., S-1,   S <= 64
///
/// for S random centers `c_i` and a common threshold `t`. Two lists L1, L2
/// are processed in three steps:
///
/// 	1. `launch_filter`: computes for every element x the mask
/// 	   m(x) = sum_i f_i(x) 2^i, and
/// 	2. inserts x into the fixed capacity bucket i for every set bit i.
/// 	   Both steps are fused into one kernel, `filter_kernel`.
/// 	3. `launch_join`: outputs every pair (x, y) in L1 x L2 with
/// 	   m(x) & m(y) != 0 exactly once: in the bucket of the lowest common
/// 	   filter, i.e. bucket c only emits (x, y) if ctz(m(x) & m(y)) == c.
///
/// Step 1 is a binary matrix multiplication (|L| x n) * (n x S), with the
/// threshold applied in the epilogue. It is implemented on the tensor cores
/// and on the CUDA cores (`__popc`). `filter_kernel` can run both kinds of
/// warps in the same CTA (TW tensor warps, PW popc warps), which pull their
/// work from the same queue (see below), so no tuning of the split is needed.
///
/// CUDA core path (`popc_mask`): one element per thread in registers. The
/// centers are part of the kernel parameters (`Filters`), and all loops over
/// the centers are unrolled, hence every `x ^ c` is a single LOP3 with a
/// constant bank operand (no loads at all). The threshold is folded into the
/// accumulation (v = sum popc - (t+1)) and the sign bit of v is shifted into
/// the mask with one funnel shift.
///
/// Tensor core path (`tensor_mask`): via
///
///   wt(x ^ c) = wt(x) + wt(c) - 2 popc(x & c) <= t
///   <=>  D := wt(x) + (wt(c) - t - 1) - 2 popc(x & c) < 0,
///
/// i.e. the sign bit of D is the result bit. D = A*B + C is computed by the
/// MMA itself: the accumulator is initialized with C = wt(x) + wt(c) - t - 1,
/// and each common bit of x and c contributes -2 (see `TensorOp`). The
/// elements are the A operand (16 per m-tile), loaded directly from global
/// memory, the centers the B operand (n-tiles of 8 centers, prepared once per
/// CTA). The centers are permuted over the columns, such that the 2*NT result
/// bits of each thread (per row) are consecutive bits of the final mask (see
/// `TensorCenters`).
///
/// Throughput of the MMAs on a RTX 5090 (sm_120), in bit pairs/s:
/// 	b1 AND (emulated: 4 IMMA.U8 + bit unpacking per mma)  ~290 T
/// 	s8                                                     ~440 T
/// 	mxf4 (sm_120a only)                                    ~890 T
/// 	__popc                                                 ~245 T (7.7 Tpopc/s)
/// The b1 MMA and __popc share execution resources on sm_120, so running both
/// kinds of warps together gains little.
///
/// `filter_kernel` on a RTX 5090, n=128, S=64, |L|=2^24, p=1/32, in pairs/s
/// (only masks / masks + buckets), 8 warps per CTA:
/// 	popc  1.55 T / 1.39 T
/// 	b1    1.49 T / 1.15 T
/// 	s8    2.66 T / 2.06 T
/// 	mxf4  3.95 T / 2.78 T   (1.48 TB/s / 1.39 TB/s of DRAM traffic; a
/// 	                         device memcpy reaches 1.53 TB/s)
/// i.e. with mxf4 both steps are memory bound for n <= 256 (n=512: 1.14 T /
/// 0.98 T). Mixing tensor and popc warps is slower for mxf4 and with buckets,
/// hence the default is tensor warps only. `join_kernel` writes the pairs with
/// ~1.5 TB/s. See `DISABLED_filter` and `DISABLED_join` in
/// tests/nn/full_length.cu.
///
/// Bucketing (`filter_kernel`): each CTA processes rounds of `E` consecutive
/// elements. The warps grab units of 32 elements via an atomic counter in
/// shared memory (tensor and popc warps alike), and insert their hits into a
/// staging area in shared memory: per bucket `scap` 16 bit row offsets. At the
/// end of each round, one global atomic per bucket reserves the slots, and
/// the staging area is copied, one warp per bucket. A hit which does not fit
/// into the staging area is inserted directly via a global atomic. The buckets
/// only store the position of the element; its mask is in `M`.
///
/// Layout of the elements: `nr_limbs` limbs of type `T` per element, stored
/// contiguously. They are read as W = nr_limbs*sizeof(T)/4 32 bit words; the
/// centers must use the same layout. Elements of n < 32*W bits must be zero
/// padded (centers as well).
namespace cryptanalysislib::nn::full_length {

/// maximal number of filters, as the masks are 64 bit
constexpr uint32_t MAX_FILTERS = 64;

/// the MMA used by the tensor core warps of `filter_kernel`. Each variant
/// encodes the bits of x and c, such that every common bit contributes -2 to
/// the accumulator:
/// 	B1:   mma .b1 .and.popc, the epilogue computes wt(x) + kc - 2 acc
/// 	      (native on sm_80 - sm_89, emulated on sm_90+).
/// 	S8:   mma .s8 m16n8k32: x bits as bytes {0, -1} (PRMT sign replication),
/// 	      c bits as bytes {0, 2}. sm_80+.
/// 	MXF4: mma .kind::mxf4 m16n8k64 (e2m1, all scales 1): x bits as nibbles
/// 	      {0, a} with a in {0.5, 1, 2} (a mask per bit position), c bits as
/// 	      {0, -2/a}. sm_120a only.
/// 	AUTO: MXF4 if available, else S8 on sm_90+, else B1.
/// If the chosen variant is not available on the architecture the kernel is
/// compiled for, MXF4 falls back to S8. Below sm_80 the tensor warps use the
/// CUDA cores (NONE).
enum class TensorOp { AUTO, B1, S8, MXF4, NONE };

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define FULL_LENGTH_MXF4 1
#endif

/// \return the variant `OP` resolves to on the architecture the device code is
/// 		compiled for. Only meaningful in device code.
template<const TensorOp OP>
__host__ __device__ constexpr TensorOp effective_op() {
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 800
	return TensorOp::NONE;
#else
#if defined(FULL_LENGTH_MXF4)
	constexpr bool mxf4 = true;
#else
	constexpr bool mxf4 = false;
#endif
	if constexpr (OP == TensorOp::AUTO) {
		return mxf4 ? TensorOp::MXF4 : (__CUDA_ARCH__ >= 900 ? TensorOp::S8 : TensorOp::B1);
	} else if constexpr ((OP == TensorOp::MXF4) && !mxf4) {
		return TensorOp::S8;
	} else {
		return OP;
	}
#endif
#else
	return OP;
#endif
}

/// the S centers and the common threshold. Passed by value as kernel parameter,
/// hence the centers are read from the constant bank.
/// NOTE: kernel parameters > 4 KiB (n=512, S=64) need CUDA 12.1+ and sm_70+.
/// \tparam W number of 32 bit words per element
/// \tparam S number of filters, 0 < S <= 64
template<const uint32_t W, const uint32_t S>
struct Filters {
	static_assert(W > 0);
	static_assert((S > 0) && (S <= MAX_FILTERS));

	/// c[i] is the center of filter i
	uint32_t c[S][W];
	/// filter i matches x if wt(x ^ c[i]) <= t
	uint32_t t;
};

/// \tparam T limb type of the centers, e.g. uint32_t or uint64_t
/// \tparam nr_limbs number of limbs of type `T` per center
/// \tparam S number of filters
/// \param centers S centers in host memory, stored contiguously
/// \param t threshold of all filters
/// \return the kernel parameter for `launch_filter`
template<typename T, const uint32_t nr_limbs, const uint32_t S>
Filters<nr_limbs * sizeof(T) / 4u, S> make_filters(const T *centers, const uint32_t t) noexcept {
	static_assert((nr_limbs * sizeof(T)) % 4u == 0);
	constexpr uint32_t W = nr_limbs * sizeof(T) / 4u;
	Filters<W, S> P;
	const uint32_t *c = (const uint32_t *)centers;
	for (uint32_t i = 0; i < S; i++) {
		for (uint32_t w = 0; w < W; w++) {
			P.c[i][w] = c[i * W + w];
		}
	}
	P.t = t;
	return P;
}

/// `nr_buckets` fixed capacity buckets in device memory. Entry `e` of bucket
/// `c` is at position c*capacity + e of `index`.
struct Buckets {
	/// position of the element in its list
	uint32_t *index = nullptr;
	/// number of elements inserted into each bucket. NOTE: can be larger than
	/// `capacity`, then only the first `capacity` (in no specific order) were
	/// stored, the remaining ones are lost.
	uint32_t *count = nullptr;
	uint32_t capacity = 0;
	uint32_t nr_buckets = 0;
};

/// allocates `nr_buckets` buckets with `capacity` entries each in device
/// memory. The counts are set to zero.
inline Buckets alloc_buckets(const uint32_t nr_buckets, const uint32_t capacity) noexcept {
	assert(nr_buckets <= MAX_FILTERS);
	Buckets B;
	B.capacity = capacity;
	B.nr_buckets = nr_buckets;
	checkCudaErrors(cudaMalloc(&B.index, sizeof(uint32_t) * size_t(nr_buckets) * capacity));
	checkCudaErrors(cudaMalloc(&B.count, sizeof(uint32_t) * nr_buckets));
	checkCudaErrors(cudaMemset(B.count, 0, sizeof(uint32_t) * nr_buckets));
	return B;
}

inline void free_buckets(Buckets &B) noexcept {
	cudaFree(B.index);
	cudaFree(B.count);
	B = Buckets{};
}

/// \return Pr[wt(x ^ c) <= t] = sum_{k <= t} binom(n, k) / 2^n for a uniform
/// 		random x of n bits, i.e. the probability that a random element
/// 		passes a filter
inline double filter_probability(const uint32_t n, const uint32_t t) noexcept {
	double p = 0.;
	for (uint32_t k = 0; (k <= t) && (k <= n); k++) {
		p += std::exp(std::lgamma(n + 1.) - std::lgamma(k + 1.) - std::lgamma(n - k + 1.) - n * std::log(2.));
	}
	return p > 1. ? 1. : p;
}

/// \return bucket capacity for a list of `N` uniform random elements of `n`
/// 		bits: expectation + `sigmas` standard deviations (+16)
inline uint32_t bucket_capacity(const size_t N, const uint32_t n, const uint32_t t,
                                const double sigmas = 8.) noexcept {
	const double p = filter_probability(n, t);
	const double mu = double(N) * p;
	const double c = mu + sigmas * std::sqrt(mu * (1. - p)) + 16.;
	return c > double(N) ? uint32_t(N) : uint32_t(c);
}

/// loads one element (W 32 bit words) into registers, with the widest loads
/// the alignment of the element allows.
template<const uint32_t W>
__device__ __forceinline__ void load_element(uint32_t (&x)[W], const uint32_t *L, const uint32_t row) {
	const uint32_t *p = L + size_t(row) * W;
	if constexpr ((W % 4u) == 0) {
		#pragma unroll
		for (uint32_t i = 0; i < W / 4u; i++) {
			const uint4 v = __ldg((const uint4 *)p + i);
			x[4 * i + 0] = v.x;
			x[4 * i + 1] = v.y;
			x[4 * i + 2] = v.z;
			x[4 * i + 3] = v.w;
		}
	} else if constexpr ((W % 2u) == 0) {
		#pragma unroll
		for (uint32_t i = 0; i < W / 2u; i++) {
			const uint2 v = __ldg((const uint2 *)p + i);
			x[2 * i + 0] = v.x;
			x[2 * i + 1] = v.y;
		}
	} else {
		#pragma unroll
		for (uint32_t i = 0; i < W; i++) {
			x[i] = __ldg(p + i);
		}
	}
}

/// CUDA core path: computes the filter mask of one element.
/// \param x the element
/// \param P the filters, must be (a reference to) the kernel parameter: all
/// 		indices into `P.c` are compile time constants, so the centers are
/// 		constant bank operands of the LOP3s.
/// \return bit i is set iff wt(x ^ P.c[i]) <= P.t
template<const uint32_t W, const uint32_t S>
__device__ __forceinline__ uint64_t popc_mask(const uint32_t (&x)[W], const Filters<W, S> &P) {
	const int t1 = int(P.t) + 1;
	uint32_t lo = 0, hi = 0;
	// descending, such that center i ends up in bit i
	#pragma unroll
	for (int i = S - 1; i >= 0; i--) {
		int v = -t1;
		#pragma unroll
		for (uint32_t w = 0; w < W; w++) {
			v += __popc(x[w] ^ P.c[i][w]);
		}

		// v < 0 <=> wt(x ^ c_i) <= t: shift the sign bit in
		if (i >= 32) {
			hi = __funnelshift_l(uint32_t(v), hi, 1);
		} else {
			lo = __funnelshift_l(uint32_t(v), lo, 1);
		}
	}
	return (uint64_t(hi) << 32u) | lo;
}

/// c += popc(A & B) on a 16x8x128 bit tile.
__device__ __forceinline__ void mma_b1_k128(int (&c)[4], const uint32_t a0, const uint32_t a1,
                                            const uint32_t b0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	asm("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
	    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
	    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
	    : "r"(a0), "r"(a1), "r"(b0));
#else
	(void)c; (void)a0; (void)a1; (void)b0;
#endif
}

/// c += popc(A & B) on a 16x8x256 bit tile.
__device__ __forceinline__ void mma_b1_k256(int (&c)[4], const uint32_t a0, const uint32_t a1,
                                            const uint32_t a2, const uint32_t a3,
                                            const uint32_t b0, const uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
	    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
	    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
	    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
	(void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

/// c += A * B on a 16x8x32 s8 tile.
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t a0, const uint32_t a1,
                                       const uint32_t a2, const uint32_t a3,
                                       const uint32_t b0, const uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
	    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
	    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
	    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
	(void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

/// c += A * B on a 16x8x64 e2m1 tile, all scale factors 1 (ue8m0 0x7f).
__device__ __forceinline__ void mma_mxf4(float (&c)[4], const uint32_t a0, const uint32_t a1,
                                         const uint32_t a2, const uint32_t a3,
                                         const uint32_t b0, const uint32_t b1) {
#if defined(FULL_LENGTH_MXF4)
	asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
	    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%11}, %10, {%11,%11};\n"
	    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
	    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
	      "r"(0x7f7f7f7fu), "h"((uint16_t)0));
#else
	(void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

/// number of mma k-steps per 32 bit word (each step consumes 8 bits of the
/// word of each thread), 0 if the B fragments are kept in registers
template<const TensorOp op>
__host__ __device__ constexpr uint32_t mma_steps() {
	return op == TensorOp::S8 ? 4u : op == TensorOp::MXF4 ? 2u : 0u;
}

/// S8: the B fragments of one word of a center, 4 steps of (b0, b1). Step s,
/// register r of the pair contains the bits 7-(2s+r) + 8i (i = 0..3) of `w`
/// as bytes {0, 2}, matching `expand_s8_a`.
__device__ __forceinline__ void expand_s8_b(uint2 (&f)[4], const uint32_t w) {
	#pragma unroll
	for (uint32_t s = 0; s < 4u; s++) {
		f[s].x = ((w << (2u * s)) >> 6u) & 0x02020202u;
		f[s].y = ((w << (2u * s + 1u)) >> 6u) & 0x02020202u;
	}
}

/// S8: the A fragments of one word of an element: e[r] contains the bits
/// 7-r + 8i (i = 0..3) of `w` as bytes {0, -1}, i.e. the sign bit of each byte
/// of w << r, replicated by PRMT.
/// NOTE: `__byte_perm` only uses the low 3 bits of each selector nibble, hence
/// 		the sign replication needs the inline PTX.
__device__ __forceinline__ void expand_s8_a(uint32_t (&e)[8], const uint32_t w) {
	#pragma unroll
	for (uint32_t r = 0; r < 8u; r++) {
		asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(e[r]) : "r"(w << r));
	}
}

/// MXF4: the B fragments of one word of a center, 2 steps of (b0, b1). Step s,
/// register r of the pair contains the bits 4i + (2s+r) (i = 0..7) of `w` as
/// e2m1 nibbles {0, -2/a}, with a the value `expand_mxf4_a` uses for the bit.
__device__ __forceinline__ void expand_mxf4_b(uint2 (&f)[2], const uint32_t w) {
	// -4 (0xE), -2 (0xC), -1 (0xA), -1 (0xA)
	f[0].x = (w & 0x11111111u) * 0xEu;
	f[0].y = ((w >> 1u) & 0x11111111u) * 0xCu;
	f[1].x = ((w >> 2u) & 0x11111111u) * 0xAu;
	f[1].y = ((w >> 3u) & 0x11111111u) * 0xAu;
}

/// MXF4: the A fragments of one word of an element: e[r] contains the bits
/// 4i + r (i = 0..7) of `w` as e2m1 nibbles: 0.5 (0x1), 1 (0x2), 2 (0x4) and
/// 2 (0x4 = bit 3 shifted down, as 0x8 is -0).
__device__ __forceinline__ void expand_mxf4_a(uint32_t (&e)[4], const uint32_t w) {
	e[0] = w & 0x11111111u;
	e[1] = w & 0x22222222u;
	e[2] = w & 0x44444444u;
	e[3] = (w >> 1u) & 0x44444444u;
}

/// per CTA / per thread state of the tensor core warps.
///
/// Fragment layout (PTX ISA, mma.m16n8k*), lane = 4g + q:
/// 	A: a0 = row g, a1 = row g+8, a2/a3 the same rows, second half of k
/// 	B: b0 = column g, b1 = column g, second half of k
/// 	C: c0 = (g, 2q), c1 = (g, 2q+1), c2 = (g+8, 2q), c3 = (g+8, 2q+1)
/// Thread q always provides the 32 bit words q, q+4, ... (one per 128 bit
/// chunk) of its rows of A, and the same words of its column of B. Within a
/// word, the bits are distributed over the k-steps of the MMA by `expand_*`,
/// in the same way for A and B, so the dot product runs over all bits.
///
/// Column `col` of n-tile `j` holds the center 2*NT*(col/2) + 2*j + col%2. So
/// the thread with q holds the centers 2*NT*q + [0, 2*NT) (per row), which
/// are consecutive bits in the final mask.
template<const uint32_t W, const uint32_t S, const TensorOp op>
struct TensorCenters {
	/// number of n-tiles (8 centers each)
	constexpr static uint32_t NT = (S + 7u) / 8u;
	/// number of 128 bit chunks per element
	constexpr static uint32_t Q = (W + 3u) / 4u;
	/// B fragments in shared memory: (j, k, s, lane), see `mma_steps`
	constexpr static uint32_t SMEM = NT * Q * mma_steps<op>() * 32u;

	/// B1: b[j][k]: B fragment of n-tile j, chunk k
	uint32_t b[op == TensorOp::B1 ? NT : 1][op == TensorOp::B1 ? Q : 1];
	/// kc[j][b] = wt(c) - t - 1 of the center in column 2q+b of n-tile j.
	/// Large for padding columns, such that they never match.
	int kc[NT][2];
	/// MXF4: kc as float
	float kcf[op == TensorOp::MXF4 ? NT : 1][2];
};

/// \return center of column `col` of n-tile `j`
template<const uint32_t NT>
__device__ __forceinline__ uint32_t column_center(const uint32_t j, const uint32_t col) {
	return 2u * NT * (col / 2u) + 2u * j + (col % 2u);
}

/// initializes the B fragments (shared memory `sB`, written by all tensor
/// warps cooperatively) and the per thread constants.
/// \param tid thread id within the tensor warps
/// \param nr_threads number of threads of the tensor warps
/// NOTE: `sB` must not be read before a __syncthreads.
template<const uint32_t W, const uint32_t S, const TensorOp op>
__device__ __forceinline__ void tensor_setup(TensorCenters<W, S, op> &tc, uint2 *sB,
                                             const Filters<W, S> &P,
                                             const uint32_t tid, const uint32_t nr_threads) {
	using TC = TensorCenters<W, S, op>;
	constexpr uint32_t NT = TC::NT, Q = TC::Q, STEPS = mma_steps<op>();
	const uint32_t lane = threadIdx.x % 32u, q = lane % 4u;

	if constexpr (STEPS > 0) {
		for (uint32_t e = tid; e < NT * Q * 32u; e += nr_threads) {
			const uint32_t l = e % 32u, k = (e / 32u) % Q, j = e / (32u * Q);
			const uint32_t center = column_center<NT>(j, l / 4u), w = 4u * k + (l % 4u);
			const uint32_t v = ((center < S) && (w < W)) ? P.c[center][w] : 0u;
			if constexpr (op == TensorOp::S8) {
				uint2 f[4];
				expand_s8_b(f, v);
				#pragma unroll
				for (uint32_t s = 0; s < 4u; s++) {
					sB[((j * Q + k) * 4u + s) * 32u + l] = f[s];
				}
			} else {
				uint2 f[2];
				expand_mxf4_b(f, v);
				#pragma unroll
				for (uint32_t s = 0; s < 2u; s++) {
					sB[((j * Q + k) * 2u + s) * 32u + l] = f[s];
				}
			}
		}
	}

	#pragma unroll
	for (uint32_t j = 0; j < NT; j++) {
		if constexpr (op == TensorOp::B1) {
			const uint32_t center = column_center<NT>(j, lane / 4u);
			#pragma unroll
			for (uint32_t k = 0; k < Q; k++) {
				const uint32_t w = 4u * k + q;
				tc.b[j][k] = ((center < S) && (w < W)) ? P.c[center][w] : 0u;
			}
		}

		#pragma unroll
		for (uint32_t b = 0; b < 2u; b++) {
			const uint32_t center = column_center<NT>(j, 2u * q + b);
			int wc = 0;
			if (center < S) {
				#pragma unroll
				for (uint32_t w = 0; w < W; w++) {
					wc += __popc(P.c[center][w]);
				}
			}
			tc.kc[j][b] = (center < S) ? wc - int(P.t) - 1 : (1 << 20);
			if constexpr (op == TensorOp::MXF4) {
				tc.kcf[j][b] = float(tc.kc[j][b]);
			}
		}
	}
}

/// the words of a unit (32 elements) a thread of a tensor core warp needs
template<const uint32_t W, const TensorOp op>
struct TensorUnit {
	constexpr static uint32_t Q = (W + 3u) / 4u;
	/// a[tile][half][chunk]: row 16*tile + 8*half + g, word 4*chunk + q
	uint32_t a[op == TensorOp::NONE ? 1 : 2][2][Q];
	/// NONE: the element of row `lane`
	uint32_t x[op == TensorOp::NONE ? W : 1];
};

/// loads the unit row0 + [0, 32) for the tensor core path
/// \param L the list, as 32 bit words
/// \param N number of elements in `L`; rows >= N are zero
template<const uint32_t W, const TensorOp op>
__device__ __forceinline__ void tensor_load(TensorUnit<W, op> &U, const uint32_t *L,
                                            const uint32_t N, const uint32_t row0) {
	const uint32_t lane = threadIdx.x % 32u;
	if constexpr (op == TensorOp::NONE) {
		if (row0 + lane < N) {
			load_element<W>(U.x, L, row0 + lane);
		} else {
			#pragma unroll
			for (uint32_t w = 0; w < W; w++) { U.x[w] = 0; }
		}
	} else {
		const uint32_t g = lane / 4u, q = lane % 4u;
		#pragma unroll
		for (uint32_t t = 0; t < 2u; t++) {
			#pragma unroll
			for (uint32_t h = 0; h < 2u; h++) {
				const uint32_t row = row0 + 16u * t + 8u * h + g;
				#pragma unroll
				for (uint32_t k = 0; k < U.Q; k++) {
					const uint32_t w = 4u * k + q;
					const bool valid = (row < N) && (((W % 4u) == 0) || (w < W));
					U.a[t][h][k] = valid ? __ldg(L + size_t(row) * W + w) : 0u;
				}
			}
		}
	}
}

/// tensor core path: computes the filter masks of the 32 elements of a unit
/// (two 16 row m-tiles), one per thread.
/// \param tc state of the thread, see `tensor_setup`
/// \param sB the B fragments in shared memory (S8, MXF4)
/// \param P the filters (only used below sm_80)
/// \param U the unit, see `tensor_load`
/// \param r output: the row (relative to the unit) whose mask is returned
/// \return the filter mask of row r of the unit
template<const uint32_t W, const uint32_t S, const TensorOp op>
__device__ __forceinline__ uint64_t tensor_mask(const TensorCenters<W, S, op> &tc,
                                                const uint2 *sB,
                                                const Filters<W, S> &P,
                                                const TensorUnit<W, op> &U,
                                                uint32_t &r) {
	using TC = TensorCenters<W, S, op>;
	constexpr uint32_t NT = TC::NT, Q = TC::Q;
	const uint32_t lane = threadIdx.x % 32u;

	if constexpr (op == TensorOp::NONE) {
		(void)tc; (void)sB;
		r = lane;
		return popc_mask<W, S>(U.x, P);
	} else {
		(void)P;
		const uint32_t g = lane / 4u, q = lane % 4u;
		const auto &a = U.a;

		// wx[tile][half]: weight of the row
		int wx[2][2];
		#pragma unroll
		for (uint32_t t = 0; t < 2u; t++) {
			#pragma unroll
			for (uint32_t h = 0; h < 2u; h++) {
				wx[t][h] = 0;
				#pragma unroll
				for (uint32_t k = 0; k < Q; k++) {
					wx[t][h] += __popc(a[t][h][k]);
				}
				wx[t][h] += __shfl_xor_sync(0xffffffffu, wx[t][h], 1);
				wx[t][h] += __shfl_xor_sync(0xffffffffu, wx[t][h], 2);
			}
		}

		// m[tile][half]: the 2*NT result bits of the row; (j, b) in bit 2j + b,
		// hence everything is shifted in in descending order
		uint32_t m[2][2] = {{0, 0}, {0, 0}};
		if constexpr (op == TensorOp::B1) {
			#pragma unroll
			for (int j = NT - 1; j >= 0; j--) {
				#pragma unroll
				for (uint32_t t = 0; t < 2u; t++) {
					int c[4] = {0, 0, 0, 0};
					#pragma unroll
					for (uint32_t k = 0; k + 1u < Q; k += 2u) {
						mma_b1_k256(c, a[t][0][k], a[t][1][k], a[t][0][k + 1u], a[t][1][k + 1u],
						            tc.b[j][k], tc.b[j][k + 1u]);
					}
					if constexpr ((Q % 2u) == 1u) {
						mma_b1_k128(c, a[t][0][Q - 1u], a[t][1][Q - 1u], tc.b[j][Q - 1u]);
					}

					// < 0 <=> wt(x ^ c) <= t
					m[t][0] = __funnelshift_l(uint32_t(wx[t][0] + tc.kc[j][1] - 2 * c[1]), m[t][0], 1);
					m[t][0] = __funnelshift_l(uint32_t(wx[t][0] + tc.kc[j][0] - 2 * c[0]), m[t][0], 1);
					m[t][1] = __funnelshift_l(uint32_t(wx[t][1] + tc.kc[j][1] - 2 * c[3]), m[t][1], 1);
					m[t][1] = __funnelshift_l(uint32_t(wx[t][1] + tc.kc[j][0] - 2 * c[2]), m[t][1], 1);
				}
			}
		} else if constexpr (op == TensorOp::MXF4) {
			// all words expanded once (16 Q registers), then one n-tile after
			// the other
			uint32_t e[2][2][Q][4];
			float wf[2][2];
			#pragma unroll
			for (uint32_t t = 0; t < 2u; t++) {
				#pragma unroll
				for (uint32_t h = 0; h < 2u; h++) {
					wf[t][h] = float(wx[t][h]);
					#pragma unroll
					for (uint32_t k = 0; k < Q; k++) {
						expand_mxf4_a(e[t][h][k], a[t][h][k]);
					}
				}
			}

			#pragma unroll
			for (int j = NT - 1; j >= 0; j--) {
				float d[2][4];
				#pragma unroll
				for (uint32_t t = 0; t < 2u; t++) {
					d[t][0] = wf[t][0] + tc.kcf[j][0];
					d[t][1] = wf[t][0] + tc.kcf[j][1];
					d[t][2] = wf[t][1] + tc.kcf[j][0];
					d[t][3] = wf[t][1] + tc.kcf[j][1];
				}
				#pragma unroll
				for (uint32_t k = 0; k < Q; k++) {
					#pragma unroll
					for (uint32_t s = 0; s < 2u; s++) {
						const uint2 b = sB[((j * Q + k) * 2u + s) * 32u + lane];
						#pragma unroll
						for (uint32_t t = 0; t < 2u; t++) {
							mma_mxf4(d[t], e[t][0][k][2u * s], e[t][1][k][2u * s],
							         e[t][0][k][2u * s + 1u], e[t][1][k][2u * s + 1u], b.x, b.y);
						}
					}
				}
				#pragma unroll
				for (uint32_t t = 0; t < 2u; t++) {
					m[t][0] = __funnelshift_l(__float_as_uint(d[t][1]), m[t][0], 1);
					m[t][0] = __funnelshift_l(__float_as_uint(d[t][0]), m[t][0], 1);
					m[t][1] = __funnelshift_l(__float_as_uint(d[t][3]), m[t][1], 1);
					m[t][1] = __funnelshift_l(__float_as_uint(d[t][2]), m[t][1], 1);
				}
			}
		} else {
			// S8: the expansion is twice as large, hence one word after the
			// other, with all accumulators live
			int d[2][NT][4];
			#pragma unroll
			for (uint32_t t = 0; t < 2u; t++) {
				#pragma unroll
				for (uint32_t j = 0; j < NT; j++) {
					d[t][j][0] = wx[t][0] + tc.kc[j][0];
					d[t][j][1] = wx[t][0] + tc.kc[j][1];
					d[t][j][2] = wx[t][1] + tc.kc[j][0];
					d[t][j][3] = wx[t][1] + tc.kc[j][1];
				}
			}
			#pragma unroll
			for (uint32_t k = 0; k < Q; k++) {
				uint32_t e[2][2][8];
				#pragma unroll
				for (uint32_t t = 0; t < 2u; t++) {
					#pragma unroll
					for (uint32_t h = 0; h < 2u; h++) {
						expand_s8_a(e[t][h], a[t][h][k]);
					}
				}
				#pragma unroll
				for (uint32_t j = 0; j < NT; j++) {
					#pragma unroll
					for (uint32_t s = 0; s < 4u; s++) {
						const uint2 b = sB[((j * Q + k) * 4u + s) * 32u + lane];
						#pragma unroll
						for (uint32_t t = 0; t < 2u; t++) {
							mma_s8(d[t][j], e[t][0][2u * s], e[t][1][2u * s],
							       e[t][0][2u * s + 1u], e[t][1][2u * s + 1u], b.x, b.y);
						}
					}
				}
			}
			#pragma unroll
			for (int j = NT - 1; j >= 0; j--) {
				#pragma unroll
				for (uint32_t t = 0; t < 2u; t++) {
					m[t][0] = __funnelshift_l(uint32_t(d[t][j][1]), m[t][0], 1);
					m[t][0] = __funnelshift_l(uint32_t(d[t][j][0]), m[t][0], 1);
					m[t][1] = __funnelshift_l(uint32_t(d[t][j][3]), m[t][1], 1);
					m[t][1] = __funnelshift_l(uint32_t(d[t][j][2]), m[t][1], 1);
				}
			}
		}

		// thread q collects row 8*(q%2) + g of tile q/2 from the 4 threads of its quad
		const uint32_t v0 = m[0][0] | (m[0][1] << 16u);
		const uint32_t v1 = m[1][0] | (m[1][1] << 16u);
		const uint32_t tile = q / 2u, half = q % 2u;
		uint64_t mask = 0;
		#pragma unroll
		for (uint32_t qq = 0; qq < 4u; qq++) {
			const uint32_t src = (lane & ~3u) | qq;
			const uint32_t x0 = __shfl_sync(0xffffffffu, v0, src);
			const uint32_t x1 = __shfl_sync(0xffffffffu, v1, src);
			const uint32_t x = tile ? x1 : x0;
			mask |= uint64_t((x >> (16u * half)) & 0xffffu) << (2u * NT * qq);
		}
		r = 16u * tile + 8u * half + g;
		return mask;
	}
}

/// \return the next unit of the round, the same for all threads of the warp
__device__ __forceinline__ uint32_t grab_unit(uint32_t *sunit) {
	uint32_t u = 0;
	if ((threadIdx.x % 32u) == 0) {
		u = atomicAdd(sunit, 1u);
	}
	return __shfl_sync(0xffffffffu, u, 0);
}

/// writes the mask `m` of row `r` of the round into `M`, and inserts the
/// row into the staging area of each bucket of its set bits. Full staging
/// buckets are bypassed via a global atomic.
/// \param rows number of rows in this round; nothing is done for r >= rows
template<const uint32_t S>
__device__ __forceinline__ void store_mask(const uint64_t m, const uint32_t r,
                                           const uint32_t rows, const uint32_t row_base,
                                           uint64_t *M, const Buckets &B,
                                           uint32_t *scnt, uint16_t *sidx,
                                           const uint32_t scap) {
	if (r >= rows) {
		return;
	}
	M[row_base + r] = m;
	// masks only
	if (B.nr_buckets == 0) {
		return;
	}
	for (uint64_t mm = m; mm; mm &= mm - 1u) {
		const uint32_t c = __ffsll(mm) - 1u;
		const uint32_t s = atomicAdd(scnt + c, 1u);
		if (s < scap) {
			sidx[c * scap + s] = r;
		} else {
			const uint32_t gs = atomicAdd(B.count + c, 1u);
			if (gs < B.capacity) {
				B.index[size_t(c) * B.capacity + gs] = row_base + r;
			}
		}
	}
}

/// computes the filter masks of the first `N` elements of `L`, writes them
/// into `M` and inserts each element into the buckets of its set bits. See the
/// top of this file for the details.
/// The tensor core warps load the next unit before they compute the current
/// one, to double the bytes in flight.
/// \tparam T limb type of the elements
/// \tparam nr_limbs limbs per element
/// \tparam S number of filters
/// \tparam TW number of tensor core warps per CTA
/// \tparam PW number of CUDA core (popc) warps per CTA, TW + PW > 0
/// \tparam E number of elements per round, multiple of 32, <= 2^16
/// \tparam OP MMA of the tensor core warps
/// \param L the list, 16 byte aligned
/// \param N number of elements, < 2^32
/// \param P the filters
/// \param M output: N masks
/// \param B output: the buckets (S of them), or B.nr_buckets == 0 to only
/// 		compute the masks. The elements are appended, i.e. the counts are
/// 		not reset.
/// \param scap capacity of each bucket in the shared memory staging area
/// NOTE: dynamic shared memory: (2S+1)*4 + S*scap*2 bytes.
/// \return the minimal number of CTAs per SM `filter_kernel` is compiled for,
/// 		i.e. the register limit. Without (or with a too large) limit ptxas
/// 		hoists the centers of the popc path from the constant bank into
/// 		(spilled) registers, and interleaves all n-tiles of the tensor path:
/// 		up to 255 registers and 1 CTA per SM, which is 1.2 - 4x slower.
/// 			popc only:        4 (<= 64 registers for 8 warps)
/// 			MXF4, n <= 128:   3 (<= 85)
/// 			else:             2 (<= 128)
template<const uint32_t TW, const uint32_t PW, const uint32_t W, const TensorOp op>
__host__ __device__ constexpr uint32_t min_blocks() {
	if constexpr (TW == 0) {
		return 4;
	} else if constexpr ((PW == 0) && (op == TensorOp::MXF4) && (W <= 4u)) {
		return 3;
	} else {
		return 2;
	}
}

template<typename T, const uint32_t nr_limbs, const uint32_t S,
         const uint32_t TW, const uint32_t PW, const uint32_t E, const TensorOp OP>
__global__ void __launch_bounds__((TW + PW) * 32u,
                                  min_blocks<TW, PW, nr_limbs * sizeof(T) / 4u, effective_op<OP>()>())
filter_kernel(const T *L, const uint32_t N,
              const Filters<nr_limbs * sizeof(T) / 4u, S> P,
              uint64_t *M, const Buckets B, const uint32_t scap) {
	constexpr uint32_t W = nr_limbs * sizeof(T) / 4u;
	constexpr TensorOp op = effective_op<OP>();
	using TC = TensorCenters<W, S, op>;
	static_assert(TW + PW > 0);
	static_assert(((E % 32u) == 0) && (E > 0) && (E <= (1u << 16u)));
	constexpr uint32_t NR_WARPS = TW + PW;

	__shared__ uint2 sB[(TW > 0) && (TC::SMEM > 0) ? TC::SMEM : 1u];
	extern __shared__ uint32_t full_length_smem[];
	uint32_t *scnt = full_length_smem;
	uint32_t *sbase = scnt + S;
	uint32_t *sunit = sbase + S;
	uint16_t *sidx = (uint16_t *)(sunit + 1);

	const uint32_t *L32 = (const uint32_t *)L;
	const uint32_t tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;
	bool tensor_warp = false;
	TC tc;
	if constexpr (TW > 0) {
		tensor_warp = warp < TW;
		if (tensor_warp) {
			tensor_setup<W, S, op>(tc, sB, P, tid, TW * 32u);
		}
	}

	for (uint32_t chunk = blockIdx.x; size_t(chunk) * E < N; chunk += gridDim.x) {
		const uint32_t row_base = chunk * E;
		const uint32_t rows = (N - row_base) < E ? (N - row_base) : E;

		// everybody is done with the staging area of the previous round
		__syncthreads();
		for (uint32_t c = tid; c < S; c += blockDim.x) {
			scnt[c] = 0;
		}
		if (tid == 0) {
			*sunit = 0;
		}
		__syncthreads();

		if ((TW > 0) && ((PW == 0) || tensor_warp)) {
			TensorUnit<W, op> U, Un;
			uint32_t u = grab_unit(sunit);
			if (u * 32u < rows) {
				tensor_load<W, op>(U, L32, N, row_base + u * 32u);
			}
			while (u * 32u < rows) {
				const uint32_t un = grab_unit(sunit);
				if (un * 32u < rows) {
					tensor_load<W, op>(Un, L32, N, row_base + un * 32u);
				}

				uint32_t r;
				const uint64_t m = tensor_mask<W, S, op>(tc, sB, P, U, r);
				store_mask<S>(m, u * 32u + r, rows, row_base, M, B, scnt, sidx, scap);
				U = Un;
				u = un;
			}
		} else {
			while (true) {
				const uint32_t u = grab_unit(sunit);
				if (u * 32u >= rows) {
					break;
				}

				uint32_t x[W];
				if (u * 32u + lane < rows) {
					load_element<W>(x, L32, row_base + u * 32u + lane);
				} else {
					#pragma unroll
					for (uint32_t w = 0; w < W; w++) { x[w] = 0; }
				}
				const uint64_t m = popc_mask<W, S>(x, P);
				store_mask<S>(m, u * 32u + lane, rows, row_base, M, B, scnt, sidx, scap);
			}
		}
		__syncthreads();

		// reserve the slots of this round
		for (uint32_t c = tid; c < S; c += blockDim.x) {
			const uint32_t k = scnt[c] < scap ? scnt[c] : scap;
			sbase[c] = k ? atomicAdd(B.count + c, k) : 0u;
		}
		__syncthreads();

		// copy the staging area, one warp per bucket
		for (uint32_t c = warp; c < S; c += NR_WARPS) {
			const uint32_t k = scnt[c] < scap ? scnt[c] : scap;
			for (uint32_t i = lane; i < k; i += 32u) {
				const uint32_t s = sbase[c] + i;
				if (s < B.capacity) {
					B.index[size_t(c) * B.capacity + s] = row_base + sidx[c * scap + i];
				}
			}
		}
	}
}

/// launches `filter_kernel` on `stream` as a persistent kernel (as many CTAs
/// as fit on the GPU, at most one per round). Does not synchronize.
/// \tparam T limb type of the elements, e.g. uint32_t or uint64_t
/// \tparam nr_limbs limbs per element
/// \tparam S number of filters, 0 < S <= 64
/// \tparam TW number of tensor core warps per CTA
/// \tparam PW number of CUDA core warps per CTA
/// \tparam E number of elements per round
/// \tparam OP MMA of the tensor core warps, see `TensorOp`
/// \param L the list in device memory, 16 byte aligned
/// \param N number of elements, < 2^32
/// \param P the filters, see `make_filters`
/// \param M output in device memory: N masks
/// \param B output: the buckets, B.nr_buckets must be S; or 0 (e.g.
/// 		`Buckets{}`) to only compute the masks
/// \param p probability that an element passes a filter, only used to size
/// 		the shared memory staging area. Negative: computed via
/// 		`filter_probability(32*W, P.t)`, i.e. for uniform random elements.
/// \param stream CUDA stream
/// \param reset if true the counts of `B` are set to zero first, otherwise
/// 		the elements are appended (e.g. to process a list in parts; the
/// 		indices are relative to `L` then).
template<typename T, const uint32_t nr_limbs, const uint32_t S,
         const uint32_t TW=8, const uint32_t PW=0, const uint32_t E=2048,
         const TensorOp OP=TensorOp::AUTO>
void launch_filter(const T *L, const size_t N,
                   const Filters<nr_limbs * sizeof(T) / 4u, S> &P,
                   uint64_t *M, const Buckets &B,
                   double p = -1., cudaStream_t stream = 0,
                   const bool reset = true) noexcept {
	constexpr uint32_t W = nr_limbs * sizeof(T) / 4u;
	assert(N < (size_t(1u) << 32u));
	assert((B.nr_buckets == S) || (B.nr_buckets == 0));
	if (reset && B.nr_buckets) {
		checkCudaErrors(cudaMemsetAsync(B.count, 0, sizeof(uint32_t) * S, stream));
	}
	if (N == 0) {
		return;
	}

	auto kernel = filter_kernel<T, nr_limbs, S, TW, PW, E, OP>;
	cudaFuncAttributes attr;
	checkCudaErrors(cudaFuncGetAttributes(&attr, kernel));

	// staging area: expectation + 4 standard deviations per round, at most
	// what fits next to the static shared memory into 48 KiB
	if (p < 0.) {
		p = filter_probability(32u * W, P.t);
	}
	const double mu = double(E) * p;
	uint32_t scap = uint32_t(mu + 4. * std::sqrt(mu) + 8.);
	scap = B.nr_buckets ? (scap + 7u) & ~7u : 0u;
	const size_t fixed = (2u * S + 1u) * sizeof(uint32_t);
	const size_t max_smem = 48u * 1024u - attr.sharedSizeBytes;
	assert(fixed <= max_smem);
	if (fixed + size_t(S) * scap * sizeof(uint16_t) > max_smem) {
		scap = (max_smem - fixed) / (S * sizeof(uint16_t));
	}
	const size_t smem = fixed + size_t(S) * scap * sizeof(uint16_t);

	int dev = 0, sms = 0, occ = 0;
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, (TW + PW) * 32u, smem));
	occ = occ > 0 ? occ : 1;
	const size_t rounds = (N + E - 1u) / E;
	const size_t grid = size_t(sms) * occ < rounds ? size_t(sms) * occ : rounds;

	kernel<<<uint32_t(grid), (TW + PW) * 32u, smem, stream>>>(L, uint32_t(N), P, M, B, scap);
	checkCudaErrors(cudaGetLastError());
}

/// outputs all pairs (x, y) in L1 x L2 with m(x) & m(y) != 0 exactly once,
/// as (position in L1, position in L2). Pair (x, y) is found in bucket
/// c = ctz(m(x) & m(y)) only.
///
/// gridDim.y = number of buckets, gridDim.x = parts of BT elements of bucket
/// c of `B2`, one per thread. Each CTA streams bucket c of `B1` in tiles of
/// T1 elements through shared memory. Per tile and warp, the pairs are first
/// only counted, then one global atomic reserves the output range of the
/// warp, and the pairs are written coalesced in a second pass over the tile.
/// \param B1 buckets of the first list
/// \param M1 masks of the first list
/// \param B2 buckets of the second list, same number of buckets
/// \param M2 masks of the second list
/// \param out output: pairs (x, y), or nullptr to only count
/// \param out_capacity number of pairs `out` can hold. Further pairs are
/// 		counted, but not written.
/// \param out_count output: number of pairs, must be init with 0
template<const uint32_t BT, const uint32_t T1>
__global__ void __launch_bounds__(BT)
join_kernel(const Buckets B1, const uint64_t *M1,
            const Buckets B2, const uint64_t *M2,
            uint2 *out, const size_t out_capacity,
            unsigned long long *out_count) {
	__shared__ uint64_t sm[T1];
	__shared__ uint32_t si[T1];

	const uint32_t c = blockIdx.y;
	const uint32_t tid = threadIdx.x, lane = tid % 32u;
	const uint32_t n1 = B1.count[c] < B1.capacity ? B1.count[c] : B1.capacity;
	const uint32_t n2 = B2.count[c] < B2.capacity ? B2.count[c] : B2.capacity;
	if ((n1 == 0) || (blockIdx.x * BT >= n2)) {
		return;
	}

	// only the filters below c are checked: c itself is common
	const uint64_t below = (uint64_t(1u) << c) - 1u;
	const uint32_t j = blockIdx.x * BT + tid;
	// invalid threads never match, as bit c is set in every mask of `B1`
	uint64_t my = ~uint64_t(0);
	uint32_t iy = 0;
	if (j < n2) {
		iy = B2.index[size_t(c) * B2.capacity + j];
		my = __ldg(M2 + iy) & below;
	}
	const uint32_t lanemask_lt = (1u << lane) - 1u;

	for (uint32_t i0 = 0; i0 < n1; i0 += T1) {
		const uint32_t k = (n1 - i0) < T1 ? (n1 - i0) : T1;
		__syncthreads();
		for (uint32_t i = tid; i < k; i += BT) {
			const uint32_t ix = B1.index[size_t(c) * B1.capacity + i0 + i];
			si[i] = ix;
			sm[i] = __ldg(M1 + ix);
		}
		__syncthreads();

		uint32_t cnt = 0;
		#pragma unroll 8
		for (uint32_t x = 0; x < k; x++) {
			cnt += (sm[x] & my) == 0;
		}
		#pragma unroll
		for (uint32_t o = 16; o > 0; o /= 2u) {
			cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
		}
		if (cnt == 0) {
			continue;
		}

		unsigned long long base = 0;
		if (lane == 0) {
			base = atomicAdd(out_count, (unsigned long long)cnt);
		}
		base = __shfl_sync(0xffffffffu, base, 0);
		if (base >= out_capacity) {
			continue;
		}

		uint32_t run = 0;
		#pragma unroll 4
		for (uint32_t x = 0; x < k; x++) {
			const bool e = (sm[x] & my) == 0;
			const uint32_t b = __ballot_sync(0xffffffffu, e);
			const size_t pos = base + run + __popc(b & lanemask_lt);
			if (e && (pos < out_capacity)) {
				out[pos] = make_uint2(si[x], iy);
			}
			run += __popc(b);
		}
	}
}

/// launches `join_kernel` on `stream`. Does not synchronize.
/// \param B1 buckets of the first list (output of `launch_filter`)
/// \param M1 masks of the first list (output of `launch_filter`)
/// \param B2 buckets of the second list, same number of buckets
/// \param M2 masks of the second list
/// \param out output in device memory: pairs (position in L1, position in
/// 		L2), in no specific order; or nullptr to only count the pairs
/// \param out_capacity number of pairs `out` can hold
/// \param out_count output in device memory: total number of pairs, which can
/// 		be larger than `out_capacity`.
/// \param stream CUDA stream
/// \param reset if true `out_count` is set to zero first
template<const uint32_t BT=256, const uint32_t T1=2048>
void launch_join(const Buckets &B1, const uint64_t *M1,
                 const Buckets &B2, const uint64_t *M2,
                 uint2 *out, const size_t out_capacity,
                 unsigned long long *out_count,
                 cudaStream_t stream = 0, const bool reset = true) noexcept {
	assert(B1.nr_buckets == B2.nr_buckets);
	if (reset) {
		checkCudaErrors(cudaMemsetAsync(out_count, 0, sizeof(unsigned long long), stream));
	}
	if ((B1.nr_buckets == 0) || (B2.capacity == 0)) {
		return;
	}

	const dim3 grid((B2.capacity + BT - 1u) / BT, B1.nr_buckets);
	join_kernel<BT, T1><<<grid, BT, 0, stream>>>(B1, M1, B2, M2, out, out ? out_capacity : 0, out_count);
	checkCudaErrors(cudaGetLastError());
}

} // namespace cryptanalysislib::nn::full_length
