#include <cstdint>
#include <cstring>

#include "math/math.h"
#include "combination/revolving_door.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define L 8
#define LANES 16

/* 
 * Constant-time algorithm to compute the position of the first and second bits
 * set in successive values of an (n+2)-bit counter initialized at (1 << (n+1)).
 * Uses O(n^2) memory.
 *
 * This uses the algorithm described in "Efficient Generation of the Binary
 * Reflected Gray Code and Its Applications" by James R. Bitner, Gideon Ehrlich,
 * Edward M. Reingold (Communications of the ACM, Volume 19, 1976).
 *
 * ffs_reset(&ffs, n)    sets the counter to "zero" (and initializes the data structure).
 * ffs_step(&ffs)        increments the counter     (and updates the data structure).
 *
 * At all times, k1 and k2 contains the position of the first and second bits
 * set in the counter.
 *
 * just after reset, k1 == n+1 and k2 == -1 / counter value = (1 << (n+1)).
 *
 * If the counter is incremented (1 << n) times, its Hamming weight is always
 * at least two (after the first incrementation), so that the two indices (k1, k2)
 * are always well-defined.
 *
 * When the n low-order bits have Hamming weight 1, then k2 == n+1.
 *
 * with n = 4, the successive values of (k1, k2) are :
 * i = 32 + 0		5, -1
 * i = 32 + 1     	0,  5        #
 * i = 32 + 2     	1,  5
 * i = 32 + 3     	0,  1
 * i = 32 + 4     	2,  5
 * i = 32 + 5     	0,  2
 * i = 32 + 6     	1,  2
 * i = 32 + 7     	0,  1
 * i = 32 + 8     	3,  5
 * i = 32 + 9     	0,  3
 * i = 32 + 10     	1,  3
 * i = 32 + 11     	0,  1
 * i = 32 + 12     	2,  3
 * i = 32 + 13     	0,  2
 * i = 32 + 14     	1,  2
 * i = 32 + 15    	0,  1
 * i = 32 + 16    	4,  5
 *
 * This implementation not always the fastest possible solution 
 * (in particular if hardware instructions are available).
 * It is also not always faster than two successive "while" loops.
 * But it IS more elegant...
 */

struct ffs_t {
	int focus[34];
	int stack[33];
	int sp;
	int k1;
	int k2;
};


/// Initializes the FFS (Find First and Second bit) data structure
///
/// Sets up the internal state of the FFS structure for tracking bit positions
/// in an (n+2)-bit counter initialized at (1 << (n+1)).
///
/// \param context[out]: Pointer to the FFS data structure to initialize
/// \param n[in]: The size parameter (counter will be initialized to track an (n+2)-bit value)
static void ffs_reset(struct ffs_t *context, int n) {
	context->k1 = n + 1;
	context->k2 = -1;
	context->sp = 1;
	context->stack[0] = n + 1;
	for (uint32_t j = 0; j <= 32; j++)
		context->focus[j] = j;
}


/// Updates the FFS (Find First and Second bit) data structure to the next counter value
///
/// Performs a single step in the FFS algorithm, updating the first and second
/// bit positions (k1 and k2) that are set in the counter.
///
/// \param context[in,out]: Pointer to the FFS data structure to update
static inline void ffs_step(struct ffs_t *context) {
	/* update k1 using focus pointers */
	int j = context->focus[0];
	context->focus[0] = 0;
	context->focus[j] = context->focus[j + 1];
	context->focus[j + 1] = j + 1;
	context->k1 = j;

	/* update k2 using stack */
	context->sp -= j;
	context->k2 = context->stack[context->sp - 1];
	context->stack[context->sp] = j;
	context->sp += 1;
}

/// Calculates the index in a flattened upper triangular matrix
///
/// Computes the position in a one-dimensional array that corresponds to 
/// the element at position (i,j) in an upper triangular matrix, where i < j.
///
/// \param i[in]: Row index (must be less than j)
/// \param j[in]: Column index
/// \return Index in the flattened array
static inline int idxq(uint32_t i, uint32_t j) {
	return j * (j - 1) / 2 + i;
}

struct __attribute__((packed)) solution_t {
public:
	uint32_t x;
	uint32_t mask;
};

// extern struct solution_t * feslite_avx2_asm_enum(const void * Fq, void * Fl, uint64_t alpha, uint64_t beta, uint64_t gamma, struct solution_t *local_buffer);
#include "avx_16x16.h"


struct context_t {
	int n;
	int m;
	uint16_t Fq[561 * LANES] __attribute__((aligned(32)));
	uint16_t Fl[34 * LANES] __attribute__((aligned(32)));

	const uint32_t *Fq_start;
	const uint32_t *Fl_start;

	int count;
	uint32_t *buffer;
	int *size;

	/* local solution buffer */
	struct solution_t local_buffer[(1 << L)];

	/* candidates */
	uint32_t candidates[LANES][32];
	int n_candidates[LANES];
	bool overflow;

	/* counter */
	struct ffs_t ffs;
};

constexpr static const uint32_t M1_HI = 0xffff0000;
constexpr static const uint32_t M1_LO = 0x0000ffff;
constexpr static const uint32_t M2_HI = 0xff00ff00;
constexpr static const uint32_t M2_LO = 0x00ff00ff;
constexpr static const uint32_t M3_HI = 0xf0f0f0f0;
constexpr static const uint32_t M3_LO = 0x0f0f0f0f;
constexpr static const uint32_t M4_HI = 0xcccccccc;
constexpr static const uint32_t M4_LO = 0x33333333;
constexpr static const uint32_t M5_HI = 0xaaaaaaaa;
constexpr static const uint32_t M5_LO = 0x55555555;

///  this code was written by Antoine Joux for his book 
/// "algorithmic cryptanalysis" (cf. http://www.joux.biz). It
/// was slightly modified by C. Bouillaguet. Just like the original, it is licensed
/// under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License.
/// Performs a 32x32 bit matrix transposition
///
/// Transposes a 32x32 bit matrix using the algorithm described by Antoine Joux
/// in "Algorithmic Cryptanalysis". The algorithm uses bit manipulation to efficiently
/// transpose the matrix in-place.
///
/// \param M[in]: Pointer to the input matrix (32 uint32_t values)
/// \param T[out]: Pointer to the output transposed matrix (32 uint32_t values)
void feslite_transpose_32(const uint32_t *M, uint32_t *T) {
	/* to unroll manually */
	for (int l = 0; l < 16; l++) {
		T[l] = (M[l] & M1_LO) | ((M[l + 16] & M1_LO) << 16);
		T[l + 16] = ((M[l] & M1_HI) >> 16) | (M[l + 16] & M1_HI);
	}

	for (int l0 = 0; l0 < 32; l0 += 16) {
		for (int l = l0; l < l0 + 8; l++) {
			uint32_t val1 = (T[l] & M2_LO) | ((T[l + 8] & M2_LO) << 8);
			uint32_t val2 = ((T[l] & M2_HI) >> 8) | (T[l + 8] & M2_HI);
			T[l] = val1;
			T[l + 8] = val2;
		}
	}

	for (int l0 = 0; l0 < 32; l0 += 8) {
		for (int l = l0; l < l0 + 4; l++) {
			uint32_t val1 = (T[l] & M3_LO) | ((T[l + 4] & M3_LO) << 4);
			uint32_t val2 = ((T[l] & M3_HI) >> 4) | (T[l + 4] & M3_HI);
			T[l] = val1;
			T[l + 4] = val2;
		}
	}

	for (int l0 = 0; l0 < 32; l0 += 4) {
		for (int l = l0; l < l0 + 2; l++) {
			uint32_t val1 = (T[l] & M4_LO) | ((T[l + 2] & M4_LO) << 2);
			uint32_t val2 = ((T[l] & M4_HI) >> 2) | (T[l + 2] & M4_HI);
			T[l] = val1;
			T[l + 2] = val2;
		}
	}

	for (int l = 0; l < 32; l += 2) {
		uint32_t val1 = (T[l] & M5_LO) | ((T[l + 1] & M5_LO) << 1);
		uint32_t val2 = ((T[l] & M5_HI) >> 1) | (T[l + 1] & M5_HI);
		T[l] = val1;
		T[l + 1] = val2;
	}
}


/// Evaluates a multivariate quadratic system using a naive bit-sliced approach
///
/// Computes the result of evaluating a system of multivariate quadratic equations
/// at a given point. Optionally can check if the input has at most w bits set.
///
/// \param n[in]: Number of variables in the system
/// \param Fq[in]: Quadratic terms of the system
/// \param Fl[in]: Linear terms of the system
/// \param stride[in]: Stride between rows in the linear terms
/// \param x[in]: Input value to evaluate (bit vector representing variable assignments)
/// \param w[in]: Optional weight constraint (if > 0, restricts to inputs with ≤ w bits set)
/// \return Evaluation result as a 32-bit mask
uint32_t feslite_naive_evaluation(int n, const uint32_t *Fq, const uint32_t *Fl, int stride, uint32_t x, const uint32_t w = 0) {
	if ((w > 0) && ((uint32_t)__builtin_popcount(x)) > w) {
		return 0;
	}
	// first expand the values of the variables from `x`
	uint32_t v[32];
	for (int k = 0; k < n; k++) {
		v[k] = (x & 0x0001) ? 0xffffffff : 0x00000000;
		x >>= 1;
	}

	uint32_t y = Fl[0];

	for (int i = 0; i < n; i++) {
		// computes the contribution of degree-1 terms
		uint32_t v_0 = v[i];
		uint32_t l = Fl[stride * (1 + i)];// FIXME : get rid of this multiplication
		y ^= l & v_0;

		for (int j = 0; j < i; j++) {
			// computes the contribution of degree-2 terms
			uint32_t v_1 = v_0 & v[j];
			uint32_t q = Fq[idxq(j, i)];
			y ^= q & v_1;
		}
	}
	return y;
}

/// Evaluates multiple inputs against a multivariate quadratic system using bit-slicing
///
/// Efficiently checks multiple inputs against a multivariate quadratic system.
/// The function transposes the inputs and uses bit-slicing to evaluate all inputs
/// in parallel against equations [16:32].
///
/// \param n[in]: Number of variables in the system
/// \param Fq[in]: Quadratic terms of the system
/// \param Fl[in]: Linear terms of the system
/// \param stride[in]: Stride between rows in the linear terms
/// \param incount[in]: Number of partial solutions to check
/// \param inbuf[in]: Buffer containing partial solutions (must be size 32 regardless of incount)
/// \param outcount[in]: Maximum number of solutions to store in the output buffer
/// \param outbuf[out]: Buffer to store solutions that pass validation
/// \param size[out]: Pointer to store the number of valid solutions found
void feslite_generic_eval_32(int n,
                             const uint32_t *Fq,
                             const uint32_t *Fl,
                             int stride,
                             int incount,
                             const uint32_t *inbuf,
                             int outcount,
                             uint32_t *outbuf,
                             int *size) {
	/* FIXME : consider getting rid of this. This function is internal !*/
	*size = 0;
	if (incount == 0 || outcount == 0)
		return;

	uint32_t bitslice[32];
	feslite_transpose_32(inbuf, bitslice);

	// for each of the inputs, does it still pass?
	uint32_t valid = 0xffffffff;

	// why 16?
	for (uint32_t i = 16; i < 32; i++) {
		/* linear terms */
		uint32_t y = (Fl[0] & (1ul << i)) ? 0xffffffff : 0;
		for (uint32_t j = 0; j < (uint32_t) n; j++)
			y ^= bitslice[j] & ((Fl[stride * (1 + j)] & (1 << i)) ? 0xffffffff : 0);

		/* quadratic terms */
		for (uint32_t j = 1; j < (uint32_t) n; j++)
			for (uint32_t k = 0; k < j; k++)
				y ^= bitslice[j] & bitslice[k] & ((Fq[idxq(k, j)] & (1 << i)) ? 0xffffffff : 0);

		valid &= ~y;
		/* early abort? */
		// if (!valid) {
		// 	*size = 0;
		// 	return;
		// }
	}

	for (int i = 0; i < incount; i++) {
		if (valid & (1ul << i)) {
			if (__builtin_popcountll(inbuf[i]) > 10) {
				continue;
			}

			outbuf[*size] = inbuf[i];
			(*size)++;

			if ((*size) == outcount) {
				break;
			}
		}
	}
}

/// \param n
/// \param L=LANES !!!!! this is very important
/// \param Fq
/// \param Fl
/// \param Fq_ output of the form:
/// 			    	  0 					        L-1 (=LANES-1=15)
///               0   limb    15				 0             15
/// 		  [ [  b,  ...,    b],         ..., [   b, ...,    b]]         (limb=L=16)
/// 		  [ [x0x1, ..., x0x1],         ..., [x0x1, ..., x0x1]]         (limb=L=16)
/// 		  [ [x0x2, ..., x0x2],         ..., [x0x2, ..., x0x2]]
/// 		  [ [x1x2, ..., x1x2],         ..., [x1x2, ..., x1x2]]
///
/// 									   ...
///n*(n-1)/2: [ [x_n-1x_n, ..., x_n-1x_n], ..., [x_n-1x_n, ..., x_n-1x_n]] (Limb=n*(n+1)/2 -1) * L)
/// 		  [	[0, ..., 0],		       ..., [0, ..., 0] ]
///  		  [ [x0x1, ..., x0x1],         ..., [x0x1, ..., x0x1]]
///  		  [ [x1x2, ..., x1x2],         ..., [x1x2, ..., x1x2]]				// NOTE: here is x_i*x_{i-1}
/// 									   ...
///  		  [ [x_n-1xn, ..., x_n-1xn],   ..., [x_n-1x_n, ..., x_n-1x_n]]
/// 	      [ 0xDEAD, 				   ..., 0xDEAD ]
///
///		NOTE: they are really the same. The input systems need to be duplicated to be able to specify even further
/// \param Fl_
/// Sets up the multivariate quadratic system for the 16x16 FES solver
///
/// Prepares the data structures needed for the AVX2-accelerated 16x16 FES solver.
/// Copies and formats the quadratic and linear terms into the required layout.
///
/// \param n[in]: Number of variables in the system
/// \param LL[in]: Number of lanes (should be LANES = 16)
/// \param Fq[in]: Original quadratic terms of the system
/// \param Fl[in]: Original linear terms of the system
/// \param Fq_[out]: Reformatted quadratic terms buffer
/// \param Fl_[out]: Reformatted linear terms buffer
static inline void setup16(int n,
                           int LL,
                           const uint32_t *Fq, const uint32_t *Fl, uint16_t *Fq_, uint16_t *Fl_) {
	/* Setup Fq */
	int N = idxq(0, n);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < LL; j++) {
			Fq_[i * LL + j] = Fq[i] & 0x0000ffff;
		}
	}

	/* Fq[0,n+1] = 0, this is needed for the first round of the enumeration */
	int k = idxq(0, n + 1);
	for (int j = 0; j < LL; j++)
		Fq_[k * LL + j] = 0;

	/* Fq[i,n+1] = Fq[i-1, i] */
	for (int i = 1; i < n; i++) {
		int u = idxq(i, n + 1);
		int v = idxq(i - 1, i);
		for (int j = 0; j < LL; j++) {
			Fq_[u * LL + j] = Fq_[v * LL + j];
		}
	}

	/* Fq[n,n+1] = arbitrary */
	int m = idxq(n, n + 1);
	for (int j = 0; j < LL; j++)
		Fq_[m * LL + j] = 0xDead;

	/* Copy Fl */
	for (int i = 0; i < (n + 1) * LL; i++) {
		Fl_[i] = Fl[i] & 0x0000ffff;
	}

	/* fix values of the extra item so that we don't read uninitialized memory */
	for (int i = 0; i < LL; i++) {
		Fl_[(n + 1) * LL + i] = 0xCafe;
	}
}
/* batch-eval all the candidates */
/// Processes and evaluates a batch of candidate solutions
///
/// Takes the accumulated candidate solutions for a specific lane and evaluates them
/// against the multivariate quadratic system. Valid solutions are stored in the output buffer.
///
/// \param context[in,out]: Pointer to the FES context structure
/// \param lane[in]: The lane (equation system specialization) to process
static inline void FLUSH_CANDIDATES(struct context_t *context,
                                    int lane) {

	int max_solutions = context->count - context->size[lane];
	int k;
	uint32_t *outbuf = context->buffer + context->count * lane + context->size[lane];
	feslite_generic_eval_32(context->n, context->Fq_start, context->Fl_start + lane, LANES,
	                        context->n_candidates[lane], context->candidates[lane],
	                        max_solutions, outbuf, &k);
	context->size[lane] += k;
	context->n_candidates[lane] = 0;
	if (context->size[lane] == context->count)
		context->overflow = true;
}


/// Adds a new candidate solution to the buffer for later evaluation
///
/// Stores a candidate solution in the appropriate lane's buffer. If the buffer
/// becomes full (32 candidates), it automatically flushes and evaluates the batch.
///
/// \param context[in,out]: Pointer to the FES context structure
/// \param x[in]: The candidate solution to add
/// \param lane[in]: The lane (equation system specialization) to add the candidate to
static inline void NEW_CANDIDATE(struct context_t *context,
                                 uint32_t x,
                                 int lane) {
	int i = context->n_candidates[lane];
	context->candidates[lane][i] = x;
	context->n_candidates[lane] = i + 1;

	if (context->n_candidates[lane] == 32)
		FLUSH_CANDIDATES(context, lane);
}

/// Converts a binary number to its Gray code representation
///
/// Transforms a binary number into its corresponding Gray code value
/// using the standard binary-to-Gray conversion formula.
///
/// \param i[in]: Input binary value
/// \return The corresponding Gray code value
static inline uint32_t to_gray(uint32_t i) {
	return (i ^ (i >> 1));
}


/// Processes a batch of solutions from the local buffer
///
/// Takes solutions from the local buffer and distributes them to the appropriate
/// candidate buffers based on their mask values. Each solution is combined with
/// the provided offset value r using Gray code conversion.
///
/// \param context[in,out]: Pointer to the FES context structure
/// \param top[in]: Pointer to the end of the valid solutions in the buffer
/// \param r[in]: Offset value to combine with the solutions
/// \param flag[in]: If true, r is added before Gray code conversion; if false, after
/// \return True if the context has overflowed (reached maximum solutions)
static inline bool FLUSH_BUFFER(struct context_t *context,
                                struct solution_t *top,
                                const uint64_t r, const bool flag=true) noexcept {
	for (struct solution_t *bot = context->local_buffer; bot != top; bot++) {
		uint32_t x;
		if (flag) {
			x = to_gray(bot->x + r);
		} else {
			x = r+to_gray(bot->x);
		}
		uint32_t mask = bot->mask;
		do {
			int i = __builtin_ctzl(mask);
			NEW_CANDIDATE(context, x, i / 2);
			mask = mask & (mask - 1);
			mask = mask & (mask - 1);
		} while (mask);
	}
	return context->overflow;
}

// static inline REWIND(int alpha, int k1, int gamma)
// {
// 	Fl[0] ^= gemv(n+1, D[k1], to_gray(i));
// 	/* update the derivatives */
// 	for (int i = 0; i < L; i++)
// 		Fl[1 + i] ^= Fq[alpha + i];
// 	for (int i = 0; i < L - 1; i++)
// 		Fl[1 + i] ^= Fq[idxq(i, L-1)];
// 	Fl[k1 + 1] ^= Fq[gamma];
// }

///
/// \param n number of variables
/// \param m number of equationssystems = number of specializations already done
/// \param Fq quadratic part, same for all specializations
/// 			   0    limb     31
/// 			[ [x0x1, .... x0x1], [x0x2, ..., x0x2], ..., [xn-1xn] ]
/// \param Fl linear part different for each specialization:
/// 			   0   limb  31
/// 			[ [b ..... b, ], [x0, ..., x0], [x0, ..., x0], ... [x0, ..., x0], [x1, ..., x1], ...]
/// 				   				  F0             F1               F_LANES           F0
/// \param count
/// \param buffer
/// \param size
/// \return
/// Enumerates solutions to a multivariate quadratic system using AVX2 acceleration
///
/// Main entry point for solving a system of multivariate quadratic equations over GF(2)
/// using a fast implementation with AVX2 instructions. Processes 16 equation systems
/// in parallel (hence the 16x16 in the name).
///
/// \param n[in]: Number of variables in the system
/// \param m[in]: Number of equation systems (must be LANES=16)
/// \param Fq[in]: Quadratic terms of the system
/// \param Fl[in]: Linear terms of the system
/// \param count[in]: Maximum number of solutions to find per equation system
/// \param buffer[out]: Buffer to store the solutions (size must be at least count*m)
/// \param size[out]: Array to store the number of solutions found for each system
/// \return 0 on success, -1 if parameters are invalid
int feslite_avx2_enum_16x16(int n, int m, const uint32_t *Fq, const uint32_t *Fl, int count, uint32_t *buffer, int *size) {
	/* verify input parameters */
	if (count <= 0 || n < L || n > 32 || m != LANES) {
		return -1;
	}

	struct context_t context;
	context.n = n;
	context.m = m;
	context.count = count;
	context.buffer = buffer;
	context.size = size;
	for (int i = 0; i < LANES; i++) {
		context.n_candidates[i] = 0;
		context.size[i] = 0;
	}
	context.overflow = false;
	context.Fq_start = Fq;
	context.Fl_start = Fl;

	setup16(n, LANES, Fq, Fl, context.Fq, context.Fl);

	ffs_reset(&context.ffs, n - L);
	int k1 = context.ffs.k1 + L;
	int k2 = context.ffs.k2 + L;

	// int npositive = 0;
	uint64_t iterations = 1ul << (n - L);
	for (uint64_t j = 0; j < iterations; j++) {
		uint32_t alpha = idxq(0, k1);
		ffs_step(&context.ffs);
		k1 = context.ffs.k1 + L;
		k2 = context.ffs.k2 + L;
		uint32_t beta = 1 + k1;// +1 for the constant term
		uint32_t gamma = idxq(k1, k2);
		struct solution_t *top = solver(context.Fq, context.Fl, alpha, beta, gamma, context.local_buffer);
		if (FLUSH_BUFFER(&context, top, j << L)) {
			break;
		}
	}

	//if (n > 16) {
	for (int i = 0; i < LANES; i++) {
		FLUSH_CANDIDATES(&context, i);
	}
	//}

	//printf("FOUND %d positive for %ld iterations\n", context.size[0], iterations);
	return 0;
}


/// Enumerates weight-constrained solutions to a multivariate quadratic system
///
/// Variant of the feslite_avx2_enum_16x16 function that restricts the search to
/// solutions with a specific Hamming weight (w). Uses a revolving door algorithm
/// to efficiently enumerate solutions with the given weight constraint.
///
/// \param n[in]: Number of variables in the system
/// \param m[in]: Number of equation systems (must be LANES=16)
/// \param w[in]: Weight constraint (must be L+2 = 10)
/// \param Fq[in]: Quadratic terms of the system
/// \param Fl[in]: Linear terms of the system
/// \param count[in]: Maximum number of solutions to find per equation system
/// \param buffer[out]: Buffer to store the solutions (size must be at least count*m)
/// \param size[out]: Array to store the number of solutions found for each system
/// \return 0 on success, -1 if parameters are invalid
int feslite_avx2_enum_16x16_w(int n, int m, const uint32_t w, const uint32_t *Fq, const uint32_t *Fl, int count, uint32_t *buffer, int *size) {
	// TODO to fix the issue with 10 is to greate two more kernels which only enumerate 6 or 7 variables
	if (count <= 0 || n < L || n > 32 || m != LANES || w != (L+2)) {
		return -1;
	}

	struct solution_t *top;
	struct context_t context;
	context.n = n;
	context.m = m;
	context.count = count;
	context.buffer = buffer;
	context.size = size;
	for (int i = 0; i < LANES; i++) {
		context.n_candidates[i] = 0;
		context.size[i] = 0;
	}
	context.overflow = false;
	context.Fq_start = Fq;
	context.Fl_start = Fl;

	setup16(n, LANES, Fq, Fl, context.Fq, context.Fl);

	// init, simply specializes 000 -> 001 -> 011 -> 111
	// until we have w-8 many ones specialized
	uint32_t alph = idxq(0, n + 1);

	// TODO: iterativer revolving door ansatz:
	// 	- also um die beiden loops ein weiteter loop der alle w' = 8,....w durchgeht

	for (uint32_t i = 0; i < w - L; i++) {
		const uint32_t beta = L + i + 1, gamma = idxq(L + i, n + 1);
		top = solver(context.Fq, context.Fl, alph, beta, gamma, context.local_buffer);
		if (FLUSH_BUFFER(&context, top, i << L, true)) { break; }
		alph = idxq(0, L + i);
	}

	combination_revdoor c(n-L, w-8);
	uint32_t k1, k2;
	uint32_t alpha = alph;
	uint64_t ctr = ((1u << (w-L)) - 1u) << L;

	// k1 = cleared, k2 = set
	c.next(&k1, &k2); k1 += L; k2 += L;
	const uint64_t iterations = bc(n - L, w - 8);
	for (uint64_t j = 0; j < iterations; j++) {
		// TODO gamma is not correct, need to proper understand it.

		// First the clearing bit-flip
		uint32_t beta = k1+1; // +1 because of the constant part
		uint32_t gamma = idxq(k1, k1+1);
		top = solver(context.Fq, context.Fl, alpha, beta, gamma, context.local_buffer);
		if (FLUSH_BUFFER(&context, top, ctr, false)) { break; }

		if (j == 1) {for (uint32_t i = 0; i < LANES; i++) { FLUSH_CANDIDATES(&context, i); } return 0;}

		// Next the setting bit-flip
		alpha = idxq(0, beta-1),
		beta = k2+1;
		gamma =  idxq(k2, n+1);
		top = solver(context.Fq, context.Fl, alpha, beta, gamma, context.local_buffer);
		if (FLUSH_BUFFER(&context, top, ctr, true)) { break; }


		alpha = idxq(0, beta-1);
		ctr ^= 1u << k1;
		ctr ^= 1u << k2;
		c.next(&k1, &k2);
		k1 += L; k2 += L;
	}

	for (uint32_t i = 0; i < LANES; i++) {
		FLUSH_CANDIDATES(&context, i);
	}
	return 0;
}
