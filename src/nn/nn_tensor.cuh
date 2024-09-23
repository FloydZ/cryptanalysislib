#include <iostream>
#include <mma.h>
#include <chrono>
#include <cassert>

#include "helper.cuh"
#include "warp_scan.cuh"

using namespace std::chrono;
using namespace nvcuda;
using namespace nvcuda::wmma;

using T = uint32_t;

// The only dimensions currently supported by WMMA, so they are global const
constexpr int WMMA_M = 8, WMMA_N = 8, WMMA_K = 128;

// base data type, we are working with, the good old bit
using TT = experimental::precision::b1;

// Declare the fragments
typedef wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, TT, wmma::row_major> A;
typedef wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, TT, wmma::col_major> B;
typedef wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> ACC; 
typedef wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> C;


/// only correct for 128 bits
/// NOTE: for testing needs to be initialized like this:
///	for (uint32_t s = 0; s < N/32; s++) {
///		for (uint32_t i = 0; i < 8; i++) {
///			for (uint32_t j = 0; j < 8; j++) {
///				din[s*32 + i*4 + j] = s*32 + j*8 + i;
///			}
///		}
///	}
template<typename T, const uint32_t dk>
__global__ void nn_tensor128_load_matrix_synx(T *ptr, T z) {
	A a_frag; B b_frag; ACC acc_frag; C c_frag;
	wmma::fill_fragment(acc_frag, 0);
	wmma::fill_fragment(c_frag, 0);
	
	wmma::load_matrix_sync(a_frag, ((const int *)ptr), 128);
	wmma::fill_fragment(b_frag, z);
	//printf("%d %d\n", idx, a_frag.x[0]);

	// Perform the matrix multiplication
	wmma::bmma_sync(c_frag, a_frag, b_frag, acc_frag);
	printf("%d %d %d\n", idx, c_frag.x[0], c_frag.x[0]);
}

/// only correct for 128 bit
/// NOTE
template<typename T, const uint32_t dk>
__global__ void nn_tensor128(T *ptr, T z) {
	A a_frag; B b_frag; ACC acc_frag; C c_frag;
	wmma::fill_fragment(acc_frag, 0);
	wmma::fill_fragment(c_frag, 0);
	
	a_frag.x[0] = ptr[idx];	//TODO: ptr[idx * 4];	
	wmma::fill_fragment(b_frag, z);
	// printf("t:%d a:%d\n", idx, a_frag.x[0]);

	// Perform the matrix multiplication
	wmma::bmma_sync(c_frag, a_frag, b_frag, acc_frag);
	//if (idx < 4) 
	//	printf("%d %d %d\n", idx, c_frag.x[0], c_frag.x[1]);
	
	const uint32_t pred1 = c_frag.x[0] == dk;
	uint32_t pred = pred1;
	pred = warp_scan_up(pred);

	//if (idx < 8) 
	//	printf("t:%d p:%d p:%d, c:%d\n", idx, pred1, pred, c_frag.x[0]);

	const uint32_t mask = __ballot_sync(0xffffffff, pred1);
	
	if (pred1) {
		/// TODO hier halt kopieren
		printf("t %d writes at %d\n", idx, pred);
	}
}

/// NOTE: e1,e2 is the number of 4limb elements
/// NOTE: only correct if n=128
/// NOTE: the kernel only works if in expection at most only a single solutuion
/// 		can be found.
/// NOTE: nr threads cannot be bigger than e1 and e2
/// NOTE: NRThreads must be divisable by 32
/// NOTE: reads over bounds (e1 and e2) in multiple of 32
template<typename T, const uint32_t d>
__global__ void tensor_bruteForce128(T *L, T *R,
		const size_t e1, const size_t e2,
		size_t *f1, size_t *f2) {
	A a_frag; B b_frag; ACC acc_frag; C c_frag;
	const uint32_t dim = blockDim.x*gridDim.x;
	assert(dim <= e1);
	assert(dim <= e2);
	assert(blockDim.x % 32u == 0u);
	
	const uint32_t warpId = idx/32;
	const size_t stepSize = dim;
	wmma::fill_fragment(acc_frag, 0);
	wmma::fill_fragment(c_frag, 0);

	/// TODO: loop increment only correct if 32 threads are used
	for (size_t i = 0; i < e1*4; i += stepSize) {
		wmma::load_matrix_sync(a_frag, ((const int *)L + i + idx/32), 128);

		// NOTE: this loop is not split over the threads
		for (size_t j = 0; j < e2*4; j += 32) {
			wmma::load_matrix_sync(b_frag, ((const int *)R + j), 128);
			wmma::bmma_sync(c_frag, a_frag, b_frag, acc_frag);
			

			// check for each out-element corresponding to a single thread
			for(uint16_t s = 0; s < acc_frag.num_elements; s++) {
				const uint32_t pred1 = c_frag.x[s] == d;
				
				// extremly unlikely
				if (pred1) {
					const uint32_t warpId = idx%32;
					// position of the collision within the warp
					const uint32_t w_l = warpId / 4;
					const uint32_t w_r = (warpId % 4) * 2 + s;
					
					(*f1) = i/4 + idx/32 + w_l;
					(*f2) = j/4 + idx/32 + w_r;
 
					//printf("t %d found at: %d,%d\n", idx, w_r, j);
					//const uint32_t mask = __ballot_sync(0xffffffff, pred1);
					/// TODO hier halt kopieren
					//printf("t %d writes at %d\n", idx, pred);
				}

				__syncwarp();
			}
		}
	}

	__syncthreads();
}
