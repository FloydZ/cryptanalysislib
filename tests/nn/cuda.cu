#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h> 
#include <chrono>
#include <curand_kernel.h>

#include "helper.cuh"
#include "nn/nn.cuh"

using namespace std::chrono;
using T = uint32_t;

int bench() {
	constexpr uint32_t iters = 10;
	constexpr uint64_t lam = 1u<<12u;
	constexpr uint32_t d = 4;
	constexpr uint32_t nr_limbs = 4;

	const uint32_t threads = 1024;
	const uint32_t blocks = (lam + threads - 1)/ threads;

	size_t *f1, *f2;
	T *L, *R;
	checkCudaErrors(cudaMallocManaged(&L, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMallocManaged(&R, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMalloc(&f1, sizeof(size_t)));
	checkCudaErrors(cudaMalloc(&f2, sizeof(size_t)));

	for (uint32_t i = 0; i < lam * nr_limbs; i++) {
		L[i] = rand();
		R[i] = rand();
	}

	printf("starting: %d tpb %d blk\n", threads, blocks);
	checkCudaErrors(cudaDeviceSynchronize());

	auto start = steady_clock::now();
	for (uint32_t i = 0; i < iters; i++) {
		bruteforce
			<T, nr_limbs, d>
			<<<blocks, threads>>>
			(L, R, lam, lam, f1, f2);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	auto end = steady_clock::now();
	auto usecs = duration_cast<duration<float, microseconds::period>>(end-start);
	auto ips = usecs.count() / iters / 1000000.;
	printf("%fs\n", ips);

	cudaFree(L);
	cudaFree(R);
	cudaFree(f1);
	cudaFree(f2);
	return ips;
}

/// inserts one pair of weight `d` into two random lists of size `lam` and runs
/// the nn on them
/// \return 0 if a pair of weight `d` was found
template<const uint64_t lam>
int run() {
	/// number of leaves per block
	constexpr uint32_t N = 4;
	constexpr uint32_t d = 4;
	constexpr uint32_t dk = 11;
	constexpr uint32_t nr_limbs = 2;

	/// this is for the bruteforce testing
	//const uint32_t threads = 1024;
	//const uint32_t blocks = lam / threads;

	/// this is for the nn testing
	const uint32_t threads 	= 256;
	const uint32_t blocks 	= 128;


	size_t *f1, *f2;
	T *L, *R;
	checkCudaErrors(cudaMallocManaged(&L, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMallocManaged(&R, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMallocManaged(&f1, sizeof(size_t)));
	checkCudaErrors(cudaMallocManaged(&f2, sizeof(size_t)));
	*f1 = size_t(-1); *f2 = size_t(-1);

	// NOTE only correct for multiple of 4
	//for (uint32_t i = 0; i < N; i++) {
	//	L[i] = 0;
	//	R[i] = 0;
	//}
	//L[0] = 1;
	//L[1] = 1;

	// NN128
	// 	<T, dk>
	// 	<<<1, 32>>>
	// 	(din, z);
	// checkCudaErrors(cudaDeviceSynchronize());
	
	size_t e1 = lam, e2 = lam;
	for (uint32_t i = 0; i < lam*nr_limbs; i++) {
		// NOTE: rand() only returns 31 bits
		L[i] = (T(rand()) << 16u) ^ T(rand());
		R[i] = (T(rand()) << 16u) ^ T(rand());
	}

	// insert special elements
	const size_t pos1 = rand() % lam, pos2 = rand() % lam;
	// copy e1 into e2
	for (uint32_t i = 0; i < nr_limbs; i++) {
		R[pos2*nr_limbs + i ] = L[pos1*nr_limbs + i];
	}

	// and flip exactly d distinct bits
	constexpr uint32_t bitsize = sizeof(T) * 8;
	for (uint32_t i = 0; i < d;) {
		const size_t new_pos = rand() % (bitsize * nr_limbs);
		const uint32_t to_limb = new_pos / bitsize;
		const uint32_t to_pos  = new_pos % bitsize;
		const T to_mask = T(1u) << to_pos;
		if ((L[pos1*nr_limbs + to_limb] ^ R[pos2*nr_limbs + to_limb]) & to_mask) {
			continue;
		}

		R[pos2*nr_limbs + to_limb] ^= to_mask;
		i += 1;
	}

	for (uint32_t i = 0; i < nr_limbs; i++) {
		printf("%u ", L[pos1*nr_limbs + i]);
	}
	printf("\n");
	for (uint32_t i = 0; i < nr_limbs; i++) {
		printf("%u ", R[pos2*nr_limbs + i]);
	}
	printf("\n");

	// generate rand states for the NN algorithm, one per block
	curandStateXORWOW_t *randState;
	checkCudaErrors(cudaMalloc(&randState, sizeof(curandStateXORWOW_t) * blocks));



	printf("%d tpb %d blk\n", threads, blocks);
	printf("sol: %zu %zu\n", pos1, pos2);
	//bruteforce
	//	<T, nr_limbs, d>
	//	<<<blocks, threads>>>
	//	(L, R, e1, e2, f1, f2);

	// first init the rng
	initRNG
		<<<blocks, 1>>>
		(randState);
	checkCudaErrors(cudaGetLastError());

	// next run the nn
	nn
		<T, nr_limbs, d, dk, N, nr_limbs>
		<<<blocks, threads>>>
		(L, R, e1, e2, f1, f2, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	printf("found: %zu %zu\n", *f1, *f2);

	// NOTE: assert() is disabled in release builds. Any pair of weight d is
	// 	a valid solution, not only the inserted one.
	uint32_t weight = d + 1;
	if ((*f1 < e1) && (*f2 < e2)) {
		weight = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			weight += __builtin_popcount(L[*f1*nr_limbs + i] ^ R[*f2*nr_limbs + i]);
		}
	}

	cudaFree(L);cudaFree(R);cudaFree(f1);cudaFree(f2);cudaFree(randState);
	return weight != d;
}

int main () {
	srand(time(0));
	//return bench();

	// nn on shared memory
	int ret = run<1u << 14u>();
	// small lists: the (exact) bruteforce is selected
	ret |= run<1u << 8u>();
	return ret;
}
