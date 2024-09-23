#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h> 
#include <chrono>
#include <curand_kernel.h>

#include "helper.cuh"
#include "nn.h"

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

int main () {
	srand(time(0));
	//return bench();

	constexpr uint64_t N = 1;
	constexpr uint64_t lam = 1u << 6u;
	constexpr uint32_t d = 4;
	constexpr uint32_t dk = d;
	constexpr uint32_t nr_limbs = 1;

	/// this is for the bruteforce testing
	//const uint32_t threads = 1024;
	//const uint32_t blocks = lam / threads;

	/// this is for the nn testing
	const uint32_t threads 	= 64;
	const uint32_t blocks 	= N;


	size_t *f1, *f2;
	T *L, *R;
	checkCudaErrors(cudaMallocManaged(&L, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMallocManaged(&R, sizeof(T) * lam * nr_limbs));
	checkCudaErrors(cudaMallocManaged(&f1, sizeof(size_t)));
	checkCudaErrors(cudaMallocManaged(&f2, sizeof(size_t)));
	*f1 = size_t(-1); *f2 = size_t(-1);

	// TODO only correct for multiple of 4
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
		L[i] = rand();
		R[i] = rand();
		//L[i] = rand();
		//R[i] = rand();
	}

	// insert special elements
	//size_t pos1 = rand() % lam, pos2 = rand() % lam;
	size_t pos1 = 2, pos2 = 2;
	size_t last_pos = 0;
	// copy e1 into e2
	for (uint32_t i = 0; i < nr_limbs; i++) { 
		R[pos2*nr_limbs + i ] = L[pos1*nr_limbs + i];
	}
	
	pos1 = 20; pos2 = 20;
	for (uint32_t i = 0; i < nr_limbs; i++) { 
		R[pos2*nr_limbs + i ] = L[pos1*nr_limbs + i];
	}
	pos1 = 34; pos2 = 34;
	for (uint32_t i = 0; i < nr_limbs; i++) { 
		R[pos2*nr_limbs + i ] = L[pos1*nr_limbs + i];
	}
	constexpr uint32_t bitsize = sizeof(T) * 8;
	for (uint32_t i = 0; i < d; i++) { 
		const size_t new_pos = last_pos + (rand() % (bitsize * nr_limbs - last_pos));	
		const uint32_t to_limb = new_pos / bitsize;
		const uint32_t to_pos  = new_pos % bitsize;
		const T to_mask = 1u << to_pos;
		
		R[pos2*nr_limbs + to_limb] ^= to_mask;
		last_pos = new_pos;
	}

	for (uint32_t i = 0; i < nr_limbs; i++) { 
		printf("%d ", L[pos1*nr_limbs + i]);
	}
	printf("\n");
	for (uint32_t i = 0; i < nr_limbs; i++) { 
		printf("%d ", R[pos2*nr_limbs + i]);
	}
	printf("\n");

	// generate rand states for the NN algorithm
	curandStateXORWOW_t *randState;
	checkCudaErrors(cudaMalloc(&randState, sizeof(curandStateXORWOW_t) * N));



	printf("%d tpb %d blk\n", threads, blocks);
	printf("sol: %d %d\n", pos1, pos2);
	//bruteforce
	//	<T, nr_limbs, d>
	//	<<<blocks, threads>>>
	//	(L, R, e1, e2, f1, f2);
	
	// first init the rng
	//initRNG
	//	<<<blocks, 1>>>
	//	(randState);

	// next run the rng
	nn
		<T, nr_limbs, d, dk, N, nr_limbs>
		<<<blocks, threads>>>
		(L, R, e1, e2, f1, f2);
	checkCudaErrors(cudaDeviceSynchronize());
	printf("sol: %d %d\n", *f1, *f2);

	assert(*f1 == pos1);
	assert(*f2 == pos2);
	cudaFree(L);cudaFree(R);cudaFree(f1);cudaFree(f2);cudaFree(randState);
} 
