#include "cuda/cuda.h"
#include "cuda/sort.cu"

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "list.h"

using DecodingValue = Value_T<BinaryContainer<G_k + G_l>>;
using DecodingLabel = Label_T<BinaryContainer<G_n - G_k>>;
using DecodingMatrix = mzd_t *;
using DecodingElement = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
using DecodingList = CUDA_List_T<DecodingElement>;
using ChangeList = std::vector<std::pair<uint32_t, uint32_t>>;

int main(int argc, char **argv) {
	const size_t size = 100;
	uint32_t *A = nullptr, *B = nullptr;
	uint32_t *dA = nullptr, *dB = nullptr;

	A = (uint32_t *) malloc(size * sizeof(uint32_t));
	B = (uint32_t *) malloc(size * sizeof(uint32_t));
	checkCudaErrors(cudaMalloc(&dA, size * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dB, size * sizeof(uint32_t)));

	for (int i = 0; i < size; i++) {
		A[i] = size - i;
		B[i] = 0;
	}

	checkCudaErrors(cudaMemcpy(dA, A, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
	counting_sort_impl<uint32_t><<<1,1>>>(dB, (const uint32_t *)dA, size);
	checkCudaErrors(cudaMemcpy(B, dB, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < size; i++) {
		std::cout << B[i] << " ";
	}
	std::cout << "\\n";

	free(A);
	free(B);
	checkCudaErrors(cudaFree(dA));
	checkCudaErrors(cudaFree(dB));

	// Partial/PrefixSum example
	const int element_sizes[] = { 10, 100, 23, 45 };
	const int ds = sizeof(element_sizes)/sizeof(element_sizes[0]);
	thrust::device_vector<int> dv_es(element_sizes, element_sizes+ds);
	thrust::device_vector<int> dv_mo(ds);
	thrust::exclusive_scan(dv_es.begin(), dv_es.end(), dv_mo.begin());
	std::cout << "element_sizes:" << std::endl;
	thrust::copy_n(dv_es.begin(), ds, std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl << "memory_offsets:" << std::endl;
	thrust::copy_n(dv_mo.begin(), ds, std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl << "memory_size:" << std::endl << dv_es[ds-1] + dv_mo[ds-1] << std::endl;


	// Sort example
	thrust::device_vector<uint32_t> dA1(A, A+size);
	thrust::device_vector<uint32_t> dB1(size);
	counting_sort_impl<uint32_t><<<1,1>>>(dB1.begin(), dA1.begin(), dA1.end());
	std::cout << "Sorted:" << std::endl;
	thrust::copy_n(dB1.begin(), size, std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl << "Unsorted:" << std::endl;



	// Cuda Parallel HashMap Example
	static constexpr ConfigCUDAParallelBucketSort config(0, 5, 10, 100, 100, 1, 1, 10, 10, 8);
	CUDAParallelBucketSort<config, DecodingList, uint32_t, uint32_t, uint32_t> hm1;
	hm1.reset();
}