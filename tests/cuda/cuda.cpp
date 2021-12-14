#define CUSTOM_ALIGNMENT 4096

#include "cuda/cuda.h"
#include "../binary/binary.h"

template<class Element>
__global__
void  join2lists_sorted_cuda(Element *L3, const Element *L2, const Element *L1, const uint32_t k_lower, const uint32_t k_higher) {
	uint32_t index = threadIdx.x;
}

__global__ void testKernel(int val) {
	printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

int main(int argc, char **argv) {
	int devID;
	cudaDeviceProp props;

	// This will pick the best possible CUDA capable device
	devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
	printf("printf() is called. Output:\n\n");

	constexpr uint32_t listsize = 10;
	const std::vector<uint64_t> lta{10, 20};

	BinaryElement* dL1 = nullptr, *dL2 = nullptr, *dL3 = nullptr;
	checkCudaErrors(cudaMalloc(&dL1, listsize * sizeof(BinaryElement)));
	checkCudaErrors(cudaMalloc(&dL2, listsize * sizeof(BinaryElement)));
	checkCudaErrors(cudaMalloc(&dL3, listsize * sizeof(BinaryElement)));


	//Kernel configuration, where a two-dimensional grid and
	//three-dimensional blocks are configured.
	//dim3 dimGrid(2, 2);
	//dim3 dimBlock(2, 2, 2);
	//testKernel<<<dimGrid, dimBlock>>>(10);

	mzd_t *A_ = mzd_init(G_n, G_n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(G_n);

	BinaryLabel target;
	target.random();
	BinaryList L1{0}, L2{0}, L3{0};
	L1.generate_base_random(listsize, A);
	L2.generate_base_random(listsize, A);

	checkCudaErrors(cudaMemcpy(dL1, L1.data(), listsize * sizeof(Element), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dL2, L2.data(), listsize * sizeof(Element), cudaMemcpyHostToDevice));

	BinaryTree::join2lists(L3, L2, L1, (const BinaryLabel)target, lta, true);
	//join2lists_cuda<BinaryElement><<<1, 10>>>(dL3, (const BinaryElement *)dL2, (const BinaryElement *)dL3, lta[0], lta[1]);

	cudaDeviceSynchronize();

	return EXIT_SUCCESS;
}