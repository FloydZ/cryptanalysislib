
#ifndef idx
#define idx (threadIdx.x + blockIdx.x * blockDim.x)
#endif

#ifdef DEBUG
#include <assert.h>
#define ASSERT(x) assert(x);
#else
#define ASSERT(x)
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif

#ifndef __always_inline
#define __always_inline
#endif

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


