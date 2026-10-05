#include <stdio.h>
#include <stdint.h>

#define WARP_SIZE 32

//
//
//

__device__ __always_inline
uint32_t
warp_scan_inc_up_add_shuffled(uint32_t w, uint32_t const width=WARP_SIZE)
{
	uint32_t v = w;
  asm("{                                   \n\t"
      "  .reg .u32  t;                     \n\t"
      "  .reg .pred p;                     \n\t");

#pragma unroll
  for (uint32_t d = 1; d < width; d *= 2)
    asm("shfl.up.b32.sync t|p, %0, %1, 0x0, 0xffffffff;     \n\t"
        "@p add.u32 %0, %0, t;             \n\t" : "+r"(v) : "r"(d));

  asm("}");

  return v;
}

__device__
uint32_t
warp_scan_exc_up_add_shuffled(uint32_t v, uint32_t const width=WARP_SIZE)
{
#if 1

  asm("{                                   \n\t"
      "  .reg .u32  t;                     \n\t"
      "  .reg .pred p;                     \n\t"
      "                                    \n\t"
      "  shfl.sync.up.b32 t|p, %0, 1, 0x0, 0xFFFFFFFF;      \n\t"
      "  selp.b32 %0, t, 0, p;             \n\t" : "+r"(v));

#pragma unroll
  for (uint32_t d = 1; d < width; d *= 2)
    asm("shfl.sync.up.b32 t|p, %0, %1, 0x0, 0xFFFFFFFF;     \n\t"
        "@p add.u32 %0, %0, t;             \n\t" : "+r"(v) : "r"(d));

  asm("}");

  return v;

#else
  return warp_scan_inc_up_add_shuffled(v,width) - v;
#endif
}

//
//
//

extern "C"
__global__
void
warp_scan_exc_up_add_shuffled_kernel(uint32_t const * const vin, uint32_t * const vout)
{
  vout[threadIdx.x] = warp_scan_exc_up_add_shuffled(vin[threadIdx.x]);
}

extern "C"
__global__
void
warp_scan_inc_up_add_shuffled_kernel(uint32_t const * const vin, uint32_t * const vout)
{
	int a = vin[threadIdx.x];
  	int b = warp_scan_inc_up_add_shuffled(a);
  	printf("%02d %d\n", threadIdx.x, b);
	  //vout[threadIdx.x] 
}

//
//
//

void
print_scan(char const * const msg, uint32_t const warp[WARP_SIZE])
{
  printf("%6s: ",msg);

  for (int ii=0; ii<WARP_SIZE; ii++)
    printf("%2d ",warp[ii]);

  printf("\n");
}

//
//
//

int
main(int argc, char** argv)
{
  //
  // warp_scan [device] [0=exclusive] -- otherwise defaults to inclusive
  //

  const int  device    = (argc >= 2) ? atoi(argv[1])      : 0;
  const bool inclusive = (argc == 3) ? atoi(argv[2]) != 0 : true;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props,device);

  printf("%s (%2d)\n",props.name,props.multiProcessorCount);
  printf("%s scan ...\n",inclusive ? "inclusive" : "exclusive");

  cudaSetDevice(device);

  //
  //
  //

  size_t const size = sizeof(uint32_t) * WARP_SIZE;

  uint32_t *vin_d, *vout_d;

  cudaMalloc(&vin_d, size);
  cudaMalloc(&vout_d,size);

  //
  //
  //

  //uint32_t const vin_h[32] = { 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1 };
  uint32_t const vin_h[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1 };

  cudaMemcpy(vin_d,vin_h,size,cudaMemcpyHostToDevice);

  //
  // LAUNCH KERNEL
  //

  if (inclusive)
    warp_scan_inc_up_add_shuffled_kernel<<<1,WARP_SIZE>>>(vin_d,vout_d);
  else
    warp_scan_exc_up_add_shuffled_kernel<<<1,WARP_SIZE>>>(vin_d,vout_d);

  cudaDeviceSynchronize();

  //
  //
  //

  uint32_t vout_h[32];

  cudaMemcpy(vout_h,vout_d,size,cudaMemcpyDeviceToHost);

  print_scan("warp",vin_h);
  print_scan("scan",vout_h);

  //
  //
  //

  cudaFree(vin_d);
  cudaFree(vout_d);

  cudaDeviceReset();

  return 0;
}

//
