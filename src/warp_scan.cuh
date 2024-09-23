#include <cstdint>
#include "helper.cuh"

#define WARP_SIZE 32

/// TODO describe whats happening here
/// computes the exclusive sum
__device__ __always_inline
uint32_t warp_scan_up(uint32_t v) {
    asm("{                                   			\n\t"
        "  .reg .u32  t;                     			\n\t"
        "  .reg .pred p;                     			\n\t"
        "                                    			\n\t"
        "  shfl.sync.up.b32 t|p, %0, 1, 0x0, 0xFFFFFFFF;\n\t"
        "  selp.b32 %0, t, 0, p;             			\n\t" : "+r"(v));

    #pragma unroll
    for (uint32_t d = 1; d < 32; d *= 2) {
      asm("shfl.sync.up.b32 t|p, %0, %1, 0x0, 0xFFFFFFFF;     \n\t"
          "@p add.u32 %0, %0, t;             \n\t" : "+r"(v) : "r"(d));
	}

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
