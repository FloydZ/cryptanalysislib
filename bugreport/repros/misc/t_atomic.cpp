#include "atomic/atomic_primitives.h"
#include "simd/simd.h"
int main(){ uint32_t x=0; __atomic_impl::store(&x,5u,std::memory_order_seq_cst); __atomic_impl::notify_all(&x); return __atomic_impl::load(&x,std::memory_order_seq_cst)!=5; }
