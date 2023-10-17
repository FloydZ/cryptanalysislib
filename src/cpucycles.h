#ifndef DECODING_CPUCYCLES_H
#define DECODING_CPUCYCLES_H

#if defined(USE_ARM)
long long cpucycles(void) {
	unsigned long long result;
    __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (result));
	return result;
}
#elif defined(USE_AVX2)
long long cpucycles(void) {
	unsigned long long result;
	asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
			: "=a"(result)::"%rdx");
	return result;
}
#else

// backup definition. If everything fails.
long long cpucycles(void) {
	return -1;
}
#endif
#endif//DECODING_CPUCYCLES_H
