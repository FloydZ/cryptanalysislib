#ifndef DECODING_CPUCYCLES_H
#define DECODING_CPUCYCLES_H

#ifdef USE_ARM
long long cpucycles(void) {
	unsigned long long result;
    __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (result));
	return result;
}
#else
long long cpucycles(void) {
	unsigned long long result;
	asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
			: "=a"(result)::"%rdx");
	return result;
}
#endif
#endif//DECODING_CPUCYCLES_H
