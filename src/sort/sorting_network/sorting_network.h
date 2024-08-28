#ifndef CRYPTANALYSISLIB_SORT_SORTING_NETWORK_H
#define CRYPTANALYSISLIB_SORT_SORTING_NETWORK_H

// code original from djb_sort
#ifdef USE_AVX
#define int32_MINMAX(a,b) 			\
do { 								\
  int32_t temp1; 					\
  asm( 								\
    "cmpl %1,%0\n\t" 				\
    "mov %0,%2\n\t" 				\
    "cmovg %1,%0\n\t" 				\
    "cmovg %2,%1\n\t" 				\
    : "+r"(a), "+r"(b), "=r"(temp1) \
    : 								\
    : "cc" 							\
  ); 								\
} while(0)


#define uint32_MINMAX(a,b) 			\
do { 								\
  uint32_t temp1; 					\
  asm( 								\
    "cmpl %1,%0\n\t" 				\
    "mov %0,%2\n\t" 				\
    "cmovb %1,%0\n\t" 				\
    "cmovb %2,%1\n\t" 				\
    : "+r"(a), "+r"(b), "=r"(temp1) \
    : 								\
    : "cc" 							\
  ); 								\
} while(0)
#else
#define int32_MINMAX(a,b)	\
do {                     	\
    int32_t tmp = a;     	\
    a = a < b ? a : b;   	\
	b = a > b ? tmp : b; 	\
}while(0)

#define uint32_MINMAX(a,b)	\
do {                     	\
    uint32_t tmp = a;    	\
    a = a < b ? a : b;   	\
	b = a > b ? tmp : b; 	\
}while(0)
#endif

static inline void sortingnetwork_sort_i32x8(int32_t *x){
	int32_t x0 = x[0];
	int32_t x1 = x[1];
	int32_t x2 = x[2];
	int32_t x3 = x[3];
	int32_t x4 = x[4];
	int32_t x5 = x[5];
	int32_t x6 = x[6];
	int32_t x7 = x[7];

	/* odd-even sort instead of bitonic sort */
	int32_MINMAX(x1,x0);
	int32_MINMAX(x3,x2);
	int32_MINMAX(x2,x0);
	int32_MINMAX(x3,x1);
	int32_MINMAX(x2,x1);

	int32_MINMAX(x5,x4);
	int32_MINMAX(x7,x6);
	int32_MINMAX(x6,x4);
	int32_MINMAX(x7,x5);
	int32_MINMAX(x6,x5);

	int32_MINMAX(x4,x0);
	int32_MINMAX(x6,x2);
	int32_MINMAX(x4,x2);

	int32_MINMAX(x5,x1);
	int32_MINMAX(x7,x3);
	int32_MINMAX(x5,x3);

	int32_MINMAX(x2,x1);
	int32_MINMAX(x4,x3);
	int32_MINMAX(x6,x5);

	x[0] = x0;
	x[1] = x1;
	x[2] = x2;
	x[3] = x3;
	x[4] = x4;
	x[5] = x5;
	x[6] = x6;
	x[7] = x7;
}

static inline void sortingnetwork_sort_u32x8(uint32_t *x){
	uint32_t x0 = x[0];
	uint32_t x1 = x[1];
	uint32_t x2 = x[2];
	uint32_t x3 = x[3];
	uint32_t x4 = x[4];
	uint32_t x5 = x[5];
	uint32_t x6 = x[6];
	uint32_t x7 = x[7];

	/* odd-even sort instead of bitonic sort */
	uint32_MINMAX(x1,x0);
	uint32_MINMAX(x3,x2);
	uint32_MINMAX(x2,x0);
	uint32_MINMAX(x3,x1);
	uint32_MINMAX(x2,x1);

	uint32_MINMAX(x5,x4);
	uint32_MINMAX(x7,x6);
	uint32_MINMAX(x6,x4);
	uint32_MINMAX(x7,x5);
	uint32_MINMAX(x6,x5);

	uint32_MINMAX(x4,x0);
	uint32_MINMAX(x6,x2);
	uint32_MINMAX(x4,x2);

	uint32_MINMAX(x5,x1);
	uint32_MINMAX(x7,x3);
	uint32_MINMAX(x5,x3);

	uint32_MINMAX(x2,x1);
	uint32_MINMAX(x4,x3);
	uint32_MINMAX(x6,x5);

	x[0] = x0;
	x[1] = x1;
	x[2] = x2;
	x[3] = x3;
	x[4] = x4;
	x[5] = x5;
	x[6] = x6;
	x[7] = x7;
}
#endif
