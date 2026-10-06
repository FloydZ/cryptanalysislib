#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include "sort/robinhoodsort.h"
static std::mt19937_64 gen(1234);
static const size_t sizes[] = {1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,100,127,128,129,255,256,257,511,512,513,1000,4096,70000};
template<typename T> void run() {
	for (size_t n : sizes) for (int kind = 0; kind < 8; kind++) {
		std::vector<T> v(n);
		for (size_t i = 0; i < n; i++) switch(kind){
			case 0: v[i]=(T)gen(); break; case 1: v[i]=(T)(gen()%4); break; case 2: v[i]=42; break;
			case 3: v[i]=(T)i; break; case 4: v[i]=(T)(n-i); break;
			case 5: v[i]=(gen()&1)?std::numeric_limits<T>::max():std::numeric_limits<T>::min(); break;
			case 6: v[i]=(T)(gen()%1000); break; case 7: v[i]=std::numeric_limits<T>::max()-(T)(gen()%3); break; }
		auto r = v; std::sort(r.begin(), r.end());
		fprintf(stderr, "T=%zu n=%zu kind=%d\n", sizeof(T), n, kind);
		rhmergesort<T>(v.data(), n);
		if (v != r) fprintf(stderr, "MISMATCH T=%zu n=%zu kind=%d\n", sizeof(T), n, kind);
	}
}
int main(int argc, char**argv){ int w = atoi(argv[1]); if(w==1)run<uint8_t>(); if(w==2)run<uint16_t>(); if(w==4)run<uint32_t>(); if(w==8)run<uint64_t>(); if(w==5)run<int32_t>(); }
