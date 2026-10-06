#include <cstdio>
#include <cstring>
#include <cstdint>
#include "hash/hash.h"
#include "hash/komihash.h"
#define XXH_INLINE_ALL
#include "xxhash.h"
static uint8_t data[20000];
int main(){
  for (int i=0;i<20000;i++) data[i]=(uint8_t)((i*131+7)&0xff);
  size_t ns[]={0,1,9,37,64,100,300,1000,5552,5553,20000};
  printf("adler32 (val=1) / crc32:\n");
  for (size_t n: ns) printf("%zu 0x%x 0x%x\n", n, adler32(1,data,n), crc32(data,n,0));
  printf("adler Wikipedia 0x%x (exp 0x11e60398)\n", adler32(1,(const uint8_t*)"Wikipedia",9));
  printf("fnv1a64(a)=%llx exp af63dc4c8601ec8c, fnv1a32(a)=%x exp e40c292c, fnv1_64(a)=%llx exp af63bd4c8601b7be, fnv1_32(a)=%x exp 50c5d7e\n",
     (unsigned long long)fnv1a<uint64_t>((const uint8_t*)"a",1), fnv1a<uint32_t>((const uint8_t*)"a",1),
     (unsigned long long)fnv1<uint64_t>((const uint8_t*)"a",1), fnv1<uint32_t>((const uint8_t*)"a",1));
  int xf=0;
  for (size_t n=0;n<2100;n++){ 
    uint64_t r = XXH3_64bits(data,n);
    uint64_t m = constexpr_xxh3::XXH3_64bits_internal((const uint8_t*)data, n, 0, constexpr_xxh3::kSecret, sizeof(constexpr_xxh3::kSecret),
       [](const uint8_t* in, size_t len, uint64_t, const void*, size_t){ return constexpr_xxh3::hashLong_64b_internal(in,len,constexpr_xxh3::kSecret,sizeof(constexpr_xxh3::kSecret));});
    if (r!=m){ if(xf<5) printf("xxh3 mismatch n=%zu\n",n); xf++;}
    for (uint64_t seed : {1ull, 0xdeadbeefcafebabeull}) {
      uint64_t rs = XXH3_64bits_withSeed(data,n,seed);
      uint64_t ms = constexpr_xxh3::XXH3_64bits_internal((const uint8_t*)data, n, seed, constexpr_xxh3::kSecret, sizeof(constexpr_xxh3::kSecret),
       [](const uint8_t* in, size_t len, uint64_t sd, const void*, size_t){ uint8_t secret[192];
        for (size_t i = 0; i < 192; i += 16) { constexpr_xxh3::writeLE64(secret + i, constexpr_xxh3::readLE64(constexpr_xxh3::kSecret + i) + sd);
          constexpr_xxh3::writeLE64(secret + i + 8, constexpr_xxh3::readLE64(constexpr_xxh3::kSecret + i + 8) - sd);}
        return constexpr_xxh3::hashLong_64b_internal(in,len,secret,192);});
      if (rs!=ms){ if(xf<10) printf("xxh3 seeded mismatch n=%zu seed=%llx\n",n,(unsigned long long)seed); xf++;}
    }
  }
  printf("xxh3 fails=%d\n",xf);
  constexpr uint64_t c = constexpr_xxh3::XXH3_64bits_const("hello");
  printf("xxh3 const hello %llx vs %llx\n",(unsigned long long)c,(unsigned long long)XXH3_64bits("hello",5));
  const char* ks[]={"This is a 32-byte testing string","The cat is out of the bag","A 16-byte string","The new string","7 chars"};
  for (auto k: ks) printf("komihash(%s)=%016llx\n",k,(unsigned long long)komihash(k,strlen(k),0));
  uint8_t bulk[256]; for(int i=0;i<256;i++) bulk[i]=i;
  for (int n: {3,6,8,12,20,31,32,40,47,48,56,64,72,80,112,132,256}) printf("bulk(%d)=%016llx\n",n,(unsigned long long)komihash(bulk,n,0));
}
