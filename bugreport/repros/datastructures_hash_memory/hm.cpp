#include <cstdio>
#include <cstdint>
#include <vector>
#include "container/hashmap.h"
using K = uint32_t;
template<size_t N> struct Mod { static constexpr size_t operator()(const K x) noexcept { return x % N; } };
struct Mul { static constexpr size_t operator()(const K x) noexcept { return (x * 2u) % 10u; } };

// SimpleHashMap: clear(tid) with nrbuckets not divisible by threads
constexpr static SimpleHashMapConfig sc{4, 10, 3};
// Simple2: multithreaded=true, sequential calls
constexpr static Simple2HashMapConfig s2mt{10, 2};
constexpr static Simple2HashMapConfig s2{10, 1};
constexpr static SimpleCompressedHashMapConfig cc{16, 4};
int main(){
  {
    using HM = SimpleHashMap<K, uint64_t, sc, Mod<10>>;
    static HM hm;
    for (K i=0;i<40;i++) hm.insert(i, i);
    for (uint32_t t=0;t<3;t++) hm.clear(t);   // all threads clear their share
    printf("[simple] after clear(tid) for all tids: total load=%zu (expected 0); load(bucket 9)=%zu\n", (size_t)hm.load(), (size_t)hm.load(9u));
  }
  {
    using HM = Simple2HashMap<K, uint64_t, s2mt, Mod<10>>;
    static HM hm;
    for (K i=0;i<20;i++) hm.insert(0, i+100);
    printf("[simple2 mt] load(0) after 20 inserts into one bucket = %zu (bucket capacity %zu)\n", (size_t)hm.load(0u), HM::internal_bucketsize);
  }
  {
    using HM = Simple2HashMap<K, uint64_t, s2, Mul>;
    static HM hm;
    // key 1 -> bucket 2; key 3 -> bucket 6
    hm.insert(1, 11); hm.insert(1, 12); hm.insert(3, 33);
    size_t l=hm.load(1u), pos=0; // find(e,load) does not compile: HashFkt undeclared
    printf("[simple2] find(key=1): pos=%zu load=%zu (expected pos=0 load=2)\n", pos, l);
    printf("[simple2] total load()=%zu (expected 3)\n", (size_t)hm.load());
  }
  {
    using HM = SimpleCompressedHashMap<K, uint32_t, cc, Mod<4>>;
    static HM hm;
    hm.insert(0, 1); hm.insert(0, 2); hm.insert(0, 3);
    uint32_t *out; uint32_t nr;
    hm.decompress(&out, nr, 0);
    printf("[compressed] inserted 1,2,3 into bucket 0; decompress -> nr=%u:", nr); for (uint32_t i=0;i<nr;i++) printf(" %u", out[i]); printf("\n");
    // overflow: bucket 1 untouched, fill bucket 0
    static HM hm2;
    hm2.insert(1, 5);
    for (int i=0;i<30;i++) hm2.insert(0, 1000*(i+1));
    hm2.decompress(&out, nr, 1);
    printf("[compressed] bucket1 after overflowing bucket0: nr=%u:", nr); for (uint32_t i=0;i<nr&&i<8;i++) printf(" %u", out[i]); printf(" (expected nr=1: 5)\n");
  }
}
