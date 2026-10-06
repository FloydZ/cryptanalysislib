#include <cstdio>
#include <cstdint>
#include <random>
#include <unordered_map>
#include "container/hashmap.h"
struct BadHash { size_t operator()(uint64_t x) const { return x & 0xF0; } }; // many collisions
template<class M> int run(const char* name){
  std::mt19937_64 r(1); std::unordered_map<uint64_t,uint64_t> ref; M m; int f=0;
  for (int it=0; it<200000; it++){
    uint64_t k = r()%3000; if (it%97==0) k=0; if (it%101==0) k=~0ull;
    int op = r()%10;
    if (op<5){ auto a=m.insert({k,it}); auto b=ref.insert({k,(uint64_t)it}); if(a.second!=b.second) f++; }
    else if (op<7){ size_t a=m.erase(k), b=ref.erase(k); if(a!=b) f++; }
    else if (op<9){ auto a=m.find(k); auto b=ref.find(k); if ((a==m.end())!=(b==ref.end()) || (a!=m.end() && a->second!=b->second)) f++; }
    else if (it%5000==0){ m.clear(); ref.clear(); }
    if (m.size()!=ref.size()) { f++; }
  }
  size_t cnt=0; for (auto& kv: m){ cnt++; auto b=ref.find(kv.first); if(b==ref.end()||b->second!=kv.second) f++; }
  if (cnt!=ref.size()) f++;
  printf("%s fails=%d size=%zu/%zu\n",name,f,m.size(),ref.size()); return f;
}
int main(){
  run<hopscotch_map<uint64_t,uint64_t>>("hopscotch");
  run<hopscotch_map<uint64_t,uint64_t,BadHash>>("hopscotch badhash");
  run<hopscotch_map<uint64_t,uint64_t,std::hash<uint64_t>,std::equal_to<uint64_t>,std::allocator<std::pair<uint64_t,uint64_t>>,30,true>>("hopscotch storehash");
}
