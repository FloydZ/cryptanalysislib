#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include "container/binary_indexed_tree.h"
int main(){ const size_t n=37; rangeBIT<int64_t> b(n); std::vector<int64_t> a(n+2,0); std::mt19937 r(3); int f=0;
  for(int it=0;it<5000;it++){ size_t i=1+r()%n, j=1+r()%n; if(i>j) std::swap(i,j); int64_t v=(int64_t)(r()%2001)-1000;
    if(r()%2){ b.rupdate(i,j,v); for(size_t k=i;k<=j;k++) a[k]+=v; } else { b.pupdate(i,v); a[i]+=v; }
    size_t q=r()%(n+1); int64_t s=0; for(size_t k=1;k<=q;k++) s+=a[k]; if (b.query(q)!=s){ if(f<3) printf("mismatch q=%zu %lld vs %lld\n",q,(long long)b.query(q),(long long)s); f++;} }
  BIT<int64_t> p(n); std::vector<int64_t> c(n+1,0);
  for(int it=0;it<5000;it++){ size_t i=1+r()%n; int64_t v=(int64_t)(r()%2001)-1000; p.update(i,v); c[i]+=v; size_t q=r()%(n+1); int64_t s=0; for(size_t k=1;k<=q;k++) s+=c[k]; if(p.query(q)!=s) f++; }
  printf("BIT fails=%d\n",f); }
