#include <functional>
#include <cstdio>
#include <cstdint>
#include "container/queue/spsc_fixed_queue.h"
int main(){
  spsc_fixed_queue<uint64_t> q(4);
  for (uint64_t i=0;i<4;i++) q.push(i);
  for (int i=0;i<3;i++) q.pop();
  q.push(100); q.push(101);           // contents now: 3,100,101 (wrapped)
  printf("size=%zu, iterate begin..end:", q.size());
  size_t off_begin = q.begin() - std::vector<uint64_t>::iterator{}; (void)off_begin;
  for (auto it=q.begin(); it!=q.end(); ++it) printf(" %llu", (unsigned long long)*it);
  printf("   (expected: 3 100 101)\n");
}
