#include <cstdio>
#include <deque>
#include "container/vector_queue.h"
int main(){
  ConstVectorQueue<int,4> q;
  (void)q.push(1); (void)q.push(2);
  printf("[VQ] after push(1),push(2): front=%d back=%d (expected 1,2) size=%zu\n", q.front(), q.back(), q.size());
  ConstVectorQueue<int,4> f;
  int ok=0; for (int i=1;i<=5;i++) ok += f.push(i);
  printf("[VQ cap 4] 5 pushes accepted=%d (expected 4), front=%d (expected 1), size=%zu (expected 4)\n", ok, f.front(), f.size());
  // wrap around: cap 4, push 4, pop 2, push 2 -> should hold 3,4,5,6
  ConstVectorQueue<int,4> w; std::deque<int> ref;
  for (int i=1;i<=4;i++){ (void)w.push(i); ref.push_back(i);} w.pop(); w.pop(); ref.pop_front(); ref.pop_front();
  bool a=w.push(5), b=w.push(6);
  printf("[VQ wrap] push5=%d push6=%d size=%zu (exp 4) front=%d (exp 3)\n", a, b, w.size(), w.front());
}
