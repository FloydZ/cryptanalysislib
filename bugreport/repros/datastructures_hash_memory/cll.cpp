#include <cstdio>
#include "container/linkedlist.h"
int main(){
  auto *l = new ConstFreeList<uint64_t>();
  l->insert(1); l->insert(2);
  l->clear();
  printf("after clear: size=%zu; now contains(1) (reads freed head)...\n", l->size()); fflush(stdout);
  int c = l->contains(1);
  printf("contains(1)=%d (expected 0)\n", c); fflush(stdout);
  delete l; // destructor calls clear() again -> double free
  printf("done\n");
}
