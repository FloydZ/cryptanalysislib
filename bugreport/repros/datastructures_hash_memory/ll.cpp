#include <cstdio>
#include <atomic>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include "container/linkedlist.h"
template<class F> void sub(const char* name, F f){ pid_t p=fork(); if(!p){ f(); _exit(0);} int st; waitpid(p,&st,0); printf("%s -> %s\n", name, WIFSIGNALED(st)?"CRASHED (signal)":"ok"); fflush(stdout);}
int main(){
  { static FreeList<uint32_t> fl; fl.insert(3); fl.insert(1); fl.insert(2);
    printf("[FreeList] iterate after inserting 1,2,3:"); for (auto v: fl) printf(" %u", v); printf("  (expected: 1 2 3)\n");
    fl.remove(2); printf("[FreeList] size after 3 inserts + 1 remove = %zu (expected 2), contains(2)=%d\n", fl.size(), fl.contains(2));
  }
  fflush(stdout);
  sub("[FreeList<uint32_t>] insert(0)", []{ static FreeList<uint32_t> f; f.insert(0); });
  sub("[FreeList<uint32_t>] insert(UINT32_MAX) then contains", []{ static FreeList<uint32_t> f; int r=f.insert(0xFFFFFFFFu); printf("   insert(max) returned %d (1 = 'already present'), size=%zu\n", r, f.size()); });
  sub("[FreeList<int32_t>] insert(5)", []{ static FreeList<int32_t> f; f.insert(5); });
}
