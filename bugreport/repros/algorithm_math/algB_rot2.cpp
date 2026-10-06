#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "forkB.h"
#include "algorithm/rotate.h"
using namespace cryptanalysislib;
void dump(const char*s,int*a,int n){printf("%s:",s);for(int i=0;i<n;i++)printf(" %d",a[i]);printf("\n");}
int main(){
  { int a[7]={1,2,3,9,9,9,9}; trinity_rotation<int,8>(a,1,2); dump("trinity(1,2,3; left=1) expect 2 3 1",a,7); }
  { int a[7]={1,2,3,9,9,9,9}; trinity_rotation<int,8>(a,2,1); dump("trinity(1,2,3; left=2) expect 3 1 2",a,7); }
#ifdef T_ALL
  int bad=0;
  for (const char* nm : {"grail","piston"}) { bad=0;
  for (int n=0;n<=70;n++) for(int l=0;l<=n;l++){ std::vector<int> a(n+4), b; for(int i=0;i<n+4;i++) a[i]=i+1; b=a;
     pid_t __pid=fork(); if(!__pid){ alarm(2); if(nm[0]=='g'&&nm[1]=='r'&&nm[2]=='a') grail_rotation(a.data(),l,n-l); else if(nm[0]=='p') piston_rotation(a.data(),l,n-l); else {}
        std::rotate(b.begin(),b.begin()+l,b.begin()+n); _exit(a==b?0:1);} int st; waitpid(__pid,&st,0);
     if(!(WIFEXITED(st)&&WEXITSTATUS(st)==0)){ if(bad++<3) printf("%s n=%d left=%d: %s\n",nm,n,l, WIFSIGNALED(st)?(WTERMSIG(st)==14?"HANG":"CRASH"):"mismatch"); } }
  printf("%s<int> bad=%d\n",nm,bad); }
#endif
}
