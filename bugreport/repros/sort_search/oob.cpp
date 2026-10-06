#include <cstdlib>
#include <cstdio>
int main(){int*p=(int*)malloc(16*4);volatile int x=p[16];printf("read %d\n",x);}
