#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include "limits.h"

void limits(void)
{
#ifdef RLIM_INFINITY
  struct rlimit r;
  r.rlim_cur = 0;
  r.rlim_max = 0;
#ifdef RLIMIT_NOFILE
  setrlimit(RLIMIT_NOFILE,&r);
#endif
#ifdef RLIMIT_NPROC
  setrlimit(RLIMIT_NPROC,&r);
#endif
#ifdef RLIMIT_CORE
  setrlimit(RLIMIT_CORE,&r);
#endif
#endif

  /* more serious jailing would be good */
}
