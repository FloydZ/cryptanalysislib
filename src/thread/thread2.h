#ifndef CRYPTANALYSISLIB_THREAD2_H
#define CRYPTANALYSISLIB_THREAD2_H
#include <err.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>


static int              /* Start function for cloned child */
childFunc(void *arg)
{
	struct utsname uts;

	/* Retrieve and display hostname. */
	printf("uts.nodename in child:  %s\n", (char *)arg);

	/* Keep the namespace open for a while, by sleeping.
              This allows some experimentation--for example, another
              process might join the namespace. */
	sleep(2);

	printf("child terminating\n");
	return 0;           /* Child terminates now */
}

/* Stack size for cloned child */
#define STACK_SIZE (1024 * 1024)

int thread_start() {
	char            *stack;         /* Start of stack buffer */
	char            *stackTop;      /* End of stack buffer */
	pid_t           pid;
	struct utsname  uts;

	/* Allocate memory to be used for the stack of the child. */

	stack = (char *)mmap(NULL, STACK_SIZE, PROT_READ | PROT_WRITE,
	             MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
	if (stack == MAP_FAILED) {
		err(EXIT_FAILURE, "mmap");
	}
	/* Assume stack grows downward */
	stackTop = stack + STACK_SIZE;

	// NOTE: maybe later for web server:
	// 	CLONE_FS | CLONE_FILES |
	int clone_flags = (CLONE_VM | CLONE_PARENT | SIGCHLD
	                   | CLONE_SYSVSEM);

	pid = clone(childFunc, stackTop, clone_flags, (void *) "kekw");
	if (pid == -1)
		err(EXIT_FAILURE, "clone");

	printf("clone() returned pid=%jd\n", (intmax_t) pid);
	printf("parent: %d\n", getpid());

	/* Display hostname in parent's UTS namespace. This will be
              different from hostname in child's UTS namespace. */
	if (uname(&uts) == -1) {
		err(EXIT_FAILURE, "uname");
	}

	printf("uts.nodename in parent: %s\n", uts.nodename);

	sleep(5);
	if (waitpid(pid, NULL, 0) == -1) {
		err(EXIT_FAILURE, "waitpid");
	}
	printf("child has terminated\n");

	exit(EXIT_SUCCESS);
}

#endif//CRYPTANALYSISLIB_THREAD2_H
