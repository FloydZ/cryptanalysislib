#ifndef CRYPTANALYSISLIB_MYTHREAD_H
#define CRYPTANALYSISLIB_MYTHREAD_H

// jeah currently thats unix only
#ifndef __APPLE__
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <err.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
// kernel stuff
#include <asm/unistd.h>

#include "atomic/futex.h"
#include "helper.h"

std::atomic<uint32_t> __global_tid = 0;

#define RUNNING 0
#define READY 1   /* Ready to be scheduled */
#define BLOCKED 2 /* Waiting on Join */
#define DEFUNCT 3 /* Dead */

#ifdef DEBUG
#define DEBUG_PRINTF(...)                          \
	__mythread_debug_futex_init();                 \
	futex_down(&debug_futex);                      \
	sprintf(debug_msg, __VA_ARGS__);               \
	(void) write(1, debug_msg, strlen(debug_msg)); \
	futex_up(&debug_futex);
#else
#define DEBUG_PRINTF(...) \
	do {                  \
	} while (0);
#endif


/// org code from:
/// https://github.com/jitesh1337/mythread_lib/blob/master/mythread_utilities.c#L11
/// but heavily modified to the c++ world by floyd


struct futex debug_futex;
static int debug_futex_init_done = 0;

char debug_msg[1000];


namespace cryptanalysislib {

	typedef struct mythread_attr {
		unsigned long stackSize; /* Stack size to be used by this thread. Default is SIGSTKSZ */
	} mythread_attr_t;

	/* Thread Handle exposed to the user */
	typedef struct mythread {
		pid_t tid; /* The thread-id of the thread */
	} mythread_t;

	/* The Actual Thread Control Block structure */
	typedef struct mythread_private {
		// The thread-id of the thread
		pid_t tid;

		// the state in which the corresponding thread will be.
		int state;

		void *(*start_func)(void *);             /* The func pointer to the thread function to be executed. */
		void *args;                              /* The arguments to be passed to the thread function. */
		void *returnValue;                       /* The return value that thread returns. */
		struct mythread_private *blockedForJoin; /* Thread blocking on this thread */
		struct futex sched_futex;                /* Futex used by the dispatcher to schedule this thread */
		struct mythread_private *prev, *next;
	} mythread_private_t;

	inline pid_t __mythread_gettid() noexcept {
		return (pid_t) syscall(SYS_gettid);
	}

	void __mythread_debug_futex_init() {
		if (debug_futex_init_done != 1) {
			futex_init(&debug_futex, 1);
			debug_futex_init_done = 0;
		}
	}
#define CLONE_SIGNAL

	/* The global extern pointer defined in mythread.h which points to the head node in
    	Queue of the Thread Control Blocks.
	*/
	mythread_private_t *mythread_q_head;

	/* The global pointer which points to the tcb of the main thread.
 	*/
	mythread_private_t *main_tcb;

	/* This structure is used to be able to refer to the Idle thread tcb.
 	*/
	mythread_t idle_u_tcb;

	/* Global futex. Please see the mythread_yield() function for more info */
	struct futex gfutex;

	/* This function initializes the Queue with a single node.
	*/
	void mythread_q_init(mythread_private_t *node) {
		node->prev = node;
		node->next = node;
		mythread_q_head = node;
	}

	/* This function adds a node to the Queue, at the end of the Queue.
		This is equivalent to Enque operation.
 	*/
	void mythread_q_add(mythread_private_t *node) {
		if (mythread_q_head == nullptr) {
			//Q is not initiazed yet. Create it.
			mythread_q_init(node);
			return;
		 }

		//Insert the node at the end of Q
		node->next = mythread_q_head;
		node->prev = mythread_q_head->prev;
		mythread_q_head->prev->next = node;
		mythread_q_head->prev = node;
	}

	/* This function deleted a specified(passed as a parameter) node from the Queue.
 	*/
	void mythread_q_delete(mythread_private_t *node) {
		mythread_private_t *p;
		if (node == mythread_q_head && node->next == mythread_q_head) {
			//There is only a single node and it is being deleted
			DEBUG_PRINTF("The Q is now Empty!\n");
			mythread_q_head = nullptr;
		}

		if (node == mythread_q_head)
			mythread_q_head = node->next;

		p = node->prev;

		p->next = node->next;
		node->next->prev = p;
	}

	/* This function iterates over the ntire Queue and prints out the state(see mythread.h to refer to various states)
   		of all the tcb members.
	*/
	void mythread_q_state_display() {
		if (mythread_q_head != nullptr) {
			//display the Q - for debug purposes
			printf("\n The Q contents are -> \n");
			mythread_private_t *p;
			p = mythread_q_head;
			do {//traverse to the last node in Q
				printf(" %d\n", p->state);
				p = p->next;
			} while (p != mythread_q_head);
		}
	}

	/// This function iterates over the Queue and prints out
	/// the state of the specified thread.
	/// \param new_tid
	/// \return
	mythread_private_t *mythread_q_search(const pid_t new_tid) noexcept {
		mythread_private_t *p;
		if (mythread_q_head != nullptr) {

			p = mythread_q_head;
			//traverse to the last node in Q
			do {
				if (p->tid == new_tid)
					return p;
				p = p->next;
			} while (p != mythread_q_head);
		}
		return nullptr;
	}

	/// This function is called from inside yield. It finds a next suitable thread
	/// which is in the READY state and wakes it up for execution.
	/// It starts looping from the next thread of the current one and hence, FIFO
	/// is ensured.
	/// \param node
	/// \return
	int __mythread_dispatcher(mythread_private_t *node) {
		mythread_private_t *ptr = node->next;
		/* Loop till we find a thread in READY state. This loop is guanrateed
	 	 * to end since idle thread is ALWAYS READY.
	 	 */
		while (ptr->state != READY) {
			ptr = ptr->next;
		}

		/* No other thread is READY. Nothing to do */
		if (ptr == node) {
			return -1;
		} else {
			DEBUG_PRINTF("Dispatcher: Wake-up:%ld Sleep:%ld %d %d\n",
			             (unsigned long) ptr->tid, (unsigned long) node->tid,
			             ptr->sched_futex.count, ptr->state);

			/* Wake up the target thread. This won't cause any races because
		 	 * it is protected by a global futex.
		 	 */
			futex_up(&ptr->sched_futex);
			DEBUG_PRINTF("Dispatcher: Woken up:%ld, to %d\n",
			             (unsigned long) ptr->tid, ptr->sched_futex.count);
			return 0;
		}
	}

	/* Return the user exposed handle to the current thread */
	inline mythread_t mythread_self() noexcept {
		mythread_t self_tcb;

		/* Get our tid */
		const pid_t tid = __mythread_gettid();
		self_tcb.tid = tid;
		return self_tcb;
	}

	// TODO remove the linked list and use the fs register to store information

///#define THREAD_SELF (*(struct mythread_private *__seg_fs *) //offsetof (struct mythread_private, header.self))
	/// Return pointed to the private TCB structure
	inline mythread_private_t *__mythread_selfptr() noexcept {
		/* Search in the queue and return the pointer */
		return mythread_q_search(__mythread_gettid());
	}


	/// Calling the glibc's exit() exits the process.
	/// Directly call the syscall instead
	inline static void __mythread_do_exit() noexcept {
		syscall(SYS_exit, 0);
	}

	/// See whether anyone is blocking on us for a join.
	/// If yes, mark that thread as READY
	/// and kill ourselves
	/// \param return_val
	void mythread_exit(void *return_val) noexcept {
		mythread_private_t *self_ptr;

		/* Get pointer to our TCB structure */
		self_ptr = __mythread_selfptr();
		ASSERT(self_ptr);

		/* Don't remove the node from the list yet. We still have to collect the return value */
		self_ptr->state = DEFUNCT;
		self_ptr->returnValue = return_val;

		/* Change the state of any thread waiting on us. FIFO dispatcher will do the
	   	 * needfull
	 	 */
		if (self_ptr->blockedForJoin != nullptr)
			self_ptr->blockedForJoin->state = READY;

		__mythread_dispatcher(self_ptr);

		/* Suicide */
		__mythread_do_exit();
	}

	/* Yield: Yield the processor to another thread. Dispatcher selects the next
	 * appropriate thread and wakes it up. Then current thread sleeps.
	 */
	int mythread_yield() {
		mythread_private_t *self;
		int retval;

		self = __mythread_selfptr();

		/* Take the global futex to avoid races in yield. There was a condition
	 	 * when after a wake-up the target thread races ahead and entered yield
	 	 * before the first thread finised. This caused deadlocks and ugly races
	 	 * when the original thread hadn't slept yet, but was tried to woken up.
	 	 * So, protect yield by a global futex and make sure the current thread
	 	 * atleast reduces its futex value to 0, before another one starts.
	 	 */
		futex_down(&gfutex);

		retval = __mythread_dispatcher(self);
		/* Only one thread. Nothing to do */
		if (retval == -1) {
			futex_up(&gfutex);
			return 0;
		}

		DEBUG_PRINTF("Yield: Might sleep on first down %ld %d\n",
		             (unsigned long) self->tid, self->sched_futex.count);
		/* The "if" condition was to fix a couple of race conditions. The purpose
	 	 * of two futex down's is to make the process sleep on the second. But
	 	 * sometimes the value of the futex is already 0, so do a conditional
	 	 * down. This, alongwith the global futex, seems to alleviate maximum
	 	 * races in yield.
	 	 */
		if (self->sched_futex.count > 0) {
			futex_down(&self->sched_futex);
		}

		futex_up(&gfutex);

		DEBUG_PRINTF("Yield: Might sleep on second down %ld %d\n",
		             (unsigned long) self->tid, self->sched_futex.count);
		/* Sleep till another process wakes us up */
		futex_down(&self->sched_futex);

		return 0;
	}

	/* Idle thread implementation.
 	 * The thread checks whether it is the only one alive, if yes, exit()
 	 * else keep scheduling someone.
 	 */
	void *mythread_idle(void *phony) {
		(void)phony;
		mythread_private_t *traverse_tcb;
		pid_t idle_tcb_tid;

		while (true) {
			DEBUG_PRINTF("I am idle\n");
			traverse_tcb = __mythread_selfptr();
			idle_tcb_tid = traverse_tcb->tid;
			traverse_tcb = traverse_tcb->next;

			/* See whether there is a NON-DEFUNCT process in the list.
		  	 * If there is, idle doesn't need to kill the process just yet */
			while (traverse_tcb->tid != idle_tcb_tid) {
				if (traverse_tcb->state != DEFUNCT) {
					break;
				}
				traverse_tcb = traverse_tcb->next;
			}

			/* Idle is the only one alive, kill the process */
			if (traverse_tcb->tid == idle_tcb_tid)
				exit(0);

			/* Some thread still awaits execution, yield ourselves */
			mythread_yield();
		}
	}

	/* A new thread is created with this wrapper pointer. The aim is to suspend the new
	 * thread until it is scheduled by the dispatcher.
	 */
	int mythread_wrapper(void *thread_tcb) noexcept {
		mythread_private_t *new_tcb;
		new_tcb = (mythread_private_t *) thread_tcb;

		DEBUG_PRINTF("Wrapper: will sleep on futex: %ld %d\n",
		             (unsigned long) __mythread_gettid(),
		             new_tcb->sched_futex.count);

		/* Suspend till explicitly woken up */
		futex_down(&new_tcb->sched_futex);

		DEBUG_PRINTF("Wrapper: futex value: %ld %d\n",
		             (unsigned long) new_tcb->tid, new_tcb->sched_futex.count);

		// We have been woken up. Now, call the user-defined function
		new_tcb->start_func(new_tcb->args);
		return 0;
	}

	/* When the first mythread_create call is invoked, we create the tcb corresponding
   		to main and idle threads. The following function adds the tcb for main thread
   		in front of the queue.
	*/
	static int __mythread_add_main_tcb() {
		DEBUG_PRINTF("add_main_tcb: Creating node for Main thread \n");
		main_tcb = (mythread_private_t *) malloc(sizeof(mythread_private_t));
		if (main_tcb == nullptr) {
			DEBUG_PRINTF("_main_tcb: Error allocating memory for main node\n");
			return -ENOMEM;
		}

		main_tcb->start_func = nullptr;
		main_tcb->args = nullptr;
		main_tcb->state = READY;
		main_tcb->returnValue = nullptr;
		main_tcb->blockedForJoin = nullptr;

		/* Get the main's tid and put it in its corresponding tcb. */
		main_tcb->tid = __mythread_gettid();

		/* Initialize futex to zero */
		futex_init(&main_tcb->sched_futex, 1);

		/* Put it in the Queue of thread blocks */
		mythread_q_add(main_tcb);
		return 0;
	}

	/* The mythread_create() function.
	  This creates a shared process context by using the clone system call.
	  We pass the pointer to a wrapper function ( See mythread_wrapper.c ) which in turn
	  sets up the thread environment and then calls the thread function.
	  The mythread_attr_t argument can optionally specify the stack size to be used
	  the newly created thread.
	*/
	int mythread_create(mythread_t *new_thread_ID,
	                    mythread_attr_t *attr,
	                    void *(*start_func)(void *), void *arg) {

		/* pointer to the stack used by the child process to be created by clone later */
		char *child_stack;

		unsigned long stackSize;
		mythread_private_t *new_node;
		pid_t tid;
		int retval;

		/* Flags to be passed to clone system call.
  		This flags variable is picked up from pthread source code - with CLONE_PTRACE removed.
		*/
		int clone_flags = (CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID | CLONE_SYSVSEM);

		if (mythread_q_head == nullptr) {
			/* This is the very first mythread_create call. Set up the Q first with tcb nodes for main thread. */
			retval = __mythread_add_main_tcb();
			if (retval != 0)
				return retval;

			/* Initialise the global futex */
			futex_init(&gfutex, 1);

			/* Now create the node for Idle thread with a recursive call to mythread_create(). */
			DEBUG_PRINTF("create: creating node for Idle thread \n");
			mythread_create(&idle_u_tcb, nullptr, mythread_idle, nullptr);
		}

		/* This particular piece of code was added as a result of a weird bug encountered in the __futex_down().
		 * In 2.6.35 (our kernel version), all threads can access main thread's stack, but
		 * on the OS machine, this stack is somehow private to main thread only.
		 */
		new_node = (mythread_private_t *) malloc(sizeof(mythread_private_t));
		if (new_node == nullptr) {
			DEBUG_PRINTF("Cannot allocate memory for node\n");
			return -ENOMEM;
		}

		/* If Stack-size argument is not provided, use the SIGSTKSZ as the default stack size
		 * Otherwise, extract the stacksize argument.
		 */
		if (attr == nullptr)
			stackSize = SIGSTKSZ;
		else
			stackSize = attr->stackSize;

		/* posix_memalign aligns the allocated memory at a 64-bit boundry. */
		if (posix_memalign((void **) &child_stack, 8, stackSize)) {
			DEBUG_PRINTF("posix_memalign failed! \n");
			return -ENOMEM;
		}

		/* We leave space for one invocation at the base of the stack */
		child_stack = child_stack + stackSize - sizeof(sigset_t);

		/* Save the thread_fun pointer and the pointer to arguments in the TCB. */
		new_node->start_func = start_func;
		new_node->args = arg;
		/* Set the state as READY - READY in Q, waiting to be scheduled. */
		new_node->state = READY;

		new_node->returnValue = nullptr;
		new_node->blockedForJoin = nullptr;
		/* Initialize the tcb's sched_futex to zero. */
		futex_init(&new_node->sched_futex, 0);

		/* Put it in the Q of thread blocks */
		mythread_q_add(new_node);

		/* Call clone with pointer to wrapper function. TCB will be passed as arg to wrapper function. */
		if ((tid = clone(mythread_wrapper, (char *) child_stack, clone_flags,
		                 new_node)) == -1) {
			printf("clone failed! \n");
			printf("ERROR: %s \n", strerror(errno));
			return (-errno);
		}
		/* Save the tid returned by clone system call in the tcb. */
		new_thread_ID->tid = tid;
		new_node->tid = tid;

		DEBUG_PRINTF("create: Finished initialising new thread: %ld\n",
		             (unsigned long) new_thread_ID->tid);
		return 0;
	}


	/// Wait on the thread specified by "target_thread". If the thread is DEFUNCT,
	/// just collect the return status. Else, wait for the thread to die and then
	///	collect the return status
	/// \param target_thread
	/// \param status
	/// \return
	int mythread_join(mythread_t target_thread,
	                  void **status) {
		mythread_private_t *self_ptr;

		self_ptr = __mythread_selfptr();
		ASSERT(self_ptr);
		DEBUG_PRINTF("Join: Got tid: %ld\n", (unsigned long) self_ptr->tid);
		mythread_private_t *target = mythread_q_search(target_thread.tid);

		/* If the thread is already dead, no need to wait. Just collect the return
 	  	* value and exit
 	  	*/
		if (target->state == DEFUNCT) {
			*status = target->returnValue;
			return 0;
		}

		DEBUG_PRINTF("Join: Checking for blocked for join\n");
		//	If the thread is not dead and someone else is already waiting on it
		//	return an error
		if (target->blockedForJoin != nullptr) {
			return -1;
		}

		/* Add ourselves as waiting for join on this thread. Set our state as
 	 	* BLOCKED so that we won't be scheduled again.
 	 	*/
		target->blockedForJoin = self_ptr;
		DEBUG_PRINTF("Join: Setting state of %ld to %d\n",
		             (unsigned long) self_ptr->tid, BLOCKED);
		self_ptr->state = BLOCKED;

		/* Schedule another thread */
		mythread_yield();

		/* Target thread died, collect return value and return */
		*status = target->returnValue;
		return 0;
	}
}// namespace cryptanalysislib

#endif
#endif
