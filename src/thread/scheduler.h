#ifndef CRYPTANALYSISLIB_SCHEDULER_H
#define CRYPTANALYSISLIB_SCHEDULER_H
/*
    sched.h - zlib - Doug Binks, Micha Mettke

ABOUT:
    This is a permissively licensed ANSI C Task Scheduler for
    creating parallel programs. Note - this is a pure ANSI C single header
    conversion of Doug Binks enkiTS library (https://github.com/dougbinks/enkiTS).

    Project Goals
    - ANSI C: Designed to be easy to embed into other languages
    - Embeddable: Designed as a single header library to be easy to embed into your code. 
    - Lightweight: Designed to be lean so you can use it anywhere easily, and understand it.
    - Fast, then scalable: Designed for consumer devices first, so performance on a low number of threads is important, followed by scalability.
    - Braided parallelism: Can issue tasks from another task as well as from the thread which created the Task System.
    - Up-front Allocation friendly: Designed for zero allocations during scheduling.

DEFINE:
    SCHED_IMPLEMENTATION
        Generates the implementation of the library into the included file.
        If not provided the library is in header only mode and can be included
        in other headers or source files without problems. But only ONE file
        should hold the implementation.

    SCHED_STATIC
        The generated implementation will stay private inside the implementation
        file and all internal symbols and functions will only be visible inside
        that file.

    ASSERT
    SCHED_USE_ASSERT
        If you define SCHED_USE_ASSERT without defining ASSERT sched.h
        will use assert.h and assert(). Otherwise it will use your assert
        method. If you do not define SCHED_USE_ASSERT no additional checks
        will be added. This is the only C standard library function used
        by sched.

    SCHED_MEMSET
        You can define this to 'memset' or your own memset replacement.
        If not, sched.h uses a naive (maybe inefficent) implementation.

    SCHED_INT32
    SCHED_UINT32
    SCHED_UINT_PTR
        If your compiler is C99 you do not need to define this.
        Otherwise, sched will try default assignments for them
        and validate them at compile time. If they are incorrect, you will
        get compile errors and will need to define them yourself.

    SCHED_SPIN_COUNT_MAX
        You can change this to set the maximum number of spins for worker
        threads to stop looking for work and go into a sleeping state.

    SCHED_PIPE_SIZE_LOG2
        You can change this to set the size of each worker thread pipe.
        The value is in power of two and needs to smaller than 32 otherwise
        the atomic integer type will overflow.


LICENSE: (zlib)
    Copyright (c) 2016 Doug Binks

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1.  The origin of this software must not be misrepresented; you must not
        claim that you wrote the original software. If you use this software
        in a product, an acknowledgment in the product documentation would be
        appreciated but is not required.
    2.  Altered source versions must be plainly marked as such, and must not be
        misrepresented as being the original software.
    3.  This notice may not be removed or altered from any source distribution.

CONTRIBUTORS:
    Doug Binks (implementation)
    Micha Mettke (single header ANSI C conversion)
*/


#ifndef SCHED_INT32
#define SCHED_INT32 int
#endif
#ifndef SCHED_UINT_PTR
#define SCHED_UINT_PTR unsigned long
#endif

#include "helper.h"
#include "atomic/semaphore.h"
#include "atomic/pipe.h"
#include "thread/thread.h"

/* ---------------------------------------------------------------
 *                          THREAD
 * ---------------------------------------------------------------*/
#include <pthread.h>
#include <time.h>
#include <unistd.h>

using namespace cryptanalysislib;

typedef pthread_t sched_thread;

static inline void sched_thread_getcpuclockid(sched_thread thread,
                                              clockid_t *cid) {
	if (pthread_getcpuclockid(thread, cid) != 0) {
		std::cout << "ERROR: sched_threadcpuclockid" << std::endl;
	}
}
static inline int32_t sched_thread_create(sched_thread *returnid,
                                          void *(*StartFunc)(void *),
                                          void *arg) {
	ASSERT(returnid);
	ASSERT(StartFunc);
	return pthread_create(returnid, nullptr, StartFunc, arg) == 0;
}

static inline int32_t sched_thread_term(sched_thread threadid) noexcept {
	pthread_cancel(threadid);
	return (pthread_join(threadid, nullptr) == 0);
}

static uint32_t sched_num_hw_threads() noexcept {
	return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
}



typedef unsigned char sched_byte;
typedef SCHED_INT32 sched_int;
typedef SCHED_UINT_PTR sched_size;
typedef SCHED_UINT_PTR sched_ptr;

// forward decl
struct scheduler;


typedef void (*sched_profiler_callback_f)(void *, uint32_t thread_id);
typedef void (*sched_run)(void *, scheduler *, sched_task_partition, uint32_t thread_num);

struct sched_task {
	// custom userdata to use in callback userdata
	void *userdata;

	// function working on the task owner structure
	sched_run exec;

	// number of elements inside the set
	uint32_t size;

	// minimum size of range when splitting a task set into partitions.
    // This should be set to a value which results in computation effort of at
    // least 10k clock cycles to minimiye task scheduler overhead.
    // NOTE: The last partition will be smaller than min_range if size is not a
    // multiple of min_range (lit.: grain size)
	uint32_t min_range;
	volatile sched_int run_count;
	uint32_t range_to_run;
};


struct sched_profiling {
public:
	using callback = std::function<void*(void *, uint32_t)>;

	// from the user provided data used in each callback
	void *userdata;

	// callback called as soon as a thread starts working
	callback thread_start;

	// callback called when as a thread is finished
	sched_profiler_callback_f thread_stop;

	// callback called if a thread begins waiting
	sched_profiler_callback_f wait_start;

	// callback called if a thread is woken up
	sched_profiler_callback_f wait_stop;

	sched_profiling(callback thread_start=callback(),
	                          sched_profiler_callback_f thread_stop=nullptr,
	                          sched_profiler_callback_f wait_start=nullptr,
	                          sched_profiler_callback_f wait_stop=nullptr) noexcept
	    : thread_start(thread_start), thread_stop(thread_stop),
	      wait_start(wait_start), wait_stop(wait_stop),
	 	  thread_start_counter(0), thread_stop_counter(0),
		  wait_start_counter(0), wait_stop_counter(0)
	{
		if (!thread_start) {
			this->thread_start = [this](void *userdata, const uint32_t p) {
				std::cout << "callback" << std::endl;
				FAA(&thread_start_counter, 1);
				return (void *)nullptr;
			};
		}

		std::invoke(this->thread_start, nullptr, 0);
		ASSERT(this->thread_start.operator bool());
	}

private:
	size_t thread_start_counter;
	size_t thread_stop_counter;
	size_t wait_start_counter;
	size_t wait_stop_counter;
};

struct sched_thread_args;
struct sched_pipe;



// number of default threads
#define SCHED_DEFAULT (-1)

/* ===============================================================
 *
 *                          IMPLEMENTATION
 *
 * ===============================================================*/
#define SCHED_INTERN static


/* Pointer to Integer type conversion for pointer alignment */
#if defined(__PTRDIFF_TYPE__) /* This case should work for GCC*/
#define SCHED_UINT_TO_PTR(x) ((void *) (__PTRDIFF_TYPE__) (x))
#define SCHED_PTR_TO_UINT(x) ((sched_size) (__PTRDIFF_TYPE__) (x))
#elif !defined(__GNUC__) /* works for compilers other than LLVM */
#define SCHED_UINT_TO_PTR(x) ((void *) &((char *) 0)[x])
#define SCHED_PTR_TO_UINT(x) ((sched_size) (((char *) x) - (char *) 0))
#elif defined(SCHED_USE_FIXED_TYPES) /* used if we have <stdint.h> */
#define SCHED_UINT_TO_PTR(x) ((void *) (uintptr_t) (x))
#define SCHED_PTR_TO_UINT(x) ((uintptr_t) (x))
#else /* generates warning but works */
#define SCHED_UINT_TO_PTR(x) ((void *) (x))
#define SCHED_PTR_TO_UINT(x) ((sched_size) (x))
#endif

/* Pointer math*/
#define SCHED_PTR_ADD(t, p, i) ((t *) ((void *) ((sched_byte *) (p) + (i))))
#define SCHED_ALIGN_PTR(x, mask) \
	(SCHED_UINT_TO_PTR((SCHED_PTR_TO_UINT((sched_byte *) (x) + (mask - 1)) & ~(mask - 1))))







/* ---------------------------------------------------------------
 *                          SCHEDULER
 * ---------------------------------------------------------------*/
/* IMPORTANT: Define this to control the maximum number of iterations for a
 * thread to check for work until it is send into a sleeping state */
#ifndef SCHED_SPIN_COUNT_MAX
#define SCHED_SPIN_COUNT_MAX 100
#endif
#ifndef SCHED_SPIN_BACKOFF_MUL
#define SCHED_SPIN_BACKOFF_MUL 10
#endif
#ifndef SCHED_MAX_NUM_INITIAL_PARTITIONS
#define SCHED_MAX_NUM_INITIAL_PARTITIONS 8
#endif

struct sched_thread_args {
	uint32_t thread_num;
	scheduler *scheduler;
};

constexpr size_t sched_pipe_align 		= std::alignment_of_v<sched_pipe>;
constexpr size_t sched_arg_align 		= std::alignment_of_v<sched_thread_args>;
constexpr size_t sched_thread_align 	= std::alignment_of_v<sched_thread>;
constexpr size_t sched_semaphore_align 	= std::alignment_of_v<semaphore>;
static __thread uint32_t gtl_thread_num = 0;

// forward
void * sched_tasking_thread_f(void *pArgs) noexcept;

inline void sched_call(sched_profiler_callback_f fn,
				void *usr,
				uint32_t threadid) noexcept {
	if (fn) { fn(usr, threadid); }
}

struct scheduler {
public: // TODO currenlty needed for the external thread handler
	/* pipe for every worker thread */
	sched_pipe *pipes;

	/* number of worker threads */
	unsigned int threads_num;

	/* data used in the os thread callback */
	sched_thread_args *args;

	/* os threads array  */
	void *threads;

	/* flag whether the scheduler is running  */
	volatile sched_int running;

	/* number of thread that are currently running */
	volatile sched_int thread_running;

	/* number of thread that are currently active */
	volatile sched_int thread_waiting;

	unsigned partitions_num;

	/* divider for the array handled by a task */
	unsigned partitions_init_num;

	/* os event to signal work */
	semaphore *new_task_semaphore;

	/* flag whether the os threads have been created */
	sched_int have_threads;

	/* profiling callbacks  */
	sched_profiling *profiling;

	/* memory size */
	sched_size memory;

	// internal mem
	void *mem;

	///
	/// \param st
	/// \param range_to_split
	/// \return
	struct sched_subset_task sched_split_task(struct sched_subset_task *st,
	                 						  uint32_t range_to_split) noexcept {
		sched_subset_task res = *st;
		uint32_t range_left = st->partition.end - st->partition.start;
		if (range_to_split > range_left) {
			range_to_split = range_left;
		}
		res.partition.end = st->partition.start + range_to_split;
		st->partition.start = res.partition.end;
		return res;
	}

	///
	inline void sched_wake_threads() noexcept {
		this->new_task_semaphore->signal(this->thread_waiting);
	}

	///
	/// \param thread_num
	/// \param st
	/// \param range_to_split
	/// \param off
	void sched_split_add_task(const uint32_t thread_num,
	                          sched_subset_task *st,
	                          const uint32_t range_to_split,
	                          const sched_int off) noexcept {
		sched_int cnt = 0;
		while (st->partition.start != st->partition.end) {
			sched_subset_task t = sched_split_task(st, range_to_split);
			++cnt;
			if (!this->pipes[gtl_thread_num]._write(&t)) {
				if (cnt > 1) {
					sched_wake_threads();
				}

				if (t.task->range_to_run < range_to_split) {
					t.partition.end = t.partition.start + t.task->range_to_run;
					st->partition.start = t.partition.end;
				}
				t.task->exec(t.task->userdata, this, t.partition, thread_num);
				--cnt;
			}
		}
		FAA(&st->task->run_count, cnt + off);
		sched_wake_threads();
	}

	///
	/// \param thread_num
	/// \param pipe_hint
	/// \return
	sched_int sched_try_running_task(const uint32_t thread_num,
	                                 uint32_t *pipe_hint) noexcept {
		/* check for tasks */
		sched_subset_task subtask;
		sched_int have_task = this->pipes[thread_num].read_front(&subtask);
		uint32_t thread_to_check = *pipe_hint;
		uint32_t check_count = 0;

		while (!have_task && check_count < this->threads_num) {
			thread_to_check = (*pipe_hint + check_count) % this->threads_num;
			if (thread_to_check != thread_num)
				have_task = this->pipes[thread_to_check].read_back(&subtask);
			++check_count;
		}

		if (have_task) {
			uint32_t part_size = subtask.partition.end - subtask.partition.start;
			/* update hint, will preserve value unless actually got task from another thread */
			*pipe_hint = thread_to_check;
			if (subtask.task->range_to_run < part_size) {
				sched_subset_task t = sched_split_task(&subtask, subtask.task->range_to_run);
				sched_split_add_task(gtl_thread_num, &subtask, subtask.task->range_to_run, 0);
				subtask.task->exec(t.task->userdata, this, t.partition, thread_num);
				FAA(&t.task->run_count, -1);
			} else {
				/* the task has already been divided up by scheduler_add, so just run */
				subtask.task->exec(subtask.task->userdata, this, subtask.partition, thread_num);
				FAA(&subtask.task->run_count, -1);
			}
		}
		return have_task;
	}

	/// \param thread_num
	void scheduler_wait_for_work(uint32_t thread_num) noexcept {
		uint32_t i = 0;
		sched_int have_tasks = 0;
		FAA(&this->thread_waiting, 1);
		for (i = 0; i < this->threads_num; ++i) {
			if (!sched_pipe_is_empty(&this->pipes[i])) {
				have_tasks = 1;
				break;
			}
		}

		if (!have_tasks) {
			//TODO sched_call(profiling.wait_start, profiling.userdata, thread_num);
			this->new_task_semaphore->wait();
			//TODO sched_call(profiling.wait_stop, profiling.userdata, thread_num);
		}
		FAA(&this->thread_waiting, -1);
	}

	///
	inline void sched_pause() noexcept { __asm__ __volatile__("pause;"); }


	///  this function clears the scheduler and calculates the needed memory to run
	///  Input:
	///  -   number of os threads to create inside the scheduler (or SCHED_DEFAULT for number of cpu cores)
	///  -   optional profiling callbacks for profiler (NULL if not wanted)
	///  Output:
	///  -   needed memory for the scheduler to run
	void init(sched_size *memory,
	          sched_int thread_count = SCHED_DEFAULT,
	          sched_profiling *prof = nullptr) noexcept {
		ASSERT(memory);

		// clear our self.
		memset(this, 0, sizeof(scheduler));
		this->threads_num = (thread_count == SCHED_DEFAULT) ? sched_num_hw_threads() : (uint32_t) thread_count;

		/// ensure we have sufficient tasks to equally fill either all threads including
	    /// main or just the threads we've launched, this is outisde the first init
	    /// as we want to be able to runtime change it
		if (this->threads_num > 1) {
			this->partitions_num = this->threads_num * (this->threads_num - 1);
			this->partitions_init_num = this->threads_num - 1;
			if (this->partitions_init_num > SCHED_MAX_NUM_INITIAL_PARTITIONS)
				this->partitions_init_num = SCHED_MAX_NUM_INITIAL_PARTITIONS;
		} else {
			this->partitions_num = 1;
			this->partitions_init_num = 1;
		}
		if (prof) {
			this->profiling = prof;
		}

		/* calculate needed memory */
		ASSERT(this->threads_num > 0);
		*memory = 0;
		*memory += sizeof(sched_pipe) * this->threads_num;
		*memory += sizeof(sched_thread_args) * this->threads_num;
		*memory += sizeof(sched_thread) * this->threads_num;
		*memory += sizeof(semaphore);
		*memory += sched_pipe_align + sched_arg_align;
		*memory += sched_thread_align + sched_semaphore_align;
		this->memory = *memory;
	}

	///  this function starts running the scheduler and creates the previously set
	///  number of threads-1, which is sufficent to fill the system by
	///  including the main thread. Start can be called multiple times - it will wait
	///  for the completion before re-initializing.
	///  Input:
	///  -   previously allocated memory to run the scheduler with
	void start(void *memory) noexcept {
		ASSERT(memory);
		if (this->have_threads) { return; }
		stop(0);

		/* setup scheduler memory */
		// sched_zero_size(memory, s->memory);
		memset(memory, 0, this->memory);
		this->pipes = (sched_pipe *) SCHED_ALIGN_PTR(memory, sched_pipe_align);
		this->threads = SCHED_ALIGN_PTR(this->pipes + this->threads_num, sched_thread_align);
		this->args = (sched_thread_args *) SCHED_ALIGN_PTR(
		        SCHED_PTR_ADD(void, this->threads, sizeof(sched_thread) * this->threads_num), sched_arg_align);
		// this->new_task_semaphore = (struct semaphore *) SCHED_ALIGN_PTR(s->args + s->threads_num, sched_semaphore_align);
		// sched_semaphore_create(this->new_task_semaphore);
		this->new_task_semaphore = new semaphore();

		/* Create one less thread than thread_num as the main thread counts as one */
		this->args[0].thread_num = 0;
		this->args[0].scheduler = this;
		this->thread_running = 1;
		this->thread_waiting = 0;
		this->running = 1;

		/* start hardware threads */
		for (uint32_t i = 1; i < this->threads_num; ++i) {
			this->args[i].thread_num = i;
			this->args[i].scheduler = this;
			sched_thread_create(&((sched_thread *) (this->threads))[i],
			                    sched_tasking_thread_f, &this->args[i]);
		}

		this->have_threads = 1;
	}

public:
	scheduler(sched_int thread_count = SCHED_DEFAULT,
	          sched_profiling *prof = nullptr) noexcept {
		size_t needed_memory;
		init(&needed_memory, thread_count, prof);
		mem = calloc(needed_memory, 1);
		start(mem);
	}

	~scheduler() {
		stop(0);
		free(mem);
	}

	///  this function adds a task into the scheduler to execute and directly returns
	///  if the pipe is not full. Otherwise the task is run directly. Should only be
	///  called from main thread or within task handler.
	///  Input:
	///  -   function to execute to process the task
	///  -   userdata to call the execution function with
	///  -   array size that will be divided over multiple threads
	///  Output:
	///  -   task handle used to wait for the task to finish or check if done. Needs
	///      to be persistent over the process of the task
	void add(struct sched_task *task,
	              	   sched_run func,
	                   void *pArg,
	                   const uint32_t size,
	                   const uint32_t min_range) noexcept {
		uint32_t range_to_split = 0;
		sched_subset_task subtask;
		ASSERT(task);
		ASSERT(func);

		task->userdata = pArg;
		task->exec = func;
		task->size = size > 0 ? size : 1;
		task->run_count = -1;
		task->min_range = min_range > 0 ? min_range : 1;
		task->range_to_run = task->size / this->partitions_num;
		if (task->range_to_run < task->min_range) {
			task->range_to_run = task->min_range;
		}

		range_to_split = task->size / this->partitions_init_num;
		if (range_to_split < task->min_range) {
			range_to_split = task->min_range;
		}

		subtask.task = task;
		subtask.partition.start = 0;
		subtask.partition.end = task->size;
		sched_split_add_task(gtl_thread_num, &subtask, range_to_split, 1);
	}

	///  this function waits for a previously started task to finish. Should only be
	///  called from thread which created the task scheduler, or within a task
	///  handler. if called with NULL it will try to run task and return if none
	///  available.
	///  Input:
	///  -   previously started task to wait until it is finished
	void join(sched_task *task) noexcept {
		uint32_t pipe_to_check = gtl_thread_num + 1;
		if (task) {
			while (task->run_count)
				sched_try_running_task(gtl_thread_num, &pipe_to_check);
		} else
			sched_try_running_task(gtl_thread_num, &pipe_to_check);
	}

	/// this function waits for all task inside the scheduler to finish. Not
	/// guaranteed to work unless we know we are in a situation where task aren't
	/// being continuosly added. */
	/// \param s
	void wait() noexcept {
		sched_int have_task = 1;
		uint32_t pipe_hint = gtl_thread_num + 1;
		while (have_task || this->thread_waiting < (this->thread_running - 1)) {
			uint32_t i = 0;
			sched_try_running_task(gtl_thread_num, &pipe_hint);
			have_task = 0;
			for (i = 0; i < this->threads_num; ++i) {
				if (!sched_pipe_is_empty(&this->pipes[i])) {
					have_task = 1;
					break;
				}
			}
		}
	}

	/// this function waits for all task inside the scheduler to finish and stops
	/// all threads and shuts the scheduler down. Not guaranteed to work unless we
	/// are in a situation where task aren't being continuosly added.
	/// Input:
	/// -   boolean flag specifing to wait for all task to finish before stopping
	void stop(int doWait) noexcept {
		uint32_t i = 0;
		if (!this->have_threads) {
			return;
		}

		/* wait for threads to quit and terminate them */
		this->running = 0;
		wait();
		while (doWait && this->thread_running > 1) {
			// keep firing event to ensure all threads pick up state of running
			this->new_task_semaphore->signal(this->thread_running);
		}
		for (i = 1; i < this->threads_num; ++i) {
			sched_thread_term(((sched_thread *) (this->threads))[i]);
		}

		this->new_task_semaphore->~semaphore();
		// sched_semaphore_close(this->new_task_semaphore);

		// TODO free
		this->new_task_semaphore = nullptr;
		this->thread_running = 0;
		this->thread_waiting = 0;
		this->have_threads = 0;
		this->threads = nullptr;
		this->pipes = nullptr;
		this->new_task_semaphore = nullptr;
		this->args = nullptr;
	}

};

///
/// \param pArgs
/// \return
void * sched_tasking_thread_f(void *pArgs) noexcept {
	uint32_t spin_count = 0, hint_pipe;
	sched_thread_args args = *(sched_thread_args *) pArgs;
	uint32_t thread_num = args.thread_num;
	scheduler *s = args.scheduler;
	gtl_thread_num = args.thread_num;

	FAA(&s->thread_running, 1);

	//sched_call(s->profiling.thread_start, s->profiling.userdata, thread_num);
	//if (s->profiling.thread_start.operator bool()) {
		std::invoke(s->profiling->thread_start, s->profiling->userdata, thread_num);
		//s->profiling.thread_start(s->profiling.userdata, thread_num);
	//}

	hint_pipe = thread_num + 1;
	while (s->running) {
		if (!s->sched_try_running_task(thread_num, &hint_pipe)) {
			++spin_count;
			if (spin_count > SCHED_SPIN_COUNT_MAX) {
				s->scheduler_wait_for_work(thread_num);
				spin_count = 0;
			} else {
				uint32_t backoff = spin_count * SCHED_SPIN_BACKOFF_MUL;
				while (backoff) {
					s->sched_pause();
					--backoff;
				}
			}
		} else
			spin_count = 0;
	}
	FAA(&s->thread_running, -1);
	//TODO sched_call(s->profiling.thread_stop, s->profiling.userdata, thread_num);
	return nullptr;
}
#endif
