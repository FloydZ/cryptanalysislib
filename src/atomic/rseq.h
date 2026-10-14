#ifndef CRYPTANALYSISLIB_ATOMIC_RSEQ_H
#define CRYPTANALYSISLIB_ATOMIC_RSEQ_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "atomic/atomic_primitives.h"

/// Restartable sequences (rseq, Linux 4.18+).
/// source: https://justine.lol/rseq/
///
/// glibc (2.35+) registers a `struct rseq` for every thread, which the kernel
/// updates with the cpu number the thread is currently running on. This gives
/// a getcpu() for the cost of a load.
/// Additionally, a thread can register a critical section via `rseq_cs`: if
/// the thread is preempted or migrated while its instruction pointer is in
/// [start_ip, start_ip + post_commit_offset), the kernel continues it at
/// `abort_ip`, which restarts the sequence. If the last instruction of the
/// sequence is the only one which writes the shared memory, this allows per
/// cpu data structures without locks or atomics.
///
/// Without rseq support (other OS, non glibc, `GLIBC_TUNABLES=glibc.pthread.rseq=0`)
/// the per cpu data structures fall back to atomics/locks.
#if defined(__linux__) && defined(__GLIBC__) && ((__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 35)) && \
    (defined(__x86_64__) || defined(__aarch64__))
#define CRYPTANALYSISLIB_RSEQ 1
#include <sched.h>
#include <unistd.h>
#include <sys/rseq.h>
#elif defined(__linux__)
#include <sched.h>
#endif

/// NOTE: not `cryptanalysislib::rseq`, which would be ambiguous with glibc's
/// 	`struct rseq` after a `using namespace cryptanalysislib`.
namespace cryptanalysislib::percpu {
#ifdef CRYPTANALYSISLIB_RSEQ
	/// \return the `struct rseq` of the calling thread, `nullptr` if rseq
	/// 	is not registered
	[[nodiscard]] inline struct ::rseq *rseq_area() noexcept {
		if (__rseq_size == 0) {
			return nullptr;
		}

		auto *r = (struct ::rseq *)((char *)__builtin_thread_pointer() + __rseq_offset);
		if (int32_t(__atomic_load_n(&r->cpu_id, __ATOMIC_RELAXED)) < 0) {
			return nullptr;
		}

		return r;
	}
#endif

	/// \return true if restartable sequences can be used by the calling thread
	[[nodiscard]] inline bool rseq_available() noexcept {
#ifdef CRYPTANALYSISLIB_RSEQ
		return rseq_area() != nullptr;
#else
		return false;
#endif
	}

	/// \return the cpu the calling thread is running on (as `sched_getcpu`),
	/// 	0 if unknown
	[[nodiscard]] inline uint32_t cpu() noexcept {
#ifdef CRYPTANALYSISLIB_RSEQ
		if (const auto *r = rseq_area()) {
			return __atomic_load_n(&r->cpu_id, __ATOMIC_RELAXED);
		}
#endif
#ifdef __linux__
		const int c = sched_getcpu();
		return c < 0 ? 0u : uint32_t(c);
#else
		return 0;
#endif
	}

	/// \return the number of configured cpus, i.e. an upper bound for `cpu()`
	[[nodiscard]] inline uint32_t number_of_cpus() noexcept {
#ifdef CRYPTANALYSISLIB_RSEQ
		const long n = sysconf(_SC_NPROCESSORS_CONF);
		return n < 1 ? 1u : uint32_t(n);
#else
		return 1;
#endif
	}

/// the `struct rseq_cs` of a critical section [301, 302) with the abort
/// handler 303, which restarts at 301. The signature (`RSEQ_SIG`) must
/// precede the abort handler. Placed in a writable section, as the absolute
/// addresses need relocations in position independent executables.
#if defined(__x86_64__)
#define CRYPTANALYSISLIB_RSEQ_CS                          \
	".pushsection __rseq_cs, \"aw\"\n"                    \
	"	.balign 32\n"                                     \
	"300:	.long 0, 0\n"                                 \
	"	.quad 301f, 302f - 301f, 303f\n"                  \
	".popsection\n"
#define CRYPTANALYSISLIB_RSEQ_ABORT                       \
	".pushsection .text.unlikely, \"ax\"\n"               \
	"	.byte 0x0f, 0xb9, 0x3d\n"                         \
	"	.long 0x53053053\n"                               \
	"303:	jmp 301b\n"                                   \
	".popsection\n"
#elif defined(__aarch64__)
#define CRYPTANALYSISLIB_RSEQ_CS                          \
	".pushsection __rseq_cs, \"aw\"\n"                    \
	"	.balign 32\n"                                     \
	"300:	.long 0, 0\n"                                 \
	"	.quad 301f, 302f - 301f, 303f\n"                  \
	".popsection\n"
#define CRYPTANALYSISLIB_RSEQ_ABORT                       \
	".pushsection .text.unlikely, \"ax\"\n"               \
	"	.long 0xd428bc00\n"                               \
	"303:	b 301b\n"                                     \
	".popsection\n"
#endif

	/// A counter with one 64 byte aligned shard per cpu. `add` only writes
	/// the shard of the current cpu, so there is no contention between threads
	/// on different cpus.
	/// \tparam T 64 bit integer
	/// \tparam max_cpus number of shards, must be >= the number of cpus,
	/// 	otherwise the shards are shared (via atomics)
	template<typename T = uint64_t,
	         const uint32_t max_cpus = 1024>
	class counter {
		static_assert(std::is_integral_v<T> && (sizeof(T) == 8));

		struct alignas(64) shard {
			T value;
		};
		shard shards[max_cpus] = {};

		/// true if all cpus have their own shard
		const bool own_shards = number_of_cpus() <= max_cpus;

	public:
		/// adds `v` to the shard of the current cpu
		inline void add(const T v) noexcept {
#ifdef CRYPTANALYSISLIB_RSEQ
			struct ::rseq *r = rseq_area();
			if (r && own_shards) {
#if defined(__x86_64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	lea 300b(%%rip), %%rcx\n"
				             "	mov %%rcx, 8(%[rs])\n"           // rseq->rseq_cs = &300
				             "	mov (%[rs]), %%ecx\n"            // rseq->cpu_id_start
				             "	shl $6, %%ecx\n"
				             "	add %[v], (%[base], %%rcx)\n"    // commit
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             :
				             : [rs] "r"(r), [base] "r"(shards), [v] "r"(v)
				             : "rcx", "cc", "memory");
#elif defined(__aarch64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	adrp x4, 300b\n"
				             "	add x4, x4, :lo12:300b\n"
				             "	str x4, [%[rs], #8]\n"           // rseq->rseq_cs = &300
				             "	ldr w4, [%[rs]]\n"               // rseq->cpu_id_start
				             "	lsl w4, w4, #6\n"
				             "	ldr x5, [%[base], x4]\n"
				             "	add x5, x5, %[v]\n"
				             "	str x5, [%[base], x4]\n"         // commit
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             :
				             : [rs] "r"(r), [base] "r"(shards), [v] "r"(v)
				             : "x4", "x5", "memory");
#endif
				return;
			}
#endif
			cryptanalysislib::atomic::fetch_add(&shards[cpu() % max_cpus].value, v, std::memory_order_relaxed);
		}

		/// \return the sum of all shards. Only exact if no thread is adding.
		[[nodiscard]] T sum() const noexcept {
			T ret = 0;
			for (uint32_t i = 0; i < max_cpus; ++i) {
				ret += cryptanalysislib::atomic::load(&shards[i].value, std::memory_order_relaxed);
			}
			return ret;
		}
	};

	/// An intrusive LIFO list (e.g. a freelist) with one 64 byte aligned
	/// list per cpu. `push`/`pop` only access the list of the current cpu.
	/// `pop` returns `nullptr` if the list of the current cpu is empty, even
	/// if other lists still have elements.
	/// \tparam Node must have a pointer `Node *next` as its first member
	/// \tparam max_cpus number of lists, must be >= the number of cpus,
	/// 	otherwise the lists are shared (via a lock)
	template<typename Node,
	         const uint32_t max_cpus = 1024>
	class list {
		struct alignas(64) shard {
			Node *head;
			/// only used without rseq
			uint32_t lock;
		};
		shard lists[max_cpus] = {};

		/// true if all cpus have their own list
		const bool own_lists = number_of_cpus() <= max_cpus;

		inline void lock(shard &l) noexcept {
			uint32_t e = 0;
			while (!cryptanalysislib::atomic::compare_exchange_weak(&l.lock, e, 1u, std::memory_order_acquire, std::memory_order_relaxed)) {
				e = 0;
#ifdef __linux__
				// the lock holder may be preempted
				sched_yield();
#endif
			}
		}

		inline void unlock(shard &l) noexcept {
			cryptanalysislib::atomic::store(&l.lock, 0u, std::memory_order_release);
		}

	public:
		/// pushes `n` onto the list of the current cpu
		inline void push(Node *n) noexcept {
			static_assert(offsetof(Node, next) == 0);
#ifdef CRYPTANALYSISLIB_RSEQ
			struct ::rseq *r = rseq_area();
			if (r && own_lists) {
#if defined(__x86_64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	lea 300b(%%rip), %%rcx\n"
				             "	mov %%rcx, 8(%[rs])\n"           // rseq->rseq_cs = &300
				             "	mov (%[rs]), %%ecx\n"            // rseq->cpu_id_start
				             "	shl $6, %%ecx\n"
				             "	mov (%[base], %%rcx), %%rdx\n"   // rdx = head
				             "	mov %%rdx, (%[n])\n"             // n->next = head
				             "	mov %[n], (%[base], %%rcx)\n"    // commit: head = n
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             :
				             : [rs] "r"(r), [base] "r"(lists), [n] "r"(n)
				             : "rcx", "rdx", "cc", "memory");
#elif defined(__aarch64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	adrp x4, 300b\n"
				             "	add x4, x4, :lo12:300b\n"
				             "	str x4, [%[rs], #8]\n"           // rseq->rseq_cs = &300
				             "	ldr w4, [%[rs]]\n"               // rseq->cpu_id_start
				             "	lsl w4, w4, #6\n"
				             "	ldr x5, [%[base], x4]\n"         // x5 = head
				             "	str x5, [%[n]]\n"                // n->next = head
				             "	str %[n], [%[base], x4]\n"       // commit: head = n
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             :
				             : [rs] "r"(r), [base] "r"(lists), [n] "r"(n)
				             : "x4", "x5", "memory");
#endif
				return;
			}
#endif
			shard &l = lists[cpu() % max_cpus];
			lock(l);
			n->next = l.head;
			l.head = n;
			unlock(l);
		}

		/// \return the head of the list of the current cpu, `nullptr` if empty
		[[nodiscard]] inline Node *pop() noexcept {
			Node *n;
#ifdef CRYPTANALYSISLIB_RSEQ
			struct ::rseq *r = rseq_area();
			if (r && own_lists) {
#if defined(__x86_64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	lea 300b(%%rip), %%rcx\n"
				             "	mov %%rcx, 8(%[rs])\n"           // rseq->rseq_cs = &300
				             "	mov (%[rs]), %%ecx\n"            // rseq->cpu_id_start
				             "	shl $6, %%ecx\n"
				             "	mov (%[base], %%rcx), %[n]\n"    // n = head
				             "	test %[n], %[n]\n"
				             "	jz 302f\n"
				             "	mov (%[n]), %%rdx\n"             // rdx = n->next
				             "	mov %%rdx, (%[base], %%rcx)\n"   // commit: head = n->next
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             : [n] "=&r"(n)
				             : [rs] "r"(r), [base] "r"(lists)
				             : "rcx", "rdx", "cc", "memory");
#elif defined(__aarch64__)
				asm volatile(CRYPTANALYSISLIB_RSEQ_CS
				             "301:	adrp x4, 300b\n"
				             "	add x4, x4, :lo12:300b\n"
				             "	str x4, [%[rs], #8]\n"           // rseq->rseq_cs = &300
				             "	ldr w4, [%[rs]]\n"               // rseq->cpu_id_start
				             "	lsl w4, w4, #6\n"
				             "	ldr %[n], [%[base], x4]\n"       // n = head
				             "	cbz %[n], 302f\n"
				             "	ldr x5, [%[n]]\n"                // x5 = n->next
				             "	str x5, [%[base], x4]\n"         // commit: head = n->next
				             "302:\n"
				             CRYPTANALYSISLIB_RSEQ_ABORT
				             : [n] "=&r"(n)
				             : [rs] "r"(r), [base] "r"(lists)
				             : "x4", "x5", "memory");
#endif
				return n;
			}
#endif
			shard &l = lists[cpu() % max_cpus];
			lock(l);
			n = l.head;
			if (n) {
				l.head = n->next;
			}
			unlock(l);
			return n;
		}

		/// removes all elements of all lists. Not thread safe.
		/// \param f[in]: called for every element
		template<typename F>
		void clear(F &&f) noexcept {
			for (uint32_t i = 0; i < max_cpus; ++i) {
				while (Node *n = lists[i].head) {
					lists[i].head = n->next;
					f(n);
				}
			}
		}
	};

#undef CRYPTANALYSISLIB_RSEQ_CS
#undef CRYPTANALYSISLIB_RSEQ_ABORT
} // end namespace cryptanalysislib::percpu
#endif
