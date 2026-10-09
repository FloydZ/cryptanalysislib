#ifndef CRYPTANALYSISLIB_ATOMIC_PIPE_H
#define CRYPTANALYSISLIB_ATOMIC_PIPE_H

#include <cstdint>
#include "helper.h"
#include "atomic_primitives.h"

/// utility function, not intended for general use.
/// Should only be used very prudently
#define sched_pipe_is_empty(p) (((p)->write - (p)->read_count) == 0)

class SchedulerPipeConfig {
	//
};
constexpr static SchedulerPipeConfig schedulerPipeConfig;

/// Single writer, multiple reader thread safe pipe using (semi) lockless programming
/// Readers can only read from the back of the pipe
/// The single writer can write to the front of the pipe, and read from both
/// ends (a writer can be a reader) for many of the principles used here,
/// see http://msdn.microsoft.com/en-us/library/windows/desktop/ee418650(v=vs.85).aspx
/// Note: using log2 sizes so we do not need to clamp (multi-operation)
/// Note this is not true lockless as the use of flags as a form of lock state.
/// based on: https://github.com/vurtun/lib/blob/master/sched.h
/// Modified by floyd for generic usages
/// \tparam T
/// \tparam config configuration class
template<class T,
		 const SchedulerPipeConfig &config=schedulerPipeConfig>
class SimplePipe {
private:

	/// IMPORTANT: Define this to control the maximum number of elements inside a
	/// pipe as a log2 number. Should be smaller than 32 since it would otherwise
	/// overflow the atomic integer type.
	constexpr static uint64_t PIPE_SIZE_LOG2 = 8;
	constexpr static uint64_t PIPE_SIZE = 2ull << PIPE_SIZE_LOG2;
	constexpr static uint64_t PIPE_MASK = PIPE_SIZE - 1ull;
	static_assert(PIPE_SIZE_LOG2 < 32, "this will overflow the std::atomic<int> type");


	/* 32-Bit for compare-and-swap */
	constexpr static uint32_t SCHED_PIPE_INVALID	= 0xFFFFFFFF;
	constexpr static uint32_t SCHED_PIPE_CAN_WRITE	= 0x00000000;
	constexpr static uint32_t SCHED_PIPE_CAN_READ	= 0x11111111;

	T buffer[PIPE_SIZE];

	// read and write index allow fast access to the pipe
    // but actual access is controlled by the access flags.
	// NOTE: all members are zero initialised (0 == SCHED_PIPE_CAN_WRITE)
	uint32_t	 	  __attribute__((aligned(4))) __write = 0;
	volatile uint32_t __attribute__((aligned(4))) read_count = 0;
	volatile uint32_t flags[PIPE_SIZE] = {};
	volatile uint32_t __attribute__((aligned(4))) read = 0;

public:
	int32_t read_back(T &dst) noexcept {
		// return false if we are unable to read. This is thread safe for both
     	// multiple readers and the writer
		uint32_t to_use;
		uint32_t previous;
		uint32_t actual_read;
		uint32_t read_count;

		// we get hold of the read index for consistency,
     	// and do first pass starting at read count */
		read_count = ACQUIRE(&this->read_count);
		to_use = read_count;
		while (true) {
			// NOTE: `__write` is written concurrently by the writer
			uint32_t write_index = ACQUIRE(&__write);
			uint32_t num_in_pipe = write_index - read_count;
			if (!num_in_pipe)
				return 0;

			/* move back to start */
			if (to_use >= write_index)
				to_use = this->read;

			/* power of two sizes ensures we can perform AND for a modulus */
			actual_read = to_use & PIPE_MASK;
			// multiple potential readers means we should check if the data is valid
         	// using an atomic compare exchange */
			// NOTE: CASnp(ptr, expected, desired): claim the slot if it is readable.
			// (The arguments were in the order of the original `sched_atomic_cmp_swap(dst, swap, cmp)`,
			// so readers never claimed a slot and the same item was read several times.)
			previous = CASnp(&this->flags[actual_read], SCHED_PIPE_CAN_READ, SCHED_PIPE_INVALID);
			if (previous == SCHED_PIPE_CAN_READ) {
				break;
			}

			/* update known read count */
			read_count = ACQUIRE(&this->read_count);
			++to_use;
		}

		// we update the read index using an atomic add, ws we've only read one piece
     	// of data. This ensures consitency of the read index, and the above loop ensures
     	// readers only read from unread data. */
		FAA((volatile int32_t *) &this->read_count, 1);
		MEMORY_BARRIER_ACQUIRE();

		/* now read data, ensuring we do so after above reads & CAS */
		dst = this->buffer[actual_read];
		// NOTE: release, the slot must not be reused before the read is done
		RELEASE(&this->flags[actual_read], SCHED_PIPE_CAN_WRITE);
		return 1;
	}

	[[nodiscard]] bool read_front(T &dst) noexcept {
		uint32_t prev;
		uint32_t actual_read = 0;
		uint32_t write_index;
		uint32_t front_read;

		write_index = ACQUIRE(&__write);
		front_read = write_index;

		// Mutliple potential reads mean we should check if the data is valid,
     	// using an atomic compare exchange - which acts as a form of lock */
		prev = SCHED_PIPE_INVALID;
		actual_read = 0;
		while (1) {
			/* power of two ensures we can use a simple cal without modulus */
			uint32_t read_count = ACQUIRE(&this->read_count);
			uint32_t num_in_this = write_index - read_count;
			if (!num_in_this || !front_read) {
				this->read = read_count;
				return false;
			}

			--front_read;
			actual_read = front_read & PIPE_MASK;
			// NOTE: see `read_back`
			prev = CASnp(&this->flags[actual_read], SCHED_PIPE_CAN_READ, SCHED_PIPE_INVALID);
			if (prev == SCHED_PIPE_CAN_READ) {
				break;
			} else if (this->read >= front_read) {
				return false;
			}
		}

		/* now read data, ensuring we do so after above reads & CAS */
		dst = this->buffer[actual_read];
		RELEASE(&this->flags[actual_read], SCHED_PIPE_CAN_WRITE);

		/* the writer owns the write index, but readers load it concurrently */
		RELEASE(&__write, __write - 1u);
		return true;
	}

	/// \param src
	/// \return false on failure,
	///			true on success
	[[nodiscard]] bool write_front(const T &src) noexcept {
		uint32_t actual_write;
		uint32_t write_index;

		/* The writer 'owns' the write index and readers can only reduce the amout of
     	 * data in the pipe. We get hold of both values for consistentcy and to
     	 * reduce false sharing impacting more than one access */
		write_index = __write;

		/* power of two sizes ensures we can perform AND for a modulus*/
		actual_write = write_index & PIPE_MASK;

		/* a read may still be reading this item, as there are multiple readers */
		if (ACQUIRE(&this->flags[actual_write]) != SCHED_PIPE_CAN_WRITE) {
			return false; /* still being read, so have caught up with tail */
		}

		/* as we are the only writer we can update the data without atomics whilst
     	 * the write index has not been updated. */
		this->buffer[actual_write] = src;
		RELEASE(&this->flags[actual_write], SCHED_PIPE_CAN_READ);

		/* the release store ensures the above occurs prior to updating the
		 * write index, otherwise another thread might read before it's finished */
		++write_index;
		RELEASE(&__write, write_index);
		return true;
	}
};

#undef sched_pipe_is_empty
#endif//CRYPTANALYSISLIB_PIPE_H
