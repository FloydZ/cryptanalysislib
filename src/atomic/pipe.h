#ifndef CRYPTANALYSISLIB_ATOMIC_PIPE_H
#define CRYPTANALYSISLIB_ATOMIC_PIPE_H

#include <cstdint>
#include "helper.h"
#include "atomic_primitives.h"

/*  PIPE
    Single writer, multiple reader thread safe pipe using (semi) lockless programming
    Readers can only read from the back of the pipe
    The single writer can write to the front of the pipe, and read from both
    ends (a writer can be a reader) for many of the principles used here,
    see http://msdn.microsoft.com/en-us/library/windows/desktop/ee418650(v=vs.85).aspx
    Note: using log2 sizes so we do not need to clamp (multi-operation)
    Note this is not true lockless as the use of flags as a form of lock state.
*/
/* IMPORTANT: Define this to control the maximum number of elements inside a
 * pipe as a log2 number. Should be smaller than 32 since it would otherwise
 * overflow the atomic integer type.*/
#define SCHED_PIPE_SIZE_LOG2 8
#define SCHED_PIPE_SIZE (2 << SCHED_PIPE_SIZE_LOG2)
#define SCHED_PIPE_MASK (SCHED_PIPE_SIZE - 1)
typedef int sched__check_pipe_size[(SCHED_PIPE_SIZE_LOG2 < 32) ? 1 : -1];

/* 32-Bit for compare-and-swap */
#define SCHED_PIPE_INVALID 0xFFFFFFFF
#define SCHED_PIPE_CAN_WRITE 0x00000000
#define SCHED_PIPE_CAN_READ 0x11111111

///
struct sched_task_partition {
	uint32_t start;
	uint32_t end;
};
struct sched_subset_task {
	struct sched_task *task;
	struct sched_task_partition partition;
};


/// utility function, not intended for general use.
// Should only be used very prudently
#define sched_pipe_is_empty(p) (((p)->write - (p)->read_count) == 0)

struct sched_pipe {
	struct sched_subset_task buffer[SCHED_PIPE_SIZE];
	/* read and write index allow fast access to the pipe
     but actual access is controlled by the access flags. */
	uint32_t	 	  __attribute__((aligned(4))) write;
	volatile uint32_t __attribute__((aligned(4))) read_count;
	volatile uint32_t flags[SCHED_PIPE_SIZE];
	volatile uint32_t __attribute__((aligned(4))) read;

	int32_t read_back(struct sched_subset_task *dst) noexcept {
		// return false if we are unable to read. This is thread safe for both
     	// multiple readers and the writer
		uint32_t to_use;
		uint32_t previous;
		uint32_t actual_read;
		uint32_t read_count;

		ASSERT(dst);

		// we get hold of the read index for consistency,
     	// and do first pass starting at read count */
		read_count = this->read_count;
		to_use = read_count;
		while (true) {
			uint32_t write_index = this->write;
			uint32_t num_in_pipe = write_index - read_count;
			if (!num_in_pipe)
				return 0;

			/* move back to start */
			if (to_use >= write_index)
				to_use = this->read;

			/* power of two sizes ensures we can perform AND for a modulus */
			actual_read = to_use & SCHED_PIPE_MASK;
			// multiple potential readers means we should check if the data is valid
         	// using an atomic compare exchange */
			previous = __sync_val_compare_and_swap(&this->flags[actual_read], SCHED_PIPE_INVALID, SCHED_PIPE_CAN_READ);
			if (previous == SCHED_PIPE_CAN_READ)
				break;

			/* update known read count */
			read_count = this->read_count;
			++to_use;
		}

		// we update the read index using an atomic add, ws we've only read one piece
     	// of data. This ensures consitency of the read index, and the above loop ensures
     	// readers only read from unread data. */
		FAA((volatile int32_t *) &this->read_count, 1);
		SCHED_BASE_MEMORY_BARRIER_ACQUIRE();

		/* now read data, ensuring we do so after above reads & CAS */
		*dst = this->buffer[actual_read];
		this->flags[actual_read] = SCHED_PIPE_CAN_WRITE;
		return 1;
	}

	int32_t read_front(struct sched_subset_task *dst) noexcept {
		uint32_t prev;
		uint32_t actual_read = 0;
		uint32_t write_index;
		uint32_t front_read;

		write_index = this->write;
		front_read = write_index;

		// Mutliple potential reads mean we should check if the data is valid,
     	// using an atomic compare exchange - which acts as a form of lock */
		prev = SCHED_PIPE_INVALID;
		actual_read = 0;
		while (1) {
			/* power of two ensures we can use a simple cal without modulus */
			uint32_t read_count = this->read_count;
			uint32_t num_in_this = write_index - read_count;
			if (!num_in_this || !front_read) {
				this->read = read_count;
				return 0;
			}

			--front_read;
			actual_read = front_read & SCHED_PIPE_MASK;
			prev = __sync_val_compare_and_swap(&this->flags[actual_read], SCHED_PIPE_INVALID, SCHED_PIPE_CAN_READ);
			if (prev == SCHED_PIPE_CAN_READ) break;
			else if (this->read >= front_read)
				return 0;
		}

		/* now read data, ensuring we do so after above reads & CAS */
		*dst = this->buffer[actual_read];
		this->flags[actual_read] = SCHED_PIPE_CAN_WRITE;
		SCHED_BASE_MEMORY_BARRIER_RELEASE();

		/* 32-bit aligned stores are atomic, and writer owns the write index */
		--this->write;
		return 1;
	}

	int32_t _write(const struct sched_subset_task *src) noexcept {
		uint32_t actual_write;
		uint32_t write_index;
		ASSERT(pipe);
		ASSERT(src);

		/* The writer 'owns' the write index and readers can only reduce the amout of
     	 * data in the pipe. We get hold of both values for consistentcy and to
     	 * reduce false sharing impacting more than one access */
		write_index = this->write;

		/* power of two sizes ensures we can perform AND for a modulus*/
		actual_write = write_index & SCHED_PIPE_MASK;
		/* a read may still be reading this item, as there are multiple readers */
		if (this->flags[actual_write] != SCHED_PIPE_CAN_WRITE)
			return 0; /* still being read, so have caught up with tail */

		/* as we are the only writer we can update the data without atomics whilst
     	 * the write index has not been updated. */
		this->buffer[actual_write] = *src;
		this->flags[actual_write] = SCHED_PIPE_CAN_READ;

		/* we need to ensure the above occur prior to updating the write index,
     	 * otherwise another thread might read before it's finished */
		SCHED_BASE_MEMORY_BARRIER_RELEASE();
		/* 32-bit aligned stores are atomic, and writer owns the write index */
		++write_index;
		this->write = write_index;
		return 1;
	}
};
#endif//CRYPTANALYSISLIB_PIPE_H
