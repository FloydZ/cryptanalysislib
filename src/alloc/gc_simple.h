#ifndef CRYPTANALYSISLIB_GC_SIMPLE_H
#define CRYPTANALYSISLIB_GC_SIMPLE_H

#include <memory>
#include <setjmp.h>
#include <sys/syslog.h>

#include "alloc/alloc.h"
#include "math/math.h"
#include "memory/memory.h"

/*
 * The size of a pointer.
 */
#define PTRSIZE sizeof(char *)

/*
 * Allocations can temporarily be tagged as "marked" an part of the
 * mark-and-sweep implementation or can be tagged as "roots" which are
 * not automatically garbage collected. The latter allows the implementation
 * of global variables.
 */
#define GC_TAG_NONE 0x0
#define GC_TAG_ROOT 0x1
#define GC_TAG_MARK 0x2

/// \param this ptr
/// \return simply returns the higher 61 bits (on a 64 bit machine)
constexpr static inline size_t gc_hash(const void *ptr) noexcept {
	return ((uintptr_t)ptr) >> 3ull;
}


/**
 * The allocation object.
 *
 * The allocation object holds all metadata for a memory location
 * in one place.
 */
struct Allocation {
	const void* ptr;		  // mem pointer
	size_t size;              // allocated size in bytes
	struct Allocation* next;  // separate chaining
	char tag;                 // the tag for mark-and-sweep

	/**
	 * Create a new allocation object.
	 *
	 * Creates a new allocation object using the system `malloc_`.
	 *
	 * @parthis[in] ptr The pointer to the memory to manage.
	 * @parthis[in] size The size of the memory range pointed to by `ptr`.
	 * @returns Pointer to the new allocation instance.
	 */
	constexpr Allocation(const void *ptr,
	                     const size_t size) noexcept :
	    ptr(ptr), size(size), next(nullptr), tag(GC_TAG_NONE) {
	}

	///
	// constexpr ~Allocation() noexcept { }

	/// print some basic information about the class
	constexpr static void info() {
		std::cout << " { name: \"ALlocation\", "
				  << " }\n";
	}
};

/**
 * The allocation hash map.
 *
 * The core data structure is a hash map that holds the allocation
 * objects and allows O(1) retrieval given the memory location. Collision
 * resolution is implemented using separate chaining.
 */
template<const AlignmentConfig &config=configAlignment>
struct AllocationMap {
private:
	constexpr AllocationMap() = default;

public:
	constexpr static size_t alignment = config.alignment;

	size_t capacity;
	size_t min_capacity;
	double downsize_factor;
	double upsize_factor;
	double sweep_factor;
	size_t sweep_limit;
	size_t size;

	Allocation** allocs;

	/// \return the fraction of slots which are already occupied
	[[nodiscard]] constexpr inline double load_factor() noexcept {
		return (double) size / (double) capacity;
	}

	/// create a new allocation
	/// \param min_capacity
	/// \param capacity
	/// \param sweep_factor
	/// \param downsize_factor
	/// \param upsize_factor
	constexpr AllocationMap(size_t min_capacity,
	        				size_t capacity,
	        				double sweep_factor,
	        				double downsize_factor,
	        				double upsize_factor) noexcept {
		this->min_capacity = next_prime(min_capacity);
		this->capacity = next_prime(capacity);
		if (this->capacity < this->min_capacity) { this->capacity = this->min_capacity; }

		this->sweep_factor = sweep_factor;
		this->sweep_limit = (int) (sweep_factor * this->capacity);
		this->downsize_factor = downsize_factor;
		this->upsize_factor = upsize_factor;
		this->allocs = (Allocation**) calloc(this->capacity, sizeof(Allocation*));
		this->size = 0;
	}

	///
	constexpr ~AllocationMap() noexcept {
		Allocation *alloc, *tmp;
		for (size_t i = 0; i < capacity; ++i) {
			if ((alloc = allocs[i])) {
				// Make sure to follow the chain inside a bucket
				while (alloc) {
					tmp = alloc;
					alloc = alloc->next;
					// free the management structure
					free(tmp);
				}
			}
		}

		free(this->allocs);
	}

	///
	/// \param new_capacity
	/// \return
	constexpr void resize(const size_t new_capacity) noexcept {
		if (new_capacity <= this->min_capacity) {
			return;
		}
		// Replaces the existing items array in the hash table
		// with a resized one and pushes items into the new, correct buckets
		Allocation** resized_allocs = (Allocation **)calloc(new_capacity, sizeof(Allocation*));

		for (size_t i = 0; i < this->capacity; ++i) {
			Allocation* alloc = this->allocs[i];
			while (alloc) {
				Allocation* next_alloc = alloc->next;
				size_t new_index = gc_hash(alloc->ptr) % new_capacity;
				alloc->next = resized_allocs[new_index];
				resized_allocs[new_index] = alloc;
				alloc = next_alloc;
			}
		}
		free(this->allocs);
		capacity = new_capacity;
		allocs = resized_allocs;
		sweep_limit = size + this->sweep_factor * (this->capacity - this->size);
	}

	///
	/// \return
	constexpr bool resize_to_fit() noexcept {
		double _load_factor = load_factor();
		if (_load_factor > upsize_factor) {
			resize(next_prime(capacity * 2));
			return true;
		}

		if (_load_factor < downsize_factor) {
			resize(next_prime(capacity / 2));
			return true;
		}
		return false;
	}

	/// \param ptr a pointer to a memory alloction
	/// \return either nullptr, if `ptr` is not found or
	/// 		the corresponding allocation
	constexpr Allocation* get(const void *ptr) noexcept {
		const size_t index = gc_hash(ptr) % capacity;
		Allocation* cur = allocs[index];
		while(cur) {
			if (cur->ptr == ptr) {
				return cur;
			}
			cur = cur->next;
		}

		return nullptr;
	}

	///
	/// \param b
	/// \return
	constexpr inline Allocation* get(const Blk &b) noexcept {
		return get(b.ptr);
	}

	/// inserts the ptr into a new alloction
	/// \param ptr pointer to the allocation to store
	/// \param size size of the allocation
	/// \return
	constexpr Allocation* put(const void* ptr,
	                          const size_t size) noexcept {
		const size_t index = gc_hash(ptr) % capacity;
		Allocation* alloc = new Allocation(ptr, size);
		ASSERT(alloc);

		Allocation* cur = allocs[index];
		Allocation* prev = nullptr;
		/* Upsert if ptr is already known (e.g. dtor update). */
		while(cur != nullptr) {
			if (cur->ptr == ptr) {
				// found it
				alloc->next = cur->next;
				if (!prev) {
					// position 0
					this->allocs[index] = alloc;
				} else {
					// in the list
					prev->next = alloc;
				}

				delete cur;
				return alloc;

			}
			prev = cur;
			cur = cur->next;
		}

		/* Insert at the front of the separate chaining list */
		cur = this->allocs[index];
		alloc->next = cur;
		this->allocs[index] = alloc;
		this->size += 1;
		const void* p = alloc->ptr;
		if (resize_to_fit()) {
			alloc = get(p);
		}

		return alloc;
	}

	///
	/// \param b
	/// \return
	constexpr inline Allocation* put(const Blk &b) noexcept {
		return put(b.ptr, b.len);
	}


	/// NOTE: does not free `ptr`
	/// \param ptr
	/// \param allow_resize
	constexpr void remove(const void* ptr,
	                      bool allow_resize=false) noexcept {
		// ignores unknown keys
		size_t index = gc_hash(ptr) % this->capacity;
		Allocation* cur = this->allocs[index];
		Allocation* prev = nullptr;
		Allocation* next;

		while(cur != nullptr) {
			next = cur->next;
			if (cur->ptr == ptr) {
				// found it
				if (!prev) {
					// first item in list
					this->allocs[index] = cur->next;
				} else {
					// not the first item in the list
					prev->next = cur->next;
				}

				delete cur;
				this->size--;
			} else {
				// move on
				prev = cur;
			}
			cur = next;
		}

		if (allow_resize) {
			resize_to_fit();
		}
	}

	/// \param b
	/// \return
	constexpr inline void remove(const Blk &b, const bool allow_resize=false) noexcept {
		remove(b.ptr, allow_resize);
	}

	constexpr static void info() {
		std::cout << " { name \"allocation_map\""
		          << ", alignment:" << alignment
				  << " }\n";
	}
};


///
/// \tparam Alloc
template<class Alloc=std::allocator<int>>
struct GarbageCollector {
	AllocationMap<>* allocs; // allocation map
	bool paused;                  // (temporarily) switch gc on/off
	void *bos;                    // bottom of stack

	///
	/// \param bos bottom of stack
	explicit constexpr GarbageCollector(void *bos) noexcept :
	    GarbageCollector(bos, 1024, 1024, 0.2, 0.8, 0.5)
	{}

	/// k
	/// \param bos
	/// \param initial_capacity
	/// \param min_capacity
	/// \param downsize_load_factor
	/// \param upsize_load_factor
	/// \param sweep_factor
	constexpr GarbageCollector(void *bos,
					  		   size_t initial_capacity,
					  		   const size_t min_capacity,
					  		   const double downsize_load_factor,
					  		   const double upsize_load_factor,
					  		   double sweep_factor) noexcept : paused(false), bos(bos) {
		double downsize_limit = downsize_load_factor > 0.0 ? downsize_load_factor : 0.2;
		double upsize_limit = upsize_load_factor > 0.0 ? upsize_load_factor : 0.8;
		sweep_factor = sweep_factor > 0.0 ? sweep_factor : 0.5;
		initial_capacity = initial_capacity < min_capacity ? min_capacity : initial_capacity;
		allocs = new AllocationMap(min_capacity, initial_capacity, sweep_factor, downsize_limit, upsize_limit);
		ASSERT(allocs);
	}

	/// stops the garbage collector and frees everything
	constexpr ~GarbageCollector() {
		const size_t bytes_freed = stop();
		(void)bytes_freed;
	}

	/// if `count` is specified: `calloc` will be called
	/// otherwise the normal `mallloc`
	/// \param count number of elements to allocate
	/// \param size size of the element to allocate
	/// \return a pointer to the allocation or nullptr
	[[nodiscard]] void* mcalloc(const size_t count,
	              				const size_t size) noexcept {
		if (count == 0) {
			return malloc(size);
		}

		return calloc(count, size);
	}

	///
	/// \return
	[[nodiscard]] constexpr inline bool needs_sweep() noexcept {
		ASSERT(allocs);
		return allocs->size > allocs->sweep_limit;
	}

	void* allocate(const size_t count,
	               const size_t size) noexcept {
		/* Allocation logic that generalizes over malloc/calloc. */

		/* Check if we reached the high-water mark and need to clean up */
		if (needs_sweep() && !paused) {
			const size_t freed_mem = run();
			(void)freed_mem;
		}

		/* With cleanup out of the way, attempt to allocate memory */
		void* ptr = mcalloc(count, size);
		const size_t alloc_size = count ? count * size : size;
		/* If allocation fails, force an out-of-policy run to free some memory and try again. */
		if (!ptr && !paused && (errno == EAGAIN || errno == ENOMEM)) {
			run();
			ptr = mcalloc(count, size);
		}
		/* Start managing the memory we received from the system */
		if (ptr) {
			Allocation* alloc = allocs->put(ptr, alloc_size);
			/* Deal with metadata allocation failure */
			if (alloc) {
				ptr = (void *)alloc->ptr;
			} else {
				/* We failed to allocate the metadata, fail cleanly. */
				free_(ptr);
				ptr = nullptr;
			}
		}
		return ptr;
	}

	void make_root(void* ptr) {
		Allocation* alloc = allocs->get(ptr);
		if (alloc) {
			alloc->tag |= GC_TAG_ROOT;
		}
	}

	inline void *malloc_(const size_t size) {
		return malloc_ext(size);
	}

	void* malloc_static(const size_t size) {
		void* ptr = malloc_ext(size);
		make_root(ptr);
		return ptr;
	}

	void* make_static(void* ptr) {
		make_root(ptr);
		return ptr;
	}

	inline void* malloc_ext(const size_t size) noexcept {
		return allocate(0, size);
	}


	inline void* calloc_ext(const size_t count,
	                 		const size_t size) noexcept {
		return allocate(count, size);
	}


	void* realloc_(void* p,
                    const size_t size) {
		Allocation* alloc = allocs->get(p);
		if (p && !alloc) {
			// the user passed an unknown pointer
			errno = EINVAL;
			return nullptr;
		}
		void* q = realloc(p, size);
		if (!q) {
			// realloc failed but p is still valid
			return nullptr;
		}
		if (!p) {
			// allocation, not reallocation
			Allocation* alloc = allocs->put(q, size);
			return (void *)alloc->ptr;
		}

		if (p == q) {
			// successful reallocation w/o copy
			alloc->size = size;
		} else {
			// successful reallocation w/ copy
			allocs->remove(p, true);
			allocs->put(q, size);
		}
		return q;
	}

	void free_(void* ptr) {
		Allocation* alloc = allocs->get(ptr);
		if (alloc) {
			// if (alloc->dtor) {
			// 	alloc->dtor(ptr);
			// }
			allocs->remove(ptr, true);
			free(ptr);
		}
	}

	void pause() noexcept {
		paused = true;
	}

	void resume() noexcept {
		paused = false;
	}

	void mark_alloc(void* ptr) noexcept {
		Allocation* alloc = allocs->get(ptr);
		/* Mark if alloc exists and is not tagged already, otherwise skip */
		if (alloc && !(alloc->tag & GC_TAG_MARK)) {
			alloc->tag |= GC_TAG_MARK;
			/* Iterate over allocation contents and mark them as well */
			for (char* p = (char*) alloc->ptr;
				 p <= (char*) alloc->ptr + alloc->size - PTRSIZE;
				 ++p) {
				mark_alloc(*(void **) p);
			}
		}
	}

	void mark_stack() {
		void *tos = __builtin_frame_address(0);
		void *_bos = this->bos;
		/* The stack grows towards smaller memory addresses, hence we scan tos->bos.
	     * Stop scanning once the distance between tos & bos is too small to hold a valid pointer */
		for (char* p = (char*) tos; p <= (char*)_bos - PTRSIZE; ++p) {
			mark_alloc(*(void **) p);
		}
	}

	void mark_roots() noexcept {
		for (size_t i = 0; i < allocs->capacity; ++i) {
			Allocation* chunk = allocs->allocs[i];
			while (chunk) {
				if (chunk->tag & GC_TAG_ROOT) {
					mark_alloc((void *) chunk->ptr);
				}
				chunk = chunk->next;
			}
		}
	}

	void mark() noexcept {
		/* Note: We only look at the stack and the heap, and ignore BSS. */
		/* Scan the heap for roots */
		mark_roots();
		/* Dump registers onto stack and scan the stack */
		void (GarbageCollector::*ptr)() = &GarbageCollector::mark_stack;
		jmp_buf ctx;
		memset(&ctx, 0, sizeof(jmp_buf));
		setjmp(ctx);
		// was soll schon schief gehen
		(*this.*ptr)();
	}

	size_t sweep() noexcept {
		size_t total = 0;
		for (size_t i = 0; i < allocs->capacity; ++i) {
			Allocation* chunk = allocs->allocs[i];
			Allocation* next;
			/* Iterate over separate chaining */
			while (chunk) {
				if (chunk->tag & GC_TAG_MARK) {
					/* unmark */
					chunk->tag &= ~GC_TAG_MARK;
					chunk = chunk->next;
				} else {
					/* no reference to this chunk, hence delete it */
					total += chunk->size;
					//if (chunk->dtor) {
					//	chunk->dtor(chunk->ptr);
					//}
					free((void *) chunk->ptr);
					ASSERT(chunk);
					/* and remove it from the bookkeeping */
					next = chunk->next;
					allocs->remove((void *)chunk->ptr, false);
					chunk = next;
				}
			}
		}

		allocs->resize_to_fit();
		return total;
	}

	/**
	* Unset the ROOT tag on all roots on the heap.
	*
	* @parthis gc A pointer to a garbage collector instance.
	*/
	void unroot_roots() noexcept {
		for (size_t i = 0; i < allocs->capacity; ++i) {
			Allocation* chunk = allocs->allocs[i];
			while (chunk) {
				if (chunk->tag & GC_TAG_ROOT) {
					chunk->tag &= ~GC_TAG_ROOT;
				}
				chunk = chunk->next;
			}
		}
	}

	[[nodiscard]] size_t stop() noexcept {
		unroot_roots();
		size_t collected = sweep();
		delete(allocs);
		return collected;
	}

	///
	/// \return freed bytes
	size_t run() noexcept {
		mark();
		return sweep();
	}

	///
	[[nodiscard]] char*strdup(const char* s) noexcept {
		size_t len = strlen(s) + 1;
		void *_new = malloc_(len);

		if (_new == nullptr) {
			return nullptr;
		}

		return (char*) memcpy(_new, s, len);
	}

	constexpr static void info() {
		std::cout <<" { name: \"Allocation\", "
				  << " }\n";
	}
};



#endif//CRYPTANALYSISLIB_GC_SIMPLE_H
