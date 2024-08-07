#ifndef CRYPTANALYSISLIB_GC_SIMPLE_H
#define CRYPTANALYSISLIB_GC_SIMPLE_H

#include "math/math.h"
#include <setjmp.h>

/*
 * The size of a pointer.
 */
#define PTRSIZE sizeof(char*)

/*
 * Allocations can temporarily be tagged as "marked" an part of the
 * mark-and-sweep implementation or can be tagged as "roots" which are
 * not automatically garbage collected. The latter allows the implementation
 * of global variables.
 */
#define GC_TAG_NONE 0x0
#define GC_TAG_ROOT 0x1
#define GC_TAG_MARK 0x2

///
/// \parthis ptr
/// \return
constexpr static inline size_t gc_hash(const void *ptr) noexcept {
	return ((uintptr_t)ptr) >> 3;
}


/**
 * The allocation object.
 *
 * The allocation object holds all metadata for a memory location
 * in one place.
 */
struct Allocation {
	const void* ptr;          // mem pointer
	size_t size;              // allocated size in bytes
	struct Allocation* next;  // separate chaining
	char tag;                 // the tag for mark-and-sweep

	/**
	 * Create a new allocation object.
	 *
	 * Creates a new allocation object using the system `malloc`.
	 *
	 * @parthis[in] ptr The pointer to the memory to manage.
	 * @parthis[in] size The size of the memory range pointed to by `ptr`.
	 * @returns Pointer to the new allocation instance.
	 */
	constexpr Allocation(const void *ptr,
	                     const size_t size) noexcept :
	    ptr(ptr), size(size){
		this->tag = GC_TAG_NONE;
		this->next = nullptr;
	}

	///
	constexpr ~Allocation() noexcept { }
};

/**
 * The allocation hash map.
 *
 * The core data structure is a hash map that holds the allocation
 * objects and allows O(1) retrieval given the memory location. Collision
 * resolution is implemented using separate chaining.
 */
struct AllocationMap {
	constexpr static size_t alignment = 32;

	size_t capacity;
	size_t min_capacity;
	double downsize_factor;
	double upsize_factor;
	double sweep_factor;
	size_t sweep_limit;
	size_t size;

	alignas(alignment) Allocation** allocs;

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

	constexpr void resize(size_t new_capacity) noexcept {
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

	constexpr Allocation* get(const void *ptr) noexcept {
		size_t index = gc_hash(ptr) % capacity;
		Allocation* cur = allocs[index];
		while(cur) {
			if (cur->ptr == ptr) {
				return cur;
			}
			cur = cur->next;
		}

		return nullptr;
	}

	constexpr Allocation* put(const void* ptr,
	                          const size_t size) noexcept {
		size_t index = gc_hash(ptr) % capacity;
		Allocation* alloc = new Allocation(ptr, size);
		Allocation* cur = allocs[index];
		Allocation* prev = nullptr;
		/* Upsert if ptr is already known (e.g. dtor update). */
		while(cur != nullptr) {
			if (cur->ptr == ptr) {
				// found it
				alloc->next = cur->next;
				if (!prev) {
					// position 0
					allocs[index] = alloc;
				} else {
					// in the list
					prev->next = alloc;
				}
				free(cur);
				return alloc;

			}
			prev = cur;
			cur = cur->next;
		}

		/* Insert at the front of the separate chaining list */
		cur = allocs[index];
		alloc->next = cur;
		allocs[index] = alloc;
		this->size += 1;
		const void* p = alloc->ptr;
		if (resize_to_fit()) {
			alloc = get(p);
		}
		return alloc;
	}


	constexpr void remove(void* ptr,
	                      bool allow_resize) noexcept {
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
				delete(cur);
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

	constexpr static void info() {
		// TODO die fnkt überall
		std::cout << " { name \"allocation_map\""
		          << ", alignment:" << alignment
				  << " }" << std::endl;
	}
};



struct GarbageCollector {
	struct AllocationMap* allocs; // allocation map
	bool paused;                  // (temporarily) switch gc on/off
	void *bos;                    // bottom of stack
	size_t min_size;


	void* gc_mcalloc(size_t count, size_t size) {
		if (!count) return malloc(size);
		return calloc(count, size);
	}

	bool needs_sweep() {
		return allocs->size > allocs->sweep_limit;
	}

	void* gc_allocate(const size_t count, const  size_t size) {
		/* Allocation logic that generalizes over malloc/calloc. */

		/* Check if we reached the high-water mark and need to clean up */
		if (needs_sweep() && !paused) {
			size_t freed_mem = gc_run();
			printf("freed mem: %lu\n", freed_mem);
		}
		/* With cleanup out of the way, attempt to allocate memory */
		void* ptr = gc_mcalloc(count, size);
		size_t alloc_size = count ? count * size : size;
		/* If allocation fails, force an out-of-policy run to free some memory and try again. */
		if (!ptr && !paused && (errno == EAGAIN || errno == ENOMEM)) {
			gc_run();
			ptr = gc_mcalloc(count, size);
		}
		/* Start managing the memory we received from the system */
		if (ptr) {
			Allocation* alloc = allocs->put(ptr, alloc_size);
			/* Deal with metadata allocation failure */
			if (alloc) {
				ptr = (void *)alloc->ptr;
			} else {
				/* We failed to allocate the metadata, fail cleanly. */
				free(ptr);
				ptr = nullptr;
			}
		}
		return ptr;
	}

	void gc_make_root(void* ptr) {
		Allocation* alloc = allocs->get(ptr);
		if (alloc) {
			alloc->tag |= GC_TAG_ROOT;
		}
	}

	inline void* gc_malloc(const size_t size) {
		return gc_malloc_ext(size);
	}

	void* gc_malloc_static(const size_t size) {
		void* ptr = gc_malloc_ext(size);
		gc_make_root(ptr);
		return ptr;
	}

	void* gc_make_static(void* ptr) {
		gc_make_root(ptr);
		return ptr;
	}

	void* gc_malloc_ext(const size_t size) {
		return gc_allocate(0, size);
	}


	void* gc_calloc(const size_t count,
	                const size_t size) {
		return gc_calloc_ext(count, size);
	}


	void* gc_calloc_ext(const size_t count,
                        const size_t size) {
		return gc_allocate(count, size);
	}


	void* gc_realloc(void* p,
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

	void gc_free(void* ptr) {
		Allocation* alloc = allocs->get(ptr);
		if (alloc) {
			// if (alloc->dtor) {
			// 	alloc->dtor(ptr);
			// }
			free(ptr);
			allocs->remove(ptr, true);
		}
	}

	void gc_start(void* bos) {
		gc_start_ext(bos, 1024, 1024, 0.2, 0.8, 0.5);
	}

	void gc_start_ext(void* bos,
					  size_t initial_capacity,
					  size_t min_capacity,
					  double downsize_load_factor,
					  double upsize_load_factor,
					  double sweep_factor)
	{
		double downsize_limit = downsize_load_factor > 0.0 ? downsize_load_factor : 0.2;
		double upsize_limit = upsize_load_factor > 0.0 ? upsize_load_factor : 0.8;
		sweep_factor = sweep_factor > 0.0 ? sweep_factor : 0.5;
		this->paused = false;
		this->bos = bos;
		initial_capacity = initial_capacity < min_capacity ? min_capacity : initial_capacity;
		this->allocs = new AllocationMap(min_capacity, initial_capacity,
	                                   sweep_factor, downsize_limit, upsize_limit);
	}

	void gc_pause() noexcept {
		paused = true;
	}

	void gc_resume() noexcept {
		paused = false;
	}

	void gc_mark_alloc(void* ptr) {
		Allocation* alloc = allocs->get(ptr);
		/* Mark if alloc exists and is not tagged already, otherwise skip */
		if (alloc && !(alloc->tag & GC_TAG_MARK)) {
			alloc->tag |= GC_TAG_MARK;
			/* Iterate over allocation contents and mark them as well */
			for (char* p = (char*) alloc->ptr;
				 p <= (char*) alloc->ptr + alloc->size - PTRSIZE;
				 ++p) {
				gc_mark_alloc(*(void**)p);
			}
		}
	}

	void gc_mark_stack() {
		void *tos = __builtin_frame_address(0);
		void *_bos = this->bos;
		/* The stack grows towards smaller memory addresses, hence we scan tos->bos.
	     * Stop scanning once the distance between tos & bos is too small to hold a valid pointer */
		for (char* p = (char*) tos; p <= (char*)_bos - PTRSIZE; ++p) {
			gc_mark_alloc(*(void**)p);
		}
	}

	void gc_mark_roots() {
		for (size_t i = 0; i < allocs->capacity; ++i) {
			Allocation* chunk = allocs->allocs[i];
			while (chunk) {
				if (chunk->tag & GC_TAG_ROOT) {
					gc_mark_alloc((void *)chunk->ptr);
				}
				chunk = chunk->next;
			}
		}
	}

	void gc_mark() {
		/* Note: We only look at the stack and the heap, and ignore BSS. */
		/* Scan the heap for roots */
		gc_mark_roots();
		/* Dump registers onto stack and scan the stack */
		// void (*_mark_stack)(void) = gc_mark_stack;
		// jmp_buf ctx;
		// memset(&ctx, 0, sizeof(jmp_buf));
		// setjmp(ctx);
		// _mark_stack();
		// TODO
		gc_mark_stack();
	}

	size_t gc_sweep() {
		size_t total = 0;
		for (size_t i = 0; i < allocs->capacity; ++i) {
			Allocation* chunk = allocs->allocs[i];
			Allocation* next = nullptr;
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
					free((void *)chunk->ptr);
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
	void gc_unroot_roots(){
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

	size_t gc_stop() {
		gc_unroot_roots();
		size_t collected = gc_sweep();
		delete allocs;
		return collected;
	}

	size_t gc_run() {
		gc_mark();
		return gc_sweep();
	}

	char* gc_strdup (const char* s) {
		size_t len = strlen(s) + 1;
		void *_new = gc_malloc(len);

		if (_new == nullptr) {
			return nullptr;
		}

		return (char*) memcpy(_new, s, len);
	}
};



#endif//CRYPTANALYSISLIB_GC_SIMPLE_H
