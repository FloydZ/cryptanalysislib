/*-
 * Copyright (c) 2012 Gleb Kurtsou
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef CRYPTANALYSISLIB_CONTAINER_HASHMAP_THASHMAP_H
#define CRYPTANALYSISLIB_CONTAINER_HASHMAP_THASHMAP_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "alloc/alloc.h"
#include "memory/memory.h"

/// thashmap: an intrusive, ordered radix trie for 30 bit keys
/// (Gleb Kurtsou, FreeBSD, https://github.com/glk/thashmap).
///
/// The key is split into 6 sub keys of 5 bits (the top level uses the bits
/// [25, 30)). Every trie node ("slot") stores only its present children
/// (a 32 bit map + 4, 8, ..., 28 pointers); a node with more children is a
/// full 32 entry array ("slotmax"). Leaves are "buckets": intrusive singly
/// linked lists of the elements with the same key. A leaf is stored at the
/// first level, where its key differs from all other keys (path compression
/// towards the root), so most keys are found after 1 or 2 levels.
///
/// The nodes are allocated from a `thm_pool` of 4 KiB pages, which are
/// divided into 128 units of 4 pointers; a node of length `slen` (1..8)
/// occupies `slen` consecutive units. The pages are kept in queues by the
/// longest free run ("rank"), nodes are extended in place if possible.
///
/// NOTE: the pool does not grow by itself: `insert` returns nullptr if no
/// 	node can be allocated, call `thm_pool::new_block()` and retry.
/// NOTE: not thread safe.
///
/// Usage:
/// \code
/// 	struct elem { uint32_t key; thm_entry entry; };
/// 	using map_t = thashmap<elem, offsetof(elem, entry), offsetof(elem, key)>;
/// 	thm_pool pool;
/// 	map_t m(pool);
/// 	elem e{42, {}};
/// 	while (m.insert(&e) == nullptr) { pool.new_block(); }
/// 	thm_bucket *b = m.find(42);           // all elements with key 42
/// 	for (elem *x = map_t::bucket_first(b); x; x = map_t::bucket_next(x)) {}
/// 	thm_cursor c;                         // ordered iteration
/// 	for (b = m.first(&c); b; b = m.next(&c)) {}
/// 	m.remove(&e);
/// \endcode
/// Tests: `tests/container/thashmap.cpp`
namespace cryptanalysislib {
	/// embed into the elements of a `thashmap` (one per map)
	struct thm_entry {
		thm_entry *te_next;
	};

	/// opaque: the first element of a bucket (all elements with the same key)
	struct thm_bucket;

	namespace internal::thm {
		constexpr uintptr_t PTR_MASK_RESERVED = 0x03;
		constexpr uintptr_t PTR_MASK_VALUE = ~PTR_MASK_RESERVED;
		constexpr uintptr_t PTR_MASK_SLEN = 0x02;
		constexpr uintptr_t PTR_MASK_SLOT = 0x01;

		constexpr uint32_t SLEN_MAX = 8;
		constexpr uint32_t SUBKEY_MAX = 6;
		constexpr uint32_t POOL_RANK_MAX = SLEN_MAX + 1;

		constexpr uint32_t SLOT_MAX_ENTRIES = 32;
		constexpr uint32_t SLOT_MIN_ENTRIES = 4;
		/// 32 bytes on 64 bit archs
		constexpr uint32_t SLOT_SIZE = SLOT_MIN_ENTRIES * uint32_t(sizeof(uintptr_t));
		/// 4096 on 64 bit archs, 2048 on 32 bit archs
		constexpr uint32_t PAGE_SIZE = 128 * SLOT_SIZE;
		constexpr uintptr_t PTR_MASK_PAGE = ~uintptr_t(PAGE_SIZE - 1);

		constexpr uint32_t SUBKEY_SHIFT = 5;
		constexpr uint32_t SUBKEY_MASK = SLOT_MAX_ENTRIES - 1;

		constexpr uint64_t MASK_01010101 = 0x5555555555555555ULL;
		constexpr uint64_t MASK_01000100 = 0x4444444444444444ULL;
		constexpr uint64_t MASK_00110011 = 0x3333333333333333ULL;
		constexpr uint64_t MASK_11001100 = 0xccccccccccccccccULL;
		constexpr uint64_t MASK_00001111 = 0x0f0f0f0f0f0f0f0fULL;
		constexpr uint64_t MASK_11110000 = 0xf0f0f0f0f0f0f0f0ULL;
		constexpr uint64_t MASK_00010000 = 0x1010101010101010ULL;
		constexpr uint64_t MASK_00100000 = 0x2020202020202020ULL;
		constexpr uint64_t MASK_01000000 = 0x4040404040404040ULL;

		/// header of a page, in its first unit(s)
		struct page {
			uint64_t tp_map1;
			uint64_t tp_map2;
			/// next page in the queue, the rank in the low bits
			uintptr_t tp_q_next;
			uintptr_t *tp_q_prevp;
		};

		/// units occupied by the page header
		constexpr uint64_t PAGE_STRUCT_SLOTS =
		        (1ULL << ((sizeof(page) + SLOT_SIZE - 1) / SLOT_SIZE)) - 1;
		static_assert(PAGE_STRUCT_SLOTS == 0x1 || PAGE_STRUCT_SLOTS == 0x3);
		constexpr uint64_t PAGE_MAP1_EMPTY = ~PAGE_STRUCT_SLOTS;
		constexpr uint64_t PAGE_MAP2_EMPTY = ~uint64_t(0);

		/// \return sub key `n` (0: the bits [25, 30)) of `k`
		[[nodiscard]] constexpr inline uint32_t subkey(const uint32_t k, const uint32_t n) noexcept {
			return (k >> (SUBKEY_SHIFT * (5 - n))) & SUBKEY_MASK;
		}

		/// \return the level of the bit `ind` (counted from the top of a 32 bit word)
		[[nodiscard]] constexpr inline uint32_t subkey_bitind(const uint32_t ind) noexcept {
			return (ind - 2) / 5;
		}

		[[nodiscard]] constexpr inline uint64_t key_bit(const uint32_t ind) noexcept {
			return 1ULL << ind;
		}

		[[nodiscard]] inline void *ptr_get_value(const uintptr_t ptr) noexcept {
			return reinterpret_cast<void *>(ptr & PTR_MASK_VALUE);
		}

		// A node ("slot") is an array of pointer sized words: word 0 is the
		// map of the present sub keys, words 1, 2, ... the present entries
		// (in the order of the sub keys). A full node ("slotmax") has no map,
		// word i is the entry of the sub key i. The node length (1..8 units)
		// is stored in the bit 1 of the words 1, 2, 3.
		using slot_t = uintptr_t;

		[[nodiscard]] inline uintptr_t &slot_map(slot_t *slot) noexcept { return slot[0]; }
		[[nodiscard]] inline uintptr_t *slot_entries(slot_t *slot) noexcept { return slot + 1; }
		[[nodiscard]] inline uintptr_t *slotmax_entry(slot_t *slot, const uint32_t ind) noexcept {
			return slot + ind;
		}

		[[nodiscard]] inline uint32_t slot_get_slen(slot_t *slot) noexcept {
			const uintptr_t *e = slot_entries(slot);
			return uint32_t(((e[0] & PTR_MASK_SLEN) >> 1) |
			                (e[1] & PTR_MASK_SLEN) |
			                ((e[2] & PTR_MASK_SLEN) << 1)) + 1;
		}

		inline void slot_set_slen(slot_t *slot, uint32_t slen) noexcept {
			uintptr_t *e = slot_entries(slot);
			slen -= 1;
			e[0] = (e[0] & ~PTR_MASK_SLEN) | ((uintptr_t(slen) << 1) & PTR_MASK_SLEN);
			e[1] = (e[1] & ~PTR_MASK_SLEN) | (uintptr_t(slen) & PTR_MASK_SLEN);
			e[2] = (e[2] & ~PTR_MASK_SLEN) | ((uintptr_t(slen) >> 1) & PTR_MASK_SLEN);
		}

		[[nodiscard]] inline page *ptr_get_page(const uintptr_t ptr) noexcept {
			return reinterpret_cast<page *>(ptr & PTR_MASK_PAGE);
		}

		[[nodiscard]] inline uintptr_t ptr_get_pageflag(const uintptr_t ptr) noexcept {
			return ptr & ~PTR_MASK_PAGE;
		}

		inline void ptr_init_page(uintptr_t *ptr, const void *addr, const uintptr_t flag) noexcept {
			assert((flag & PTR_MASK_PAGE) == 0);
			*ptr = (reinterpret_cast<uintptr_t>(addr) & PTR_MASK_PAGE) | ptr_get_pageflag(flag);
		}

		inline void ptr_set_page(uintptr_t *ptr, const void *addr) noexcept {
			ptr_init_page(ptr, addr, *ptr & ~PTR_MASK_PAGE);
		}

		inline void ptr_set_slot(uintptr_t *ptr, slot_t *slot) noexcept {
			assert((reinterpret_cast<uintptr_t>(slot) & PTR_MASK_RESERVED) == 0);
			*ptr = (*ptr & PTR_MASK_SLEN) | PTR_MASK_SLOT | reinterpret_cast<uintptr_t>(slot);
		}

		inline void ptr_set_value(uintptr_t *ptr, const void *value) noexcept {
			assert((reinterpret_cast<uintptr_t>(value) & PTR_MASK_RESERVED) == 0);
			*ptr = (*ptr & PTR_MASK_SLEN) | reinterpret_cast<uintptr_t>(value);
		}

		[[nodiscard]] inline page *addr_get_page(const void *addr) noexcept {
			return ptr_get_page(reinterpret_cast<uintptr_t>(addr));
		}

		/// \return the unit index of `slot` in its page
		[[nodiscard]] inline uint32_t slot_get_offset(const slot_t *slot) noexcept {
			const uintptr_t off = reinterpret_cast<uintptr_t>(slot) & ~PTR_MASK_PAGE;
			return uint32_t(off / SLOT_SIZE);
		}

		/// sets the bucket to `entry`, without changing `entry->te_next`
		inline void bucket_set(uintptr_t *ptr, thm_entry *entry) noexcept {
			ptr_set_value(ptr, entry);
		}

		/// prepends `entry` to the bucket
		inline void bucket_insert(uintptr_t *ptr, thm_entry *entry) noexcept {
			entry->te_next = static_cast<thm_entry *>(ptr_get_value(*ptr));
			ptr_set_value(ptr, entry);
		}

		/// removes `entry` from the bucket, it must be in the bucket
		inline void bucket_remove(uintptr_t *ptr, thm_entry *entry) noexcept {
			thm_entry *i = static_cast<thm_entry *>(ptr_get_value(*ptr));
			if (entry == i) {
				ptr_set_value(ptr, entry->te_next);
				return;
			}

			while (i->te_next != entry) {
				i = i->te_next;
			}
			i->te_next = entry->te_next;
		}

		/// counts the free runs of a page map (`frags[l-1]`: runs of length
		/// `l`, longer runs count as `len / maxfrag` runs of `maxfrag`)
		inline void pagemap_count_fragments(uint64_t map, uint64_t *frags, const uint32_t maxfrag) noexcept {
			uint32_t bit, count, left;

			map = ~map;
			left = 64;

			do {
				bit = uint32_t(map & 1u);
				if (bit) {
					if (~map == 0ULL) {
						count = left;
					} else {
						count = uint32_t(__builtin_ctzll(~map));
					}
				} else {
					if (map == 0ULL) {
						count = left;
					} else {
						count = uint32_t(__builtin_ctzll(map));
					}
				}
				assert(count > 0);
				// NOTE: `count` can be 64, a shift by 64 is undefined
				map = (count >= 64) ? 0 : (map >> count);
				left -= count;
				if (bit != 0) {
					continue;
				}
				if (count > maxfrag) {
					frags[maxfrag - 1] += count / maxfrag;
				} else {
					frags[count - 1] += 1;
				}
			} while (left > 0);
		}
	} // end namespace internal::thm

	/// cursor for the ordered iteration (`first`, `last`, `next`, `prev`) and
	/// the result of `find`/`nfind`
	struct thm_cursor {
		uintptr_t *tc_path[internal::thm::SUBKEY_MAX + 1];
		uint32_t tc_level;
	};

	/// statistics of a `thm_pool`
	struct thm_pool_stats {
		uint64_t tp_pages;
		uint64_t tp_slots;
		uint64_t tp_slots_free;
		uint64_t tp_queues[internal::thm::POOL_RANK_MAX];
		uint64_t tp_fragments[internal::thm::SLEN_MAX];
	};

	/// allocator of the trie nodes of one or more `thashmap`s
	class thm_pool {
		using page = internal::thm::page;
		using slot_t = internal::thm::slot_t;

		struct pool_queue {
			uintptr_t tpq_first;
			uintptr_t *tpq_last;
		};

		pool_queue tp_queue[internal::thm::POOL_RANK_MAX];

		friend class thm_head;

		// queues of pages by rank (TAILQ_* equivalents), the rank of a page
		// is stored in the low bits of its `tp_q_next`

		[[nodiscard]] page *pool_first(const uint32_t rank) noexcept {
			return internal::thm::ptr_get_page(tp_queue[rank].tpq_first);
		}

		[[nodiscard]] static page *pool_next(page *pg) noexcept {
			return internal::thm::ptr_get_page(pg->tp_q_next);
		}

		[[nodiscard]] static uint32_t page_get_rank(page *pg) noexcept {
			const uint32_t rank = uint32_t(pg->tp_q_next & ~internal::thm::PTR_MASK_PAGE);
			assert(rank < internal::thm::POOL_RANK_MAX);
			return rank;
		}

		void pool_insert_head(const uint32_t rank, page *pg) noexcept {
			using namespace internal::thm;
			pool_queue *qhead = &tp_queue[rank];
			page *first = ptr_get_page(qhead->tpq_first);

			assert(rank < POOL_RANK_MAX);
			assert(first == nullptr || first->tp_q_prevp == &qhead->tpq_first);

			ptr_init_page(&pg->tp_q_next, first, rank);
			if (first != nullptr) {
				first->tp_q_prevp = &pg->tp_q_next;
			} else {
				qhead->tpq_last = &pg->tp_q_next;
			}
			qhead->tpq_first = reinterpret_cast<uintptr_t>(pg);
			pg->tp_q_prevp = &qhead->tpq_first;
		}

		void pool_insert_tail(const uint32_t rank, page *pg) noexcept {
			using namespace internal::thm;
			pool_queue *qhead = &tp_queue[rank];

			assert(rank < POOL_RANK_MAX);
			assert(qhead->tpq_last != nullptr);

			ptr_init_page(&pg->tp_q_next, nullptr, rank);
			pg->tp_q_prevp = qhead->tpq_last;
			ptr_set_page(qhead->tpq_last, pg);
			qhead->tpq_last = &pg->tp_q_next;
		}

		void pool_remove(const uint32_t rank, page *pg) noexcept {
			using namespace internal::thm;
			pool_queue *qhead = &tp_queue[rank];
			page *next = ptr_get_page(pg->tp_q_next);

			assert(rank == page_get_rank(pg));
			assert(rank < POOL_RANK_MAX);
			assert(next == nullptr || next->tp_q_prevp == &pg->tp_q_next);
			assert(ptr_get_page(*pg->tp_q_prevp) == pg);

			if (next != nullptr) {
				next->tp_q_prevp = pg->tp_q_prevp;
			} else {
				qhead->tpq_last = pg->tp_q_prevp;
			}

			ptr_set_page(pg->tp_q_prevp, next);
		}

		/// \return a new page, nullptr if the allocation failed
		[[nodiscard]] static page *page_alloc() noexcept {
			using namespace internal::thm;
			page *pg = static_cast<page *>(cryptanalysislib::aligned_alloc(PAGE_SIZE, PAGE_SIZE));
			if (pg == nullptr) {
				return nullptr;
			}

			assert(pg == addr_get_page(pg));

			pg->tp_map1 = PAGE_MAP1_EMPTY;
			pg->tp_map2 = PAGE_MAP2_EMPTY;
			return pg;
		}

		static void page_free(page *pg) noexcept {
			assert(pg->tp_map1 == internal::thm::PAGE_MAP1_EMPTY);
			assert(pg->tp_map2 == internal::thm::PAGE_MAP2_EMPTY);
			cryptanalysislib::aligned_free(pg);
		}

		/// \return a free run of `slen` units in `pg`, nullptr if there is none
		[[nodiscard]] static slot_t *page_alloc_slot(page *pg, const uint32_t slen) noexcept {
			using namespace internal::thm;
			uint64_t m1, m2, mt, mask;
			uint32_t off, ioff;

			ioff = 1;
			m1 = pg->tp_map1;
			m2 = pg->tp_map2;

		again:
			// bit i of m1/m2: the units [i, i + slen) (shifted by ioff - 1) are free
			switch (slen) {
				case 8:
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					m1 = (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
					m1 = (m1 & MASK_00001111) & ((m1 & MASK_11110000) >> 4);
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					m2 = (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
					m2 = (m2 & MASK_00001111) & ((m2 & MASK_11110000) >> 4);
					break;
				case 7:
					mt = (m1 & MASK_01000000) >> 6;
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					mt = mt & ((m1 & MASK_00010000) >> 4);
					m1 = mt & (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
					mt = (m2 & MASK_01000000) >> 6;
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					mt = mt & ((m2 & MASK_00010000) >> 4);
					m2 = mt & (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
					break;
				case 6:
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					mt = (m1 & MASK_00010000) >> 4;
					m1 = mt & (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					mt = (m2 & MASK_00010000) >> 4;
					m2 = mt & (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
					break;
				case 5:
					mt = ((m1 & MASK_00010000) & ((m1 & MASK_00100000) >> 1)) >> 4;
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					m1 = mt & (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
					mt = ((m2 & MASK_00010000) & ((m2 & MASK_00100000) >> 1)) >> 4;
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					m2 = mt & (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
					break;
				case 4:
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					m1 = (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					m2 = (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
					break;
				case 3:
					mt = (m1 & MASK_01000100) >> 2;
					m1 = mt & m1 & (m1 >> 1) & MASK_01010101;
					mt = (m2 & MASK_01000100) >> 2;
					m2 = mt & m2 & (m2 >> 1) & MASK_01010101;
					break;
				case 2:
					m1 = m1 & (m1 >> 1) & MASK_01010101;
					m2 = m2 & (m2 >> 1) & MASK_01010101;
					break;
				case 1:
					break;
			}

			mask = (1ULL << slen) - 1;
			if (m1 != 0) {
				off = uint32_t(__builtin_ctzll(m1)) + ioff - 1;
				mask <<= off;
				assert((mask & pg->tp_map1) == mask);
				pg->tp_map1 &= ~mask;
				return reinterpret_cast<slot_t *>(pg) + (off * SLOT_MIN_ENTRIES);
			} else if (m2 != 0) {
				off = uint32_t(__builtin_ctzll(m2)) + ioff - 1;
				mask <<= off;
				assert((mask & pg->tp_map2) == mask);
				pg->tp_map2 &= ~mask;
				return reinterpret_cast<slot_t *>(pg) + ((off + 64) * SLOT_MIN_ENTRIES);
			}

			// the aligned search failed, try the other alignments
			if (ioff * 2 < slen) {
				m1 = pg->tp_map1 >> ioff;
				m2 = pg->tp_map2 >> ioff;
				ioff++;
				goto again;
			}

			return nullptr;
		}

		/// \return `rank` if `pg` has a free run for it, otherwise 0
		[[nodiscard]] static uint32_t page_promote_rank(page *pg, const uint32_t rank) noexcept {
			using namespace internal::thm;
			uint64_t m1, m2;
			uint32_t count;

			assert(rank < POOL_RANK_MAX);

			count = uint32_t(__builtin_popcountll(pg->tp_map1) + __builtin_popcountll(pg->tp_map2));
			if (count <= rank * 3) {
				return 0;
			}

			m1 = pg->tp_map1;
			m2 = pg->tp_map2;
			m1 = m1 & (m1 >> 1) & MASK_01010101;
			m2 = m2 & (m2 >> 1) & MASK_01010101;
			if (rank >= 4) {
				m1 = (m1 & MASK_00110011) & ((m1 & MASK_11001100) >> 2);
				m2 = (m2 & MASK_00110011) & ((m2 & MASK_11001100) >> 2);
				if (rank >= 8) {
					m1 = (m1 & MASK_00001111) & ((m1 & MASK_11110000) >> 4);
					m2 = (m2 & MASK_00001111) & ((m2 & MASK_11110000) >> 4);
				}
			}

			if (m1 != 0 || m2 != 0) {
				return rank;
			}
			return 0;
		}

		/// called after units of `pg` were freed: moves it to a higher rank,
		/// frees it if it is empty
		void page_promote(page *pg) noexcept {
			using namespace internal::thm;
			const uint32_t rank = page_get_rank(pg);

			if (rank == POOL_RANK_MAX - 1) {
				if (pg->tp_map1 == PAGE_MAP1_EMPTY && pg->tp_map2 == PAGE_MAP2_EMPTY) {
					pool_remove(rank, pg);
					page_free(pg);
				}
				return;
			}

			const uint32_t rank_new = page_promote_rank(pg, rank + 1);
			if (rank_new > rank) {
				pool_remove(rank, pg);
				pool_insert_tail(rank_new, pg);
			}
		}

		/// \return the rank of `pg`: the longest run length < `rank`, of which
		/// 	`pg` has at least two free runs
		[[nodiscard]] static uint32_t page_demote_rank(page *pg, uint32_t rank) noexcept {
			uint64_t frags[internal::thm::SLEN_MAX] = {0};
			uint64_t acc;

			if (rank == 0) {
				return 0;
			}

			internal::thm::pagemap_count_fragments(pg->tp_map1, frags, rank);
			internal::thm::pagemap_count_fragments(pg->tp_map2, frags, rank);

			for (acc = 0; rank > 0; rank--) {
				if (frags[rank - 1] + acc < 2) {
					acc += frags[rank - 1];
					continue;
				}
				return rank;
			}

			return uint32_t(acc);
		}

		/// called if `pg` has no free run of `slen` units
		void page_demote(page *pg, const uint32_t rank, const uint32_t slen) noexcept {
			const uint32_t rank_new = page_demote_rank(pg, slen - 1);

			if (rank_new != rank) {
				pool_remove(rank, pg);
				pool_insert_head(rank_new, pg);
			}
		}

		/// converts the full node `slotmax` into a node of length `slen_new`
		static void slotmax_fix_shrink(slot_t *slotmax, slot_t *slot_new, const uint32_t slen_new) noexcept {
			using namespace internal::thm;
			uintptr_t xbuf[SLOT_MAX_ENTRIES], *buf;
			uintptr_t map = 0, keybit = 1;
			uint32_t i, keyind;

			for (i = 0; i < SLOT_MAX_ENTRIES; i++) {
				if (ptr_get_value(*slotmax_entry(slotmax, i)) != nullptr) {
					map |= keybit;
				}
				keybit <<= 1;
			}

			if (slot_new != slotmax) {
				buf = slot_entries(slot_new);
			} else {
				buf = xbuf;
			}
			for (i = 0, keyind = 0; i < SLOT_MAX_ENTRIES; i++) {
				if (ptr_get_value(*slotmax_entry(slotmax, i)) == nullptr) {
					continue;
				}
				buf[keyind] = *slotmax_entry(slotmax, i);
				keyind++;
			}
			assert(keyind > 0 && keyind < slen_new * SLOT_MIN_ENTRIES);
			if (slot_new == slotmax) {
				cryptanalysislib::memcpy(slot_entries(slot_new), buf, keyind);
			}

			slot_map(slot_new) = map;
			slot_set_slen(slot_new, slen_new);
		}

		/// converts the node `slot_old` into the full node `slotmax`
		static void slotmax_fix_extend(slot_t *slot_old, slot_t *slotmax) noexcept {
			using namespace internal::thm;
			uintptr_t xbuf[SLOT_MAX_ENTRIES], *buf;
			uint32_t smap, i, keyind;

			smap = uint32_t(slot_map(slot_old));
			keyind = 0;

			assert(smap != 0);
			assert(uint32_t(__builtin_popcount(smap)) + 1 <=
			       slot_get_slen(slot_old) * SLOT_MIN_ENTRIES);

			if (slot_old != slotmax) {
				buf = slotmax;
			} else {
				buf = xbuf;
			}

			cryptanalysislib::memset(buf, uintptr_t(0), SLOT_MAX_ENTRIES);
			while (smap != 0) {
				i = uint32_t(__builtin_ctz(smap));
				smap &= ~uint32_t(key_bit(i));
				buf[i] = slot_entries(slot_old)[keyind];
				keyind++;
			}

			if (slot_old == slotmax) {
				cryptanalysislib::memcpy(slotmax, buf, SLOT_MAX_ENTRIES);
			}

			slot_set_slen(slotmax, SLEN_MAX);
		}

		/// frees the units [slen_new, slen_old) of `slot`
		void slot_shrink(slot_t *slot, const uint32_t slen_old, const uint32_t slen_new) noexcept {
			using namespace internal::thm;
			uint64_t mask, *map;
			uint32_t off, maskoff;

			assert(slen_old > slen_new);

			if (slen_new != 0) {
				if (slen_old == SLEN_MAX) {
					slotmax_fix_shrink(slot, slot, slen_new);
				} else {
					slot_set_slen(slot, slen_new);
				}
			}

			page *pg = addr_get_page(slot);
			off = slot_get_offset(slot);
			if (off < 64) {
				maskoff = off;
				map = &pg->tp_map1;
			} else {
				maskoff = off - 64;
				map = &pg->tp_map2;
			}

			// 0x00XXXNNN
			mask = ((1ULL << slen_old) - 1) - ((1ULL << slen_new) - 1);
			mask <<= maskoff;

			assert((*map & mask) == 0);
			*map |= mask;

			page_promote(pg);
		}

		/// extends `slot` in place to `slen_new` units
		/// \return false if the units behind `slot` are not free
		bool slot_tryextend(slot_t *slot, const uint32_t slen_old, const uint32_t slen_new) noexcept {
			using namespace internal::thm;
			uint64_t mask, *map;
			uint32_t off, maskoff;

			page *pg = addr_get_page(slot);
			off = slot_get_offset(slot);

			// Forbid crossing map boundary
			if (((off ^ (off + slen_new)) & (64 | 128)) != 0) {
				return false;
			}

			if (off < 64) {
				maskoff = off;
				map = &pg->tp_map1;
			} else {
				maskoff = off - 64;
				map = &pg->tp_map2;
			}

			// 0x00NNNPPP
			mask = ((1ULL << slen_new) - 1) - ((1ULL << slen_old) - 1);
			mask <<= maskoff;

			if ((*map & mask) == mask) {
				*map &= ~mask;
				if (slen_new == SLEN_MAX) {
					slotmax_fix_extend(slot, slot);
				} else {
					slot_set_slen(slot, slen_new);
				}
				return true;
			}

			return false;
		}

		[[nodiscard]] slot_t *slot_alloc_step(page *pg, const uint32_t slen) noexcept {
			const uint32_t rank = page_get_rank(pg);
			if (rank < slen) {
				return nullptr;
			}

			slot_t *slot = page_alloc_slot(pg, slen);
			if (slot != nullptr) {
				return slot;
			}

			page_demote(pg, rank, slen);
			return nullptr;
		}

		/// \param hint[in]: an address in the page to try first, or nullptr
		/// \return `slen` units, nullptr if no page has a free run
		[[nodiscard]] slot_t *slot_alloc(const uint32_t slen, const void *hint) noexcept {
			using namespace internal::thm;
			assert(slen <= SLEN_MAX);

			if (hint != nullptr) {
				slot_t *slot = slot_alloc_step(addr_get_page(hint), slen);
				if (slot != nullptr) {
					return slot;
				}
			}

			for (uint32_t rank = slen; rank < POOL_RANK_MAX; rank++) {
				page *pg = pool_first(rank);
				while (pg != nullptr) {
					page *npage = pool_next(pg);
					slot_t *slot = slot_alloc_step(pg, slen);
					if (slot != nullptr) {
						return slot;
					}
					pg = npage;
				}
			}

			return nullptr;
		}

		[[nodiscard]] slot_t *slot_alloc_zero(const uint32_t slen, const void *hint) noexcept {
			slot_t *slot = slot_alloc(slen, hint);
			if (slot == nullptr) {
				return nullptr;
			}

			cryptanalysislib::memset(slot, uintptr_t(0), slen * internal::thm::SLOT_MIN_ENTRIES);
			internal::thm::slot_set_slen(slot, slen);
			return slot;
		}

		void slot_free(slot_t *slot, const uint32_t slen) noexcept {
			slot_shrink(slot, slen, 0);
		}

	public:
		/// allocates the first page
		thm_pool() noexcept {
			for (uint32_t rank = 0; rank < internal::thm::POOL_RANK_MAX; rank++) {
				pool_queue *qhead = &tp_queue[rank];
				qhead->tpq_first = 0;
				qhead->tpq_last = &qhead->tpq_first;
			}

			new_block();
		}

		thm_pool(const thm_pool &) = delete;
		thm_pool &operator=(const thm_pool &) = delete;

		/// frees all pages; all maps using this pool must be destroyed before
		~thm_pool() noexcept {
			for (uint32_t rank = 0; rank < internal::thm::POOL_RANK_MAX; rank++) {
				page *pg;
				while ((pg = pool_first(rank)) != nullptr) {
					pool_remove(rank, pg);
					cryptanalysislib::aligned_free(pg);
				}
			}
		}

		/// adds a new page (call it, if `thashmap::insert` returned nullptr)
		/// \return false if the allocation failed
		bool new_block() noexcept {
			page *pg = page_alloc();
			if (pg == nullptr) {
				return false;
			}
			pool_insert_head(internal::thm::POOL_RANK_MAX - 1, pg);
			return true;
		}

		/// \param stats[out]: statistics of the pool
		void get_stats(thm_pool_stats *stats) noexcept {
			using namespace internal::thm;
			uint64_t frags[SLEN_MAX] = {0};
			uint64_t used_total = 0;

			*stats = thm_pool_stats{};
			for (uint32_t rank = 0; rank < POOL_RANK_MAX; rank++) {
				for (page *pg = pool_first(rank); pg != nullptr; pg = pool_next(pg)) {
					used_total += uint64_t(__builtin_popcountll(~pg->tp_map1) +
					                       __builtin_popcountll(~pg->tp_map2));

					stats->tp_pages += 1;
					stats->tp_slots += 128;
					stats->tp_queues[rank] += 1;

					pagemap_count_fragments(pg->tp_map1, frags, SLEN_MAX);
					pagemap_count_fragments(pg->tp_map2, frags, SLEN_MAX);
				}
			}

			stats->tp_slots_free = stats->tp_slots - used_total;
			for (uint32_t i = 0; i < SLEN_MAX; i++) {
				stats->tp_fragments[i] = frags[i];
			}
		}
	};

	/// untyped trie: the elements are given by their `thm_entry`, the 30 bit
	/// key is at a fixed byte offset from the entry. Use `thashmap`.
	class thm_head {
		using slot_t = internal::thm::slot_t;

		thm_pool *th_pool;
		uintptr_t th_root;
		/// byte offset of the key from the entry
		ptrdiff_t th_keyoffset;

		[[nodiscard]] uint32_t entry_get_key(const void *entry) const noexcept {
			const uint32_t *keyp = reinterpret_cast<const uint32_t *>(
			        static_cast<const char *>(entry) + th_keyoffset);
			return *keyp & key_mask;
		}

		static void cursor_push(thm_cursor *cr, uintptr_t *entp) noexcept {
			cr->tc_level++;
			assert(cr->tc_level < internal::thm::SUBKEY_MAX + 1);
			cr->tc_path[cr->tc_level] = entp;
		}

		/// descends to the first leaf below `cr->tc_path[cr->tc_level]`
		static thm_bucket *first_impl(thm_cursor *cr) noexcept {
			using namespace internal::thm;
			void *entval = ptr_get_value(*cr->tc_path[cr->tc_level]);
			uintptr_t *entp;

			do {
				slot_t *slot = static_cast<slot_t *>(entval);
				if (slot_get_slen(slot) == SLEN_MAX) {
					entp = nullptr;
					for (uint32_t i = 0; i < SLOT_MAX_ENTRIES; i++) {
						if (ptr_get_value(*slotmax_entry(slot, i)) != nullptr) {
							entp = slotmax_entry(slot, i);
							break;
						}
					}
					if (entp == nullptr) {
						return nullptr;
					}
				} else {
					if (slot_map(slot) == 0) {
						return nullptr;
					}
					entp = &slot_entries(slot)[0];
				}
				cursor_push(cr, entp);
				entval = ptr_get_value(*entp);
			} while ((*entp & PTR_MASK_SLOT) != 0);

			return static_cast<thm_bucket *>(entval);
		}

		/// descends to the last leaf below `cr->tc_path[cr->tc_level]`
		static thm_bucket *last_impl(thm_cursor *cr) noexcept {
			using namespace internal::thm;
			void *entval = ptr_get_value(*cr->tc_path[cr->tc_level]);
			uintptr_t *entp;

			do {
				slot_t *slot = static_cast<slot_t *>(entval);
				if (slot_get_slen(slot) == SLEN_MAX) {
					entp = nullptr;
					for (uint32_t i = SLOT_MAX_ENTRIES; i-- > 0;) {
						if (ptr_get_value(*slotmax_entry(slot, i)) != nullptr) {
							entp = slotmax_entry(slot, i);
							break;
						}
					}
					if (entp == nullptr) {
						return nullptr;
					}
				} else {
					if (slot_map(slot) == 0) {
						return nullptr;
					}
					const uint32_t i = uint32_t(__builtin_popcount(uint32_t(slot_map(slot))));
					entp = &slot_entries(slot)[i - 1];
				}
				cursor_push(cr, entp);
				entval = ptr_get_value(*entp);
			} while ((*entp & PTR_MASK_SLOT) != 0);

			return static_cast<thm_bucket *>(entval);
		}

		/// \return the entry after `entp` in `slot`, nullptr if there is none
		static uintptr_t *next_step(slot_t *slot, uintptr_t *entp) noexcept {
			using namespace internal::thm;
			if (slot_get_slen(slot) == SLEN_MAX) {
				uint32_t i = uint32_t(entp - slotmax_entry(slot, 0));
				assert(i < SLOT_MAX_ENTRIES);
				for (i += 1; i < SLOT_MAX_ENTRIES; i++) {
					entp = slotmax_entry(slot, i);
					if (ptr_get_value(*entp) != nullptr) {
						return entp;
					}
				}
				return nullptr;
			}

			const uint32_t count = uint32_t(__builtin_popcount(uint32_t(slot_map(slot))));
			const uint32_t i = uint32_t(entp - slot_entries(slot));
			assert(i < count);
			if (i + 1 >= count) {
				return nullptr;
			}
			return &slot_entries(slot)[i + 1];
		}

		/// \return the entry before `entp` in `slot`, nullptr if there is none
		static uintptr_t *prev_step(slot_t *slot, uintptr_t *entp) noexcept {
			using namespace internal::thm;
			if (slot_get_slen(slot) == SLEN_MAX) {
				uint32_t i = uint32_t(entp - slotmax_entry(slot, 0));
				assert(i < SLOT_MAX_ENTRIES);
				while (i-- > 0) {
					entp = slotmax_entry(slot, i);
					if (ptr_get_value(*entp) != nullptr) {
						return entp;
					}
				}
				return nullptr;
			}

			const uint32_t i = uint32_t(entp - slot_entries(slot));
			if (i == 0) {
				return nullptr;
			}
			return &slot_entries(slot)[i - 1];
		}

		/// \return the entry of the sub key `key` in `slot`, nullptr if there is none
		static uintptr_t *find_step(slot_t *slot, const uint32_t key) noexcept {
			using namespace internal::thm;
			assert(key < SLOT_MAX_ENTRIES);

			if (slot_get_slen(slot) == SLEN_MAX) {
				uintptr_t *entp = slotmax_entry(slot, key);
				if (ptr_get_value(*entp) == nullptr) {
					return nullptr;
				}
				return entp;
			}

			uint32_t smap = uint32_t(slot_map(slot));
			const uint64_t keybit = key_bit(key);
			if ((smap & keybit) == 0) {
				return nullptr;
			}

			smap &= uint32_t(keybit - 1);
			return &slot_entries(slot)[__builtin_popcount(smap)];
		}

		/// \return the bucket of `key`; if not found, `cr->tc_path[cr->tc_level]`
		/// 	is the node, in which the search stopped
		thm_bucket *find_impl(const uint32_t key, thm_cursor *cr) const noexcept {
			using namespace internal::thm;
			cr->tc_path[0] = const_cast<uintptr_t *>(&th_root);
			void *entval = ptr_get_value(th_root);

			for (uint32_t n = 0; n < SUBKEY_MAX; n++) {
				uintptr_t *entp = find_step(static_cast<slot_t *>(entval), subkey(key, n));
				if (entp == nullptr) {
					cr->tc_level = n;
					return nullptr;
				}
				cr->tc_path[n + 1] = entp;
				entval = ptr_get_value(*entp);
				if ((*entp & PTR_MASK_SLOT) == 0) {
					cr->tc_level = n + 1;
					break;
				}
				assert(n + 1 < SUBKEY_MAX);
			}

			if (key != entry_get_key(entval)) {
				cr->tc_level--;
				return nullptr;
			}

			return static_cast<thm_bucket *>(entval);
		}

		/// \return the entry of the sub key `key` in the node `*slotp`, which
		/// 	is created (the node is extended or reallocated) if needed;
		/// 	nullptr if the allocation failed
		uintptr_t *insert_step(uintptr_t *slotp, const uint32_t key) noexcept {
			using namespace internal::thm;
			thm_pool *pool = th_pool;
			assert(key < SLOT_MAX_ENTRIES);

			slot_t *slot = static_cast<slot_t *>(ptr_get_value(*slotp));
			uint32_t slen = slot_get_slen(slot);
			if (slen == SLEN_MAX) {
				return slotmax_entry(slot, key);
			}

			const uint32_t smap = uint32_t(slot_map(slot));
			const uint64_t keybit = key_bit(key);
			const uint32_t keyind = uint32_t(__builtin_popcount(smap & uint32_t(keybit - 1)));

			if ((smap & keybit) != 0) {
				return &slot_entries(slot)[keyind];
			}

			// Insert new entry
			const uint32_t count = uint32_t(__builtin_popcount(smap));
			assert(count + 1 <= slen * SLOT_MIN_ENTRIES);

			if (count + 1 + 1 > slen * SLOT_MIN_ENTRIES &&
			    pool->slot_tryextend(slot, slen, slen + 1)) {
				slen += 1;
				if (slen == SLEN_MAX) {
					return slotmax_entry(slot, key);
				}
			}
			if (count + 1 + 1 <= slen * SLOT_MIN_ENTRIES) {
				slot_map(slot) |= keybit;
				for (uint32_t i = count; i > keyind; i--) {
					slot_entries(slot)[i] = slot_entries(slot)[i - 1];
				}
				slot_entries(slot)[keyind] = 0;
				if (keyind < 3) {
					slot_set_slen(slot, slen);
				}
				assert(uint32_t(__builtin_popcount(uint32_t(slot_map(slot)))) + 1 <=
				       slen * SLOT_MIN_ENTRIES);
				return &slot_entries(slot)[keyind];
			}

			// Allocate larger slot
			slot_t *oslot = slot;
			slot = pool->slot_alloc(slen + 1, oslot);
			if (slot == nullptr) {
				return nullptr;
			}

			ptr_set_slot(slotp, slot);
			slen += 1;

			if (slen == SLEN_MAX) {
				thm_pool::slotmax_fix_extend(oslot, slot);
				pool->slot_free(oslot, slen - 1);
				return slotmax_entry(slot, key);
			}

			slot_map(slot) = slot_map(oslot) | keybit;
			for (uint32_t i = 0; i < keyind; i++) {
				slot_entries(slot)[i] = slot_entries(oslot)[i];
			}
			slot_entries(slot)[keyind] = 0;
			for (uint32_t i = keyind; i < count; i++) {
				slot_entries(slot)[i + 1] = slot_entries(oslot)[i];
			}
			slot_set_slen(slot, slen);
			pool->slot_free(oslot, slen - 1);

			assert(uint32_t(__builtin_popcount(uint32_t(slot_map(slot)))) + 1 <=
			       slen * SLOT_MIN_ENTRIES);

			return &slot_entries(slot)[keyind];
		}

		/// replaces the leaf `*slotp_top` (`entry2` with `key2`) by a chain of
		/// nodes down to the level, where `key1` and `key2` differ
		/// \return the entry of `entry1`, nullptr if the allocation failed
		uintptr_t *insert_mkslot(uintptr_t *slotp_top, uint32_t subkey_n,
		                         thm_entry *entry1, const uint32_t key1,
		                         thm_entry *entry2, const uint32_t key2) noexcept {
			using namespace internal::thm;
			uintptr_t *ent1, *ent2;
			uintptr_t *slotp = slotp_top;

			uint32_t nslots = uint32_t(__builtin_clz(key1 ^ key2));
			nslots = subkey_bitind(nslots) - subkey_n + 1;
			assert(nslots >= 1);

			// one allocation for all nodes of the chain (each of length 1)
			slot_t *slot = th_pool->slot_alloc(nslots, slotp);
			if (slot == nullptr) {
				return nullptr;
			}
			cryptanalysislib::memset(slot, uintptr_t(0), nslots * SLOT_MIN_ENTRIES);

			while (true) {
				const uint32_t subkey1 = subkey(key1, subkey_n);
				const uint32_t subkey2 = subkey(key2, subkey_n);
				if (subkey1 != subkey2) {
					slot_map(slot) = (uintptr_t(1) << subkey1) | (uintptr_t(1) << subkey2);
					ptr_set_slot(slotp, slot);
					break;
				}

				slot_map(slot) = uintptr_t(1) << subkey1;
				ptr_set_slot(slotp, slot);
				slotp = &slot_entries(slot)[0];
				slot = slot + SLOT_MIN_ENTRIES;
				subkey_n++;
			}

			if (key1 < key2) {
				ent1 = &slot_entries(slot)[0];
				ent2 = &slot_entries(slot)[1];
			} else {
				ent1 = &slot_entries(slot)[1];
				ent2 = &slot_entries(slot)[0];
			}
			bucket_insert(ent1, entry1);
			bucket_set(ent2, entry2);

			return ent1;
		}

		/// removes the entry `entp` (of the sub key `key`) from `slot`
		/// \return true if `slot` is empty now
		bool remove_step(slot_t *slot, uintptr_t *entp, const uint32_t key) noexcept {
			using namespace internal::thm;
			uint32_t count;
			const uint32_t slen = slot_get_slen(slot);

			if (slen == SLEN_MAX) {
				assert(entp >= slotmax_entry(slot, 0) &&
				       entp < slotmax_entry(slot, SLOT_MAX_ENTRIES));

				ptr_set_value(entp, nullptr);
				count = 0;
				for (uint32_t i = 0; i < SLOT_MAX_ENTRIES; i++) {
					if (ptr_get_value(*slotmax_entry(slot, i)) != nullptr) {
						count++;
					}
				}
			} else {
				const uint32_t keyind = uint32_t(entp - slot_entries(slot));
				const uint64_t keybit = key_bit(key);

				assert(entp >= slot_entries(slot) && keyind < slen * SLOT_MIN_ENTRIES);
				assert((slot_map(slot) & keybit) != 0);

				slot_map(slot) &= ~keybit;
				count = uint32_t(__builtin_popcount(uint32_t(slot_map(slot))));

				for (uint32_t i = keyind; i < count; i++) {
					slot_entries(slot)[i] = slot_entries(slot)[i + 1];
				}
				if (keyind < 3) {
					slot_set_slen(slot, slen);
				}
			}

			if (count == 0) {
				return true;
			}

			// NOTE: as upstream, nodes are only shrunk for SLOT_MIN_ENTRIES > 4,
			// 	i.e. never
			if constexpr (SLOT_MIN_ENTRIES > 4) {
				if (count + 1 + 2 <= (slen - 1) * SLOT_MIN_ENTRIES && slen > 1) {
					th_pool->slot_shrink(slot, slen, slen - 1);
				}
			}

			return false;
		}

		void dump_tree_step(slot_t *slot) const noexcept {
			using namespace internal::thm;
			uintptr_t buf[SLOT_MAX_ENTRIES], *ents;

			if (slot_get_slen(slot) == SLEN_MAX) {
				ents = slot;
			} else {
				ents = buf;
				cryptanalysislib::memset(buf, uintptr_t(0), SLOT_MAX_ENTRIES);
				uint32_t smap = uint32_t(slot_map(slot));
				uint32_t keyind = 0;
				while (smap != 0) {
					const uint32_t i = uint32_t(__builtin_ctz(smap));
					smap &= ~uint32_t(key_bit(i));
					ents[i] = slot_entries(slot)[keyind];
					keyind++;
				}
			}

			printf("S:%p:%u: ", static_cast<void *>(slot), slot_get_slen(slot) * SLOT_MIN_ENTRIES);
			for (uint32_t i = 0; i < SLOT_MAX_ENTRIES; i++) {
				void *entval = ptr_get_value(ents[i]);
				if (entval == nullptr) {
					continue;
				}
				if ((ents[i] & PTR_MASK_SLOT) == 0) {
					printf("%u:D:%p:K%08x ", i, entval, entry_get_key(entval));
				} else {
					printf("%u:S:%p ", i, entval);
				}
			}
			printf("\n");

			for (uint32_t i = 0; i < SLOT_MAX_ENTRIES; i++) {
				void *entval = ptr_get_value(ents[i]);
				if (entval == nullptr || (ents[i] & PTR_MASK_SLOT) == 0) {
					continue;
				}
				dump_tree_step(static_cast<slot_t *>(entval));
			}
		}

	public:
		/// the keys have 30 bits, the upper two bits are ignored
		constexpr static uint32_t key_mask = 0x3fffffff;

		/// \param pool[in]: allocator of the nodes, must outlive this map
		/// \param keyoffset[in]: byte offset of the 32 bit key from the entry
		thm_head(thm_pool &pool, const ptrdiff_t keyoffset) noexcept :
		    th_pool(&pool), th_root(0), th_keyoffset(keyoffset) {
			slot_t *root = pool.slot_alloc_zero(1, nullptr);
			while (root == nullptr && pool.new_block()) {
				root = pool.slot_alloc_zero(1, nullptr);
			}
			assert(root != nullptr);
			th_root = reinterpret_cast<uintptr_t>(root);
		}

		thm_head(const thm_head &) = delete;
		thm_head &operator=(const thm_head &) = delete;

		/// frees the root node; the map must be empty
		~thm_head() noexcept {
			slot_t *slot = static_cast<slot_t *>(internal::thm::ptr_get_value(th_root));
			if (slot != nullptr) {
				th_pool->slot_free(slot, internal::thm::slot_get_slen(slot));
			}
		}

		/// \return true if the map is empty
		[[nodiscard]] bool empty() const noexcept {
			using namespace internal::thm;
			slot_t *slot = static_cast<slot_t *>(ptr_get_value(th_root));

			if (slot_get_slen(slot) == SLEN_MAX) {
				for (uint32_t i = 0; i < SLOT_MAX_ENTRIES; i++) {
					if (ptr_get_value(*slotmax_entry(slot, i)) != nullptr) {
						return false;
					}
				}
				return true;
			}

			return slot_map(slot) == 0;
		}

		/// \param cr[out]: cursor for `next`/`prev`, or nullptr
		/// \return the bucket with the smallest key, nullptr if empty
		thm_bucket *first(thm_cursor *cr = nullptr) noexcept {
			thm_cursor xcr;
			if (cr == nullptr) { cr = &xcr; }

			cr->tc_level = 0;
			cr->tc_path[0] = &th_root;

			thm_bucket *bucket = first_impl(cr);
			assert(bucket != nullptr || cr->tc_level == 0);
			return bucket;
		}

		/// \param cr[out]: cursor for `next`/`prev`, or nullptr
		/// \return the bucket with the largest key, nullptr if empty
		thm_bucket *last(thm_cursor *cr = nullptr) noexcept {
			thm_cursor xcr;
			if (cr == nullptr) { cr = &xcr; }

			cr->tc_level = 0;
			cr->tc_path[0] = &th_root;

			thm_bucket *bucket = last_impl(cr);
			assert(bucket != nullptr || cr->tc_level == 0);
			return bucket;
		}

		/// \param cr[in,out]: cursor of a bucket (`first`, `find`, ...)
		/// \return the bucket with the next larger key, nullptr at the end
		static thm_bucket *next(thm_cursor *cr) noexcept {
			using namespace internal::thm;
			assert(cr->tc_level < SUBKEY_MAX + 1);

			for (; cr->tc_level > 0; cr->tc_level--) {
				slot_t *slot = static_cast<slot_t *>(ptr_get_value(*cr->tc_path[cr->tc_level - 1]));
				uintptr_t *entp = next_step(slot, cr->tc_path[cr->tc_level]);
				if (entp == nullptr) {
					continue;
				}

				cr->tc_path[cr->tc_level] = entp;
				if ((*entp & PTR_MASK_SLOT) == 0) {
					return static_cast<thm_bucket *>(ptr_get_value(*entp));
				}
				return first_impl(cr);
			}

			return nullptr;
		}

		/// \param cr[in,out]: cursor of a bucket (`last`, `find`, ...)
		/// \return the bucket with the next smaller key, nullptr at the end
		static thm_bucket *prev(thm_cursor *cr) noexcept {
			using namespace internal::thm;
			assert(cr->tc_level < SUBKEY_MAX + 1);

			for (; cr->tc_level > 0; cr->tc_level--) {
				slot_t *slot = static_cast<slot_t *>(ptr_get_value(*cr->tc_path[cr->tc_level - 1]));
				uintptr_t *entp = prev_step(slot, cr->tc_path[cr->tc_level]);
				if (entp == nullptr) {
					continue;
				}

				cr->tc_path[cr->tc_level] = entp;
				if ((*entp & PTR_MASK_SLOT) == 0) {
					return static_cast<thm_bucket *>(ptr_get_value(*entp));
				}
				return last_impl(cr);
			}

			return nullptr;
		}

		/// \param key[in]: key (the upper two bits are ignored)
		/// \param cr[out]: cursor for `next`/`prev`, or nullptr
		/// \return the bucket of `key`, nullptr if not found
		thm_bucket *find(uint32_t key, thm_cursor *cr = nullptr) noexcept {
			thm_cursor xcr;
			key &= key_mask;
			if (cr == nullptr) { cr = &xcr; }
			return find_impl(key, cr);
		}

		/// \param key[in]: key (the upper two bits are ignored)
		/// \param cr[out]: cursor for `next`/`prev`, or nullptr
		/// \return the bucket of `key`, or the bucket with the next larger
		/// 	key, nullptr if there is none
		thm_bucket *nfind(uint32_t key, thm_cursor *cr = nullptr) noexcept {
			using namespace internal::thm;
			thm_cursor xcr;
			slot_t *slot;
			uintptr_t *entp;
			void *entval;
			uint32_t subkey_v;

			if (cr == nullptr) { cr = &xcr; }
			key &= key_mask;

			entval = find_impl(key, cr);
			if (entval != nullptr) {
				return static_cast<thm_bucket *>(entval);
			}

		restart:
			subkey_v = subkey(key, cr->tc_level);
			slot = static_cast<slot_t *>(ptr_get_value(*cr->tc_path[cr->tc_level]));

			if (slot_get_slen(slot) == SLEN_MAX) {
				entp = slotmax_entry(slot, subkey_v);
				if (ptr_get_value(*entp) != nullptr) {
					goto found_eq;
				}
				for (uint32_t i = subkey_v + 1; i < SLOT_MAX_ENTRIES; i++) {
					entp = slotmax_entry(slot, i);
					if (ptr_get_value(*entp) != nullptr) {
						goto found_gt;
					}
				}
			} else {
				const uint32_t smap = uint32_t(slot_map(slot));
				if (smap == 0) {
					assert(cr->tc_level == 0);
					return nullptr;
				}
				const uint32_t count = uint32_t(__builtin_popcount(smap));
				const uint64_t keybit = key_bit(subkey_v);
				const uint32_t i = uint32_t(__builtin_popcount(smap & uint32_t(keybit - 1)));
				if (i < count) {
					entp = &slot_entries(slot)[i];
					if ((smap & keybit) != 0) {
						goto found_eq;
					}
					goto found_gt;
				}
			}

			// back track
			entval = next(cr);
			goto done;

		found_gt:
			// NOTE: the leaf is pushed as well, so `next`/`prev` continue from it
			cursor_push(cr, entp);
			if ((*entp & PTR_MASK_SLOT) != 0) {
				entval = first_impl(cr);
			} else {
				entval = ptr_get_value(*entp);
				assert(entry_get_key(entval) > key);
			}
			goto done;

		found_eq:
			cursor_push(cr, entp);
			if ((*entp & PTR_MASK_SLOT) != 0) {
				goto restart;
			}
			entval = ptr_get_value(*entp);
			if (entry_get_key(entval) < key) {
				entval = next(cr);
			}

		done:
			assert(entval == nullptr || entry_get_key(entval) > key);
			return static_cast<thm_bucket *>(entval);
		}

		/// inserts `entry` (prepends it to the bucket of its key)
		/// \return the bucket of `entry`, nullptr if a node could not be
		/// 	allocated (call `thm_pool::new_block()` and retry)
		thm_bucket *insert(thm_entry *entry) noexcept {
			using namespace internal::thm;
			const uint32_t key = entry_get_key(entry);

			uintptr_t *parentp = &th_root;
			uintptr_t *entp = nullptr;
			uint32_t subkey_n = 0;
			for (; subkey_n < SUBKEY_MAX; subkey_n++) {
				entp = insert_step(parentp, subkey(key, subkey_n));
				if (entp == nullptr) {
					return nullptr;
				}
				if ((*entp & PTR_MASK_SLOT) == 0) {
					break;
				}
				assert(subkey_n + 1 < SUBKEY_MAX);
				parentp = entp;
			}

			thm_entry *xentry = static_cast<thm_entry *>(ptr_get_value(*entp));
			if (xentry != nullptr) {
				const uint32_t xkey = entry_get_key(xentry);
				if (xkey != key) {
					entp = insert_mkslot(entp, subkey_n + 1, entry, key, xentry, xkey);
					if (entp == nullptr) {
						return nullptr;
					}
					return static_cast<thm_bucket *>(ptr_get_value(*entp));
				}
			}

			bucket_insert(entp, entry);
			return static_cast<thm_bucket *>(ptr_get_value(*entp));
		}

		/// removes `entry`, which must be in the map
		void remove(thm_entry *entry) noexcept {
			using namespace internal::thm;
			thm_cursor cr;
			const uint32_t key = entry_get_key(entry);

			[[maybe_unused]] void *entval = find_impl(key, &cr);
			assert(entval != nullptr);

			int32_t subkey_n = int32_t(cr.tc_level) - 1;
			uintptr_t *entp = cr.tc_path[cr.tc_level];

			bucket_remove(entp, entry);
			if (ptr_get_value(*entp) != nullptr) {
				return;
			}

			// Remove slot
			for (; subkey_n >= 0; subkey_n--) {
				// slot for entp
				slot_t *slot = static_cast<slot_t *>(ptr_get_value(*cr.tc_path[subkey_n]));
				if (!remove_step(slot, entp, subkey(key, uint32_t(subkey_n)))) {
					break;
				}
				if (subkey_n > 0) {
					th_pool->slot_free(slot, slot_get_slen(slot));
				}
				entp = cr.tc_path[subkey_n];
			}
		}

		/// prints all nodes
		void dump_tree() const noexcept {
			dump_tree_step(static_cast<slot_t *>(internal::thm::ptr_get_value(th_root)));
		}

		/// \return the first element of the bucket
		[[nodiscard]] static thm_entry *bucket_first(thm_bucket *bucket) noexcept {
			return reinterpret_cast<thm_entry *>(bucket);
		}

		/// \return the next element of the bucket, nullptr at the end
		[[nodiscard]] static thm_entry *bucket_next(thm_entry *entry) noexcept {
			return entry->te_next;
		}
	};

	/// Intrusive ordered map from 30 bit keys to buckets of elements `T`
	/// (see the top of this file).
	/// \tparam T element type, contains a `thm_entry` and a `uint32_t` key
	/// \tparam entry_offset `offsetof(T, <entry member>)`
	/// \tparam key_offset `offsetof(T, <key member>)`
	template<typename T, const size_t entry_offset, const size_t key_offset>
	class thashmap {
		thm_head head;

		[[nodiscard]] static thm_entry *field(T *elm) noexcept {
			return reinterpret_cast<thm_entry *>(reinterpret_cast<char *>(elm) + entry_offset);
		}

	public:
		/// \param pool[in]: allocator of the nodes, must outlive this map
		explicit thashmap(thm_pool &pool) noexcept :
		    head(pool, ptrdiff_t(key_offset) - ptrdiff_t(entry_offset)) {}

		/// \return the element of `ent`, nullptr for nullptr
		[[nodiscard]] static T *entry(thm_entry *ent) noexcept {
			if (ent == nullptr) {
				return nullptr;
			}
			return reinterpret_cast<T *>(reinterpret_cast<char *>(ent) - entry_offset);
		}

		[[nodiscard]] bool empty() const noexcept { return head.empty(); }
		thm_bucket *first(thm_cursor *cr = nullptr) noexcept { return head.first(cr); }
		thm_bucket *last(thm_cursor *cr = nullptr) noexcept { return head.last(cr); }
		static thm_bucket *next(thm_cursor *cr) noexcept { return thm_head::next(cr); }
		static thm_bucket *prev(thm_cursor *cr) noexcept { return thm_head::prev(cr); }
		thm_bucket *find(const uint32_t key, thm_cursor *cr = nullptr) noexcept { return head.find(key, cr); }
		thm_bucket *nfind(const uint32_t key, thm_cursor *cr = nullptr) noexcept { return head.nfind(key, cr); }

		/// inserts `elm` (prepends it to the bucket of its key)
		/// \return the bucket of `elm`, nullptr if a node could not be
		/// 	allocated (call `thm_pool::new_block()` and retry)
		thm_bucket *insert(T *elm) noexcept { return head.insert(field(elm)); }

		/// removes `elm`, which must be in the map
		void remove(T *elm) noexcept { head.remove(field(elm)); }

		void dump_tree() const noexcept { head.dump_tree(); }

		/// \return the first element of `bucket`, nullptr for nullptr
		[[nodiscard]] static T *bucket_first(thm_bucket *bucket) noexcept {
			return entry(thm_head::bucket_first(bucket));
		}

		/// \return the next element of the bucket of `elm`, nullptr at the end
		[[nodiscard]] static T *bucket_next(T *elm) noexcept {
			return entry(thm_head::bucket_next(field(elm)));
		}
	};
} // end namespace cryptanalysislib
#endif
