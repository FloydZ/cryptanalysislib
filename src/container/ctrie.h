#ifndef CRYPTANALYSISLIB_CTRIE_H
#define CRYPTANALYSISLIB_CTRIE_H

/// implementation of http://aleksandar-prokopec.com/resources/docs/p137-prokopec.pdf
/// see scala code: https://github.com/reactors-io/reactors/blob/master/reactors-common/jvm/src/main/scala/io/reactors/common/concurrent/CacheTrie.scala

/// TODO/Ideas:
/// the whole pointer annotating paring with computational jump/goto table
/// KeyCompareClass
/// all new in function wrappen: darauf avhten das alignmet und pointer stimmen
/// anstatt pointer: referenzen: viel spass
/// mask schreiben die ANode und ANNode catched



#include <array>
#include <cstdint>
#include <cstring>
#include <sys/types.h>

#include "helper.h"
#include "math/log.h"
#include "atomic_primitives.h"

template<class K, class V>
class SNode_t {
public:
	uint64_t hash = 0;
	K key;
	V value;
	void *txn = nullptr;
};

#define ANodeValue  0b0001
#define ANNodeValue 0b0010
#define AANodeValue 0b0011 // special value matching
#define SNodeValue  0b0100
#define ENodeValue  0b0101
#define FNodeValue  0b0110
#define LNodeValue  0b1000
#define XNodeValue  0b1001
#define FullValue   0b1111

#define isFVNode(ptr) (ptr == nullptr)
#define isFSNode(ptr) ((uintptr_t )ptr == (uintptr_t )0)

#define isANode(ptr)  ((((uintptr_t)ptr) & FullValue) == ANodeValue)
#define isANNode(ptr) ((((uintptr_t)ptr) & FullValue) == ANNodeValue)
#define isAANode(ptr) ((((uintptr_t)ptr) & AANodeValue)) // special: checks if either ANode or ANNode
#define isSNode(ptr)  ((((uintptr_t)ptr) & FullValue) == SNodeValue)
#define isENode(ptr)  ((((uintptr_t)ptr) & FullValue) == ENodeValue)
#define isFNode(ptr)  ((((uintptr_t)ptr) & FullValue) == FNodeValue)
#define isLNode(ptr)  ((((uintptr_t)ptr) & FullValue) == LNodeValue)
#define isXNode(ptr)  ((((uintptr_t)ptr) & FullValue) == XNodeValue)

#define maskANode(ptr)  ((ptr) = (ANode *) ((uintptr_t)(ptr) ^ ANodeValue))
#define maskANNode(ptr) ((ptr) = (ANNode *)((uintptr_t)(ptr) ^ ANNodeValue))
#define maskSNode(ptr)  ((ptr) = (SNode *) ((uintptr_t)(ptr) ^ SNodeValue))
#define maskENode(ptr)  ((ptr) = (ENode *) ((uintptr_t)(ptr) ^ ENodeValue))
#define maskFNode(ptr)  ((ptr) = (FNode *) ((uintptr_t)(ptr) ^ FNodeValue))
#define maskLNode(ptr)  ((ptr) = (LNode *) ((uintptr_t)(ptr) ^ LNodeValue))
#define maskXNode(ptr)  ((ptr) = (XNode *) ((uintptr_t)(ptr) ^ XNodeValue))

#define accessNode(ptr) ((uintptr_t )ptr & 0xfffffffffffffff8)
#define isNode(ptr)     ((uintptr_t )ptr & FullValue)

#define NUMBER_CPUS 1 // TODO

// #if NUMBER_CPUS == 1
// #define READ(ptr, pos) 	((void *)((uintptr_t *)(ptr))[pos])
// #define READ_TXN(ptr) 	(*((void *)(uintptr_t *)(((uint8_t *)ptr) + 24)))
// #define READ_WIDE(ptr) 	(*((void *)(uintptr_t *)(((uint8_t *)ptr) + 40)))
//
// #define READ_A_COUNT(ptr) 	(*(uintptr_t *)(((uint8_t *)ptr) + (16*8))))
// #define READ_AN_COUNT(ptr) 	(*uintptr_t *)(((uint8_t *)ptr) + (4*8))))
//
// #define CAS_(ptr, pos, ov, nv) 	(*((void *)((uintptr_t *)(ptr)) + (pos), (uintptr_t *)ov, (uintptr_t)nv))
// #define CAS_WIDE(ptr, ov, nv) 	(*((void *)((uintptr_t *)(((uint8_t *)ptr) + 40u)), (uintptr_t *)ov, (uintptr_t)nv))
// #define CAS_TXN(ptr, ov, nv) 	(*((void *)((uintptr_t *)(((uint8_t *)ptr) + 24u)), (uintptr_t *)ov, (uintptr_t)nv))
//
// #define CAS_CACHE(ov, nv) 		(*((void *)(uintptr_t *)&cache_ptr, (uintptr_t *)ov, (uintptr_t)nv))
//
// #define CAS_A_COUNT(ptr, ov, nv) 	(* ((uintptr_t *)(((uint8_t *)ptr) + (16u*8u))) = nv);
// #define CAS_AN_COUNT(ptr, ov, nv) 	(* ((uintptr_t *)(((uint8_t *)ptr) + (4u*8u))) = nv);
//
// #define WRITE(ptr, pos, nv) (((uintptr_t *)ptr)[pos] = (uintptr_t )nv);
// #else
#define READ(ptr, pos) 	((void *)ACQUIRE(((uintptr_t *)(ptr)) + (pos)))
#define READ_TXN(ptr) 	((void *)ACQUIRE((uintptr_t *)(((uint8_t *)ptr) + 24)))
#define READ_WIDE(ptr) 	((void *)ACQUIRE((uintptr_t *)(((uint8_t *)ptr) + 40)))

#define READ_A_COUNT(ptr) 	((void *)ACQUIRE((uintptr_t *)(((uint8_t *)ptr) + (16*8))))
#define READ_AN_COUNT(ptr) 	((void *)ACQUIRE((uintptr_t *)(((uint8_t *)ptr) + (4*8))))

#define CAS_(ptr, pos, ov, nv) 	((void *)CAS(((uintptr_t *)(ptr)) + (pos), (uintptr_t *)ov, (uintptr_t)nv))
#define CAS_WIDE(ptr, ov, nv) 	((void *)CAS(((uintptr_t *)(((uint8_t *)ptr) + 40u)), (uintptr_t *)ov, (uintptr_t)nv))
#define CAS_TXN(ptr, ov, nv) 	((void *)CAS(((uintptr_t *)(((uint8_t *)ptr) + 24u)), (uintptr_t *)ov, (uintptr_t)nv))

#define CAS_CACHE(ov, nv) 		((void *)CAS((uintptr_t *)&cache_ptr, (uintptr_t *)ov, (uintptr_t)nv))

#define CAS_A_COUNT(ptr, ov, nv) 	((void *)CAS(((uintptr_t *)(((uint8_t *)ptr) + (16*8u))), (uintptr_t *)ov, (uintptr_t)nv))
#define CAS_AN_COUNT(ptr, ov, nv) 	((void *)CAS(((uintptr_t *)(((uint8_t *)ptr) + (4*8u))), (uintptr_t *)ov, (uintptr_t)nv))

#define WRITE(ptr, pos, nv) (STORE(((uintptr_t *)(((uintptr_t *)ptr) + pos)), (uintptr_t )nv))
//#endif


template<class K, class V>
class CacheTrie {
	constexpr static uint32_t alignment = 32;
	constexpr static uint32_t wayness = 16;
	constexpr static uint32_t wMask = (1u << bits_log2(wayness)) - 1u;
	constexpr static uint32_t lW = 4;
	constexpr static uint32_t threads = 1;

	/// some flags to enable/disable features
	constexpr static bool useCompression = true;
	constexpr static bool useCounters = true;
	constexpr static bool useCache = true;

	class ANNode {
		void *data [bits_log2(wayness)] = {nullptr};
		uint64_t load = 0;

	public:
		constexpr void*& operator[](const std::size_t i) noexcept {
			return at(i);
		}

		constexpr void*& at(const std::size_t i) noexcept {
			return data[i];
		}

		[[nodiscard]] constexpr uint64_t size() const noexcept {
			return load;
		}
		constexpr void size(const uint64_t data_) noexcept {
			load = data_;
		}

		constexpr void inc() noexcept {
			load += 1;
		}
	};

	class ANode {
		void *data [wayness] = {nullptr};
		uint64_t load = 0;

	public:
		constexpr void*& operator[](const std::size_t i) noexcept {
			return at(i);
		}

		constexpr void*& at(const std::size_t i) noexcept {
			return data[i];
		}

		[[nodiscard]] constexpr uint64_t size() const noexcept {
			return load;
		}

		constexpr void size(const uint64_t data_) noexcept {
			load = data_;
		}
		constexpr void inc() noexcept {
			load += 1;
		}
	};

	class CacheNode {
	public:
		CacheNode *parent = nullptr;
		uint32_t level = 0;
		uint32_t missCounts[NUMBER_CPUS * 16] = {0};
		CacheNode() = delete;
		CacheNode(CacheNode *parent, uint32_t level) : parent(parent), level(level) {}

		constexpr uint32_t pos() {
			const uint32_t tid = 0;
			return (tid ^ (tid >> 16u)) & (level - 1u);
		}

		constexpr inline uint32_t approximateMissCount() noexcept {
			return missCounts[0];
		}

		constexpr inline uint32_t resetMissCount() noexcept {
			return missCounts[0] = 0;
		}

		constexpr inline uint32_t bumpMissCount() noexcept {
			return missCounts[0] += 1;
		}
	};

	using SNode = SNode_t<K, V>;
	class FNode {
	public:
		void *frozen;
	};

	class ENode {
	public:
		ANode *parent;
		uint64_t parentpos;
		ANode *narrow;
		uint64_t hash;
		uint64_t level;
		ANode *wide;
	};

	class LNode {
	public:
		uint64_t hash = 0;
		K key;
		V value;
		LNode *next;

		constexpr LNode() = default;
		constexpr LNode(const LNode &sn) noexcept : hash(sn.hash), key(sn.key), value(sn.value), next(sn.next) {}
		explicit constexpr LNode(SNode &sn) noexcept : hash(sn.hash), key(sn.key), value(sn.value), next(nullptr) {}
		explicit constexpr LNode(SNode *sn) noexcept : hash(sn->hash), key(sn->key), value(sn->value), next(nullptr) {}
		explicit constexpr LNode(uint64_t hash, K key, V value, LNode *next) noexcept : hash(hash), key(key), value(value), next(next) {}
	};

	class XNode {
	public:
		void *parent;
		uint32_t parentpos;
		void * stale;
		uint64_t hash;
		uint32_t level;
	};

	void *FVNode = nullptr;
	void *FSNode = nullptr;

	void *cache_ptr = nullptr;
	size_t cache_size = 0;
	constexpr static size_t missCountMax = 2048;

	/// +1, because we need to fake that the root is a `ANode`
	alignas(alignment) void* root[wayness+1] = { nullptr };
	alignas(alignment) void* rawRoot = nullptr;

	inline void* createCacheArray(const uint32_t level) noexcept {
		return aligned_alloc(alignment, sizeof(void *) * (1 + (1u << level)));
	}

	constexpr inline uint64_t H(const K key) const noexcept {
		return key; // TODO
	}

	[[nodiscard]] constexpr inline uint32_t spread(const uint32_t h) const noexcept {
		return (h ^ (h >> 16u)) & 0x7fffffff;
	}

	constexpr inline ANode* createWideArray() noexcept {
		auto *ret = (ANode *)new (std::align_val_t(alignment)) ANode{};
		maskANode(ret);
		return (ANode *)ret;
	}

	constexpr inline ANNode* createNarrowArray() noexcept {
		auto *ret = (ANNode *)new (std::align_val_t(alignment)) ANNode{};
		maskANNode(ret);
		return (ANNode *)ret;
	}

	constexpr inline size_t usedLength(void *ptr) const {
		ASSERT(((uintptr_t )ptr));
		ASSERT(isNode(ptr));
		if (isANode(ptr)){
			return wayness;
		}

		if (isANNode(ptr)){
			return lW;
		}

		ASSERT(false);
	}

	bool isFrozenS(void *ptr) const {
		if (isSNode(ptr)) {
			auto *p = (SNode *)accessNode(ptr);
			void *f = READ_TXN(p);
			return isFSNode(f);
		}

		return false;
	}

	bool isFrozenA(void *ptr) const {
		if (!isFNode(ptr)) {
			return false;
		}

		auto *f = (FNode *)accessNode(ptr);
		return isANode(f->frozen);
	}

	bool isFrozenL(void *ptr) {
		if (!isFNode(ptr)) {
			return false;
		}

		auto *f = (FNode *)accessNode(ptr);
		return isLNode(f->frozen);
	}

	void decrementCount(void *array_) {
		ASSERT(isAANode(array_));
		auto *array = (ANode *)accessNode(array_);

		if (isANNode(array_)) {
			const auto count = (uint64_t)READ_AN_COUNT(array);
			const uint64_t newCount = count - 1;
			if (!CAS_AN_COUNT(array, (uintptr_t)&count, newCount)) decrementCount(array_);
			return;
		}

		const auto count = (uint64_t)READ_A_COUNT(array);
		const uint64_t newCount = count - 1;
		if (!CAS_A_COUNT(array, (uintptr_t)&count, newCount)) decrementCount(array_);
	}

	void incrementCount(void *array_) {
		ASSERT(isAANode(array_));
		auto *array = (ANode *)accessNode(array_);

		if (isANNode(array_)) {
			const auto count = (uint64_t)READ_AN_COUNT(array);
			const uint64_t newCount = count + 1;
			ASSERT(count < 4);
			if (!CAS_AN_COUNT(array, (uintptr_t)&count, newCount)) decrementCount(array_);
			return;
		}

		const auto count = (uint64_t)READ_A_COUNT(array);
		const uint64_t newCount = count + 1;
		ASSERT(count < 16);
		if (!CAS_A_COUNT(array, (uintptr_t)&count, newCount)) decrementCount(array_);
	}
public:

	constexpr CacheTrie() noexcept {
		// this is hideous: we need to mark the root node as an A node
		rawRoot = (void *)root;
		maskANode(rawRoot);

		if constexpr (useCache) {
			// cache_size = 1024;
			// cache_ptr = (void *) aligned_alloc(alignment, (sizeof(CacheNode) * cache_size));
			// memset(cache_ptr, 0, sizeof(void *) * cache_size);
		}
	}

	void inhabitCache (void *cache, void *nv, const uint64_t hash, const uint32_t cacheeLevel) {
		if constexpr (!useCache) {
			return;
		}

		if (cache == nullptr) {
			// Only create the cache if the entry is at least level 12,
			// since the expectation on the number of elements is ~80.
			// This means that we can afford to create a cache with 256 entries.
			if (cacheeLevel >= 12) {
				auto **cn = (CacheNode **)createCacheArray(8);
				cn[0] = new (std::align_val_t(alignment)) CacheNode{nullptr, 8};
				void *np = nullptr;
				CAS_CACHE(&np, cn);
				void * newCache = cache_ptr;
				inhabitCache(newCache, nv, hash, cacheeLevel);
			}
		} else {
			const uint32_t len = cache_size;
			const uint32_t cacheLevel = __builtin_ctz(len - 1);
			if (cacheeLevel == cacheLevel) {
				const uint64_t mask = len - 1 - 1;
				const uint32_t pos = 1 + (hash & mask);
				WRITE(cache, pos, nv);
			}   // else {
				// We have a cache level miss -- update statistics, and rebuild if necessary.
				// Probably not necessary here.
			    // }
		}
	}

	void sampleAndUpdateCache(void *cache, void *stats) {
		ASSERT(false);
	}

	void recordCacheMiss() {
		if constexpr (!useCache) {
			return;
		}

		if (cache_ptr == nullptr) {
			return ;
		}

		void *cache = cache_ptr;
		if (cache == nullptr) {
			auto *stats = ((CacheNode *)READ(cache, 0));
			 if (stats->approximateMissCount() > missCountMax) {
				// We must again check if the cache level is obsolete.
				// Reset the miss count.
				stats->resetMissCount();
				// Resample to find out if cache needs to be repaired.
				sampleAndUpdateCache(cache, stats);
			} else {
				stats->bumpMissCount();
			}
		}
	}

	// recursively count the number of used pointers
	void sequentialFixCount(ANode *array) {
		ASSERT(isNode(array));

    	uint32_t i = 0;
    	uint64_t count = 0;
    	while (i < array->size()) {
    		void *entry = array->at(i);
    		if (entry != nullptr) { count += 1; }
    		if (isANode(entry)) {
    		  	sequentialFixCount((ANode *)entry);
    		}
    		i += 1;
    	}

		array->size(count);
	}

	bool isCompressible(void *current) {
		if constexpr (!useCompression) {
			return false;
		}

		if constexpr (useCounters) {
			const uint32_t len = isANode(current) ? wayness : bits_log2(wayness);
			const auto count = (uint64_t)READ(current, len);
			if (count > 1) {
				return false;
			}
		}

		void *found = nullptr;
		uint32_t i = 0;

		while(i < usedLength(current)) {
			void *old = READ(current, i);
			if (old != nullptr) {
				if ((found == nullptr) && (isSNode(old))) {
					found = old;
				} else {
					return false;
				}
			}
			i += 1;
		}

		return true;
	}

	bool compressingSingleLevel(void *cache, void *current, void *parent, const uint64_t hash, const uint32_t level) {
		if (parent == nullptr) {
			return false;
		}

		if (!isCompressible(current)) {
			return false;
		}

		const uint64_t parentMask = usedLength(parent) - 1u;
		const uint32_t parentPos = (hash >> (level - 4u)) & parentMask;

		auto *xn = new (std::align_val_t(alignment)) XNode {parent, parentPos, current, hash, level};
		maskXNode(xn);
		if (CAS_(parent, parentPos, (uintptr_t)&current, xn)) {
			return completeCompression(cache, xn);
		} else {
			return false;
		}
	}

	void compressAscend(void *cache, void *current, void *parent, const uint64_t hash, const uint32_t level) {
		if (compressingSingleLevel(cache, current, parent, hash, level)) {
			// Continue compressing if possible.
			// Investigate if full ascend is feasible.
			compressDescend(rawRoot, nullptr, hash, 0);
		}
	}

	bool compressDescend(void *current, void *parent, const uint64_t hash, const uint32_t level) {
		// Dive into the cache starting from the root for the given hash,
		// and compress as much as possible.
		const uint64_t pos = (hash >> level) & (usedLength(current) - 1u);
		void *old = READ(current, pos);
		if (isANode(old)){
			if (!compressDescend(old, current, hash, level + 4)) {
				return false;
			}
		}

		// We do not care about maintaining the cache in the slow compression path,
		// so we just use the top-level cache.
		if (parent != nullptr) {
			return compressingSingleLevel(cache_ptr, current, parent, hash, level);
		}

		return false;
	}

	void *compressFrozen(void *frozen, const uint32_t level) {
		void *single;
		uint32_t i = 0;
		while(i < usedLength(frozen)) {
			void *old = READ(frozen, i);
			if (!isFVNode(old)) {
				if ((single == nullptr) && isSNode(old)) {
					// It is possible that this is the only entry in the array node.
					single = old;
				} else {
					// There are at least 2 nodes that are not FVNode.
					// Unfortunately, the node was modified before it was completely frozen.
					if (usedLength(frozen) == 16) {
						ANode *wide = createWideArray();
						sequentialTransfer(frozen, wide, level);
						sequentialFixCount(wide);
						return wide;
					} else {
						void *narrow = createNarrowArray();
						sequentialTransferNarrow(frozen, narrow, level);
						sequentialFixCount((ANode *)narrow);
						return narrow;
					}
				}
			}

			i += 1;
		}

		if (single != nullptr) {
			auto *oldsn = (SNode *) accessNode(single);
			single = new (std::align_val_t(alignment)) SNode {oldsn->hash, oldsn->key, oldsn->value, nullptr};
			maskSNode(single);
		}

		return single;
	}

	void completeExpansion(void *cache, ENode *enode_){
		ASSERT(isNode(enode_));
		auto *enode = (ENode *) accessNode(enode_);
    	void *parent = (void *)accessNode(enode->parent);
    	uint64_t parentpos = enode->parentpos;
    	uint32_t level = enode->level;

    	// First, freeze the subtree beneath the narrow node.
    	void *narrow = enode->narrow;
    	freeze(cache, narrow);

    	// Second, populate the target array, and CAS it into the parent.
    	void *wide = createWideArray();
    	sequentialTransfer(narrow, wide, level);
    	sequentialFixCount((ANode *)wide);

		void *ptr = nullptr;
    	// If this CAS fails, then somebody else already committed the wide array.
    	if (!CAS_WIDE(enode, (uintptr_t)&ptr, wide)) {
    		wide = READ_WIDE(enode);
    	}

    	// We need to write the agreed value back into the parent.
    	// If we failed, it means that somebody else succeeded.
    	// If we succeeded, then we must update the cache.
    	// Note that not all nodes will get cached from this site,
    	// because some array nodes get created outside expansion
    	// (e.g. when creating a node to resolve collisions in sequentialTransfer).
    	if (CAS_(parent, parentpos, (uintptr_t)&enode_, wide)) {
    		inhabitCache(cache, wide, enode->hash, level);
    	}
	}

	bool completeCompression(void *cache, XNode *xn_) {
		auto *xn = (XNode *) accessNode(xn_);
		void *parent_ = (void *) xn->parent;
		void *parent = (void *) accessNode(xn->parent);
		const uint64_t parentPos = xn->parentpos;
		const uint32_t level = xn->level;

		// First, freeze and compress the subtree below.
		void *stale = xn->stale;
		freeze(cache, stale);

		// Then, replace with the compressed version in the parent.
		void *compressed = compressFrozen(stale, level);
		if (CAS_(parent, parentPos, (uintptr_t)&xn_, compressed)) {
			if (compressed == nullptr) {
				decrementCount(parent_);
			}

			return (compressed == nullptr) || isSNode(compressed);
		}
		return false;
	}

	void freeze(void *cache, void *current_) {
		ASSERT(isNode(current_));

		const ANode *current = (ANode *) accessNode(current_);
		uint32_t i = 0;
		while (i < usedLength(current_)) {
			void *node = READ(current, i);
			ASSERT(isNode(node));
			if (node == nullptr) {
				//  Freeze null.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then reinspect the current slot.
				if (!CAS_(current, i, (uintptr_t)&node, FVNode)) { i -= 1; }
			} else if (isSNode(node)) {
				auto *sn = (SNode *) accessNode(node);
				void *txn = READ_TXN(sn);
				if (txn == nullptr) {
					void *ptr = nullptr;
					// Freeze single node.
					// If it fails, then either someone helped or another txn is in progress.
					// If another txn is in progress, then we must reinspect the current slot.
					if (!CAS_TXN(sn, (uintptr_t)&ptr, FSNode)) { i -= 1; }
				} else if (isFSNode(txn)) {
					// We can skip, another thread previously froze this node.
				} else {
					// Another thread is trying to replace the single node.
					// In this case, we help and retry.
					CAS_(current, i, (uintptr_t)&node, txn);
					i -= 1;
				}
			} else if (isLNode(node)) {
				// Freeze list node.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then we must reinspect the current slot.
				CAS_(current, i, (uintptr_t)&node, node);
				i -= 1;
			} else if (isANode(node) || isANNode(node)) {
				// Freeze the array node.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then reinspect the current slot.
				CAS_(current, i, (uintptr_t)&node, node);
				i -= 1;
			} else if (isFrozenL(node)) {
				// We can skip, another thread previously helped with freezing this node.
			} else if (isFNode(node)) {
				// We still need to freeze the subtree recursively.
				const FNode *fnode = (FNode *) accessNode(node);
				void *subnode = fnode->frozen;
				freeze(cache, subnode);
			} else if (isFVNode(node)) {
				// We can continue, another thread already froze this slot.
			} else if (isENode(node)) {
				// If some other txn is in progress, help complete it,
				// then restart from the current position.
				completeExpansion(cache, (ENode *)node);
				i -= 1;
			} else  if (isXNode(node)) {
				// It some other txn is in progress, help complete it,
				// then restart from the current position.
				completeCompression(cache, (XNode *)node);
				i -= 1;
			} else {
				ASSERT(false);
			}
			i += 1;
		}
	}

	void* freezeAndCompress(void *cache, void *current, const uint32_t level) {
		void *single = nullptr;
		uint32_t i = 0;
		while (i < usedLength(current)) {
			void *node = READ(current, i);
			if (node == nullptr) {
				// Freeze null.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then reinspect the current slot.
				if (!CAS_(current, i, (uintptr_t)&node, FVNode))  {
					i -= 1;
				}
			} else 

			if (isSNode(node)) {
				auto *sn = (SNode *)accessNode(node);
				void *txn = READ_TXN(sn);
				if (txn == nullptr) {
					void *ptr = nullptr;
					// Freeze single node.
					// If it fails, then either someone helped or another txn is in progress.
					// If another txn is in progress, then we must reinspect the current slot.
					if (!CAS_TXN(sn, (uintptr_t)&ptr, FSNode)) { i -= 1; }
					else {
						if (single == nullptr) {
							single = sn;
						} else {
							single = current;
						}
					}
				} else if (isFSNode(txn)) {
					// We can skip, another thread previously froze this node.
					single = current;
				} else  {
					// Another thread is trying to replace the single node.
					// In this case, we help and retry.
					single = current;
					CAS_(current, i, (uintptr_t)&node, txn);
					i -= 1;
				}
			} else

			if (isLNode(node)) {
				// Freeze list node.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then we must reinspect the current slot.
				single = current;
				auto *fnode = new (std::align_val_t(alignment)) FNode(node);
				maskFNode(fnode);
				CAS_(current, i, (uintptr_t)&node, fnode);
				i -= 1;
			} else

			if (isANode(node)) {
				// Freeze the array node.
				// If it fails, then either someone helped or another txn is in progress.
				// If another txn is in progress, then reinspect the current slot.
				single = current;
				auto *fnode = new (std::align_val_t(alignment)) FNode(node);
				maskFNode(fnode);
				CAS_(current, i, (uintptr_t)&node, fnode);
				i -= 1;
			} else

			if (isFrozenL(node)) {
				// We can skip, another thread previously helped with freezing this node.
				single = current;
			} else

			if (isFNode(node)) {
				// We still need to freeze the subtree recursively.
				single = current;
				auto *a = (FNode *)accessNode(node);
				void *subnode = a->frozen;
				freeze(cache, subnode);
			} else

			if (node == FVNode) {
				// We can continue, another thread already froze this slot.
				single = current;
			} else 

			if (isENode(node)) {
				// If some other txn is in progress, help complete it,
				// then restart from the current position.
				single = current;
				completeExpansion(cache, node);
				i -= 1;
			} else 

			if (isXNode(node)) {
				// It some other txn is in progress, help complete it,
				// then restart from the current position.
				single = current;
				completeCompression(cache, node);
				i -= 1;
			} else {
				ASSERT(false);
			}

			i += 1;
		}

		if (isSNode(single)) {
			auto *oldsn = (SNode *) accessNode(single);
			auto *single = new (std::align_val_t(alignment)) SNode{oldsn->hash, oldsn->key, oldsn->value};
			maskSNode(single);
			return single;
		} 

		 if (single != nullptr) {
			return compressFrozen(current, level);
		} 
		
		return single;
	}

	void sequentialInsert(SNode *sn_, void *wide, const uint32_t level) {
		ASSERT(isAANode(wide));
		ASSERT(isNode(sn_));
		auto *sn = (SNode *) accessNode(sn_);

		uint64_t mask = usedLength(wide) - 1;
		uint32_t pos = (sn->hash >> level) & mask;
		
		auto *w = (ANode *)accessNode(wide);
		if(w->at(pos) == nullptr) {
			w->at(pos) = sn;
		} else {
			sequentialInsert(sn, wide, level, pos);
		}
	}
	
	void sequentialInsert(SNode *sn_, void *wide_, const uint32_t level, const uint32_t pos) {
		auto *wide = (ANode *) accessNode(wide_);
		auto *sn = (SNode *) accessNode(sn_);
		ASSERT(isAANode(wide));

		void *old = wide->at(pos);
		if (isSNode(old)) {
			wide->at(pos) = newNarrowOrWideNodeUsingFreshThatsNeedsCountFix((SNode *)old, sn, level + 4);
		} else if (isANode(old)) {
			auto *oldan = (ANode *) old;
			const uint64_t npos = (sn->hash >> (level + 4)) & (usedLength(oldan) - 1);
			if (oldan->at(npos) == nullptr) {
				oldan->at(npos) = sn;
			} else if (usedLength(oldan) == 4) {
				void *an = createWideArray();
				sequentialTransfer(oldan, an, level + 4);
				wide->at(pos)= an;
				sequentialInsert(sn, wide, level, pos);
			} else {
				sequentialInsert(sn, oldan, level + 4, npos);
			}
		} else if (isLNode(old)) {
			wide->at(pos) = newListNarrowOrWideNode((LNode *)old, sn->hash, sn->key, sn->value, level + 4);
		} else {
			ASSERT(false);
		}
	}

	void sequentialTransfer(void *source_, void *wide_, const uint32_t level) {
    	uint32_t i = 0;
		ASSERT(isAANode(source_));
		ASSERT(isAANode(wide_));
		auto *source = (ANode *) accessNode(source_);
		auto *wide = (ANode *) accessNode(wide_);
		const uint64_t mask = usedLength(wide_) - 1u;
		const uint64_t len = usedLength(source_);
    	while (i < len) {
    	  void *node = source->at(i);
    	  if (isFVNode(node)) {
    	    // We can skip, the slot was empty.
    	  } else if (isFrozenS(node)) {
    	    // We can copy it over to the wide node.
    	    auto *oldsn = (SNode *) accessNode(node);
    	    auto *sn = new (std::align_val_t(alignment)) SNode(oldsn->hash, oldsn->key, oldsn->value);
    	    const uint32_t pos = (sn->hash >> level) & mask;
			maskSNode(sn);
    	    if (wide->at(pos) == nullptr) {
				wide->at(pos) = sn;
			} else { 
				sequentialInsert(sn, wide_, level, pos);
			}
    	  } else if (isFrozenL(node)) {
    	    auto *fn = (FNode *)accessNode(node);
    	    auto *tail = (LNode *)fn->frozen;
    	    while (tail != nullptr) {
    	      auto *sn = new (std::align_val_t(alignment)) SNode{tail->hash, tail->key, tail->value, nullptr};
    	      const uint32_t pos = (sn->hash >> level) & mask;
    	      sequentialInsert(sn, wide_, level, pos);
    	      tail = tail->next;
    	    }
    	  } else if (isFNode(node)) {
    	    auto *fn = (FNode *)accessNode(node);
    	    sequentialTransfer(fn->frozen, wide_, level);
    	  } else {
			  ASSERT(false);
    	  }
    	  i += 1;
    	}
	}

	void sequentialTransferNarrow(void *source_, void *narrow_, const uint32_t level) {
		// its really not used
		(void)level;
		ASSERT(isAANode(source_));
		ASSERT(isAANode(narrow_));
		auto *source = (ANode *)source_;
		auto *narrow = (ANNode *)narrow_;

		uint32_t i = 0;
		while(i < 4) {
			void *node = source->at(i);
			if (isFVNode(node)){
				// we can skip, this was empty
			} else if (isFrozenS(node)) {
				auto *oldsn = (SNode *)accessNode(node);
				auto *sn = new (std::align_val_t(alignment)) SNode(oldsn->hash, oldsn->key, oldsn->value);
				maskSNode(sn);
				narrow->at(i) = sn;
			} else if (isFrozenL(node)) {
				auto *o = (FNode *)accessNode(node);
				auto *sn = (LNode *)o->frozen;
				narrow->at(i) = sn;
			} else {
				ASSERT(false);
			}
			
			i += 1;
		}
	}

	void* newListNarrowOrWideNode(LNode *oldln, const uint64_t hash,
	                              const K k, const V v, const uint32_t level) {
		ASSERT(isNode(oldln));
		auto *tail = (LNode *) accessNode(oldln);
		LNode *ln = nullptr;
		while (tail != nullptr) {
			ln = new (std::align_val_t(alignment)) LNode{tail->hash, tail->key, tail->value, nullptr};
			tail = tail->next;
		}
		
		if (ln->hash == hash) {
			maskLNode(ln);
			auto *l = new (std::align_val_t(alignment)) LNode{hash, k, v, ln};
			maskLNode(l);
			return l;
		} else {
			ANode *an_ = createWideArray();
			auto *an = (ANode *)accessNode(an_);
			uint32_t pos1 = (ln->hash >> level) & (usedLength(an_) - 1u);
			an->at(pos1) = ln;
			auto *sn = new (std::align_val_t(alignment)) SNode{hash, k, v, nullptr};
			maskSNode(sn);
			sequentialInsert(sn, an_, level);
			sequentialFixCount(an_);
			return an_;
		}
	}

	void *newNarrowOrWideNode(const uint64_t h1, const K k1, const V v1,
											   const uint64_t h2, const K k2, const V v2,
											   const uint32_t level) {
		auto *sn1 = new (std::align_val_t(alignment)) SNode {h1, k1, v1, nullptr};
		auto *sn2 = new (std::align_val_t(alignment)) SNode {h2, k2, v2, nullptr};
		maskSNode(sn1);
		maskSNode(sn2);
		return newNarrowOrWideNodeUsingFresh(sn1, sn2, level);
	}

	void* newListNarrowOrWideNode(const uint64_t h1, const K k1, const V v1,
			const uint64_t h2, const K k2, const V v2,
			const uint32_t level) {

		auto *sn1 = new (std::align_val_t(alignment)) SNode {h1, k1, v1, nullptr};
		auto *sn2 = new (std::align_val_t(alignment)) SNode {h2, k2, v2, nullptr};
		maskSNode(sn1);
		maskSNode(sn2);
		return newNarrowOrWideNodeUsingFresh(sn1, sn2, level);
	}

	void *newNarrowOrWideNodeUsingFresh(SNode *sn1_, SNode *sn2_, const uint32_t level) {
		ASSERT(isNode(sn1_));
		ASSERT(isNode(sn2_));
		auto *sn1 = (SNode *) accessNode(sn1_);
		auto *sn2 = (SNode *) accessNode(sn2_);

		if (sn1->hash == sn2->hash) {
			auto *ln1 = new (std::align_val_t(alignment)) LNode{sn1};
			auto *ln2 = new (std::align_val_t(alignment)) LNode{sn2};
			ln2->next = ln1;
			maskLNode(ln1);
			maskLNode(ln2);

			delete sn1;
			delete sn2;
			return ln2;
		} else {
			const uint32_t pos1_ = (sn1->hash >> level) & (4 - 1);
			const uint32_t pos2_ = (sn2->hash >> level) & (4 - 1);
			if (pos1_ != pos2_) {
				ANNode *an_ = createNarrowArray(); // NOTE: already masked
				auto *an = (ANNode *) accessNode(an_);
				const uint64_t pos1 = (sn1->hash >> level) & (usedLength(an_) - 1u);
				const uint64_t pos2 = (sn2->hash >> level) & (usedLength(an_) - 1u);
				an->at(pos1) = sn1_;
				an->at(pos2) = sn2_;
				an->size(2);
				return an_;
			} else {
				ANode *an = createWideArray(); // NOTE: already masked
				sequentialInsert(sn1_, an, level);
				sequentialInsert(sn2_, an, level);
				sequentialFixCount(an);
				return an;
			}
		}
	}
	
	void *newNarrowOrWideNodeThatNeedsCountFix(const uint64_t h1, const K k1, const V v1,
			const uint64_t h2, const K k2, const V v2,
			const uint32_t level) {

		auto *sn1 = new (std::align_val_t(alignment)) SNode {h1, k1, v1};
		auto *sn2 = new (std::align_val_t(alignment)) SNode {h2, k2, v2};
		maskSNode(sn1);
		maskSNode(sn2);
		return newNarrowOrWiedeNodeUsingFreshThatsNeedsCountFix(sn1, sn2, level);
	}

	void *newNarrowOrWideNodeUsingFreshThatsNeedsCountFix(SNode *sn1_, SNode *sn2_, const uint32_t level) {
		ASSERT(isNode(sn1_));
		ASSERT(isNode(sn2_));
		auto *sn1 = (SNode *) accessNode(sn1_);
		auto *sn2 = (SNode *) accessNode(sn2_);
		if (sn1->hash == sn2->hash) {
			auto *ln1 = new (std::align_val_t(alignment)) LNode(sn1);
			auto *ln2 = new (std::align_val_t(alignment)) LNode{sn2};
			ln2->next = ln1;
			ASSERT(false);
			maskLNode(ln1);
			maskLNode(ln2);
			return ln2;
		} else {
			const uint32_t pos1_ = (sn1->hash >> level) & (4 - 1);
			const uint32_t pos2_ = (sn2->hash >> level) & (4 - 1);
			if (pos1_ != pos2_) {
				ANNode *an = createNarrowArray(); // node already masked
				const uint32_t pos1__ = (sn1->hash >> level) & (usedLength(an) - 1u);
				const uint32_t pos2__ = (sn2->hash >> level) & (usedLength(an) - 1u);
				an->at(pos1__) = sn1;
				an->at(pos2__) = sn2;
				return an;
			} else {
				ANode *an = createWideArray(); // NOTE: node already masked
				sequentialInsert(sn1, an, level);
				sequentialInsert(sn2, an, level);
				sequentialFixCount(an);
				return an;
			}
		}
	}

	LNode *newListNodeWithoutKey(void *result, LNode *oldln_, const uint64_t hash, const K k) {
		ASSERT(isNode(oldln_));
		result = nullptr;
		auto *tail = (LNode *) accessNode(oldln_);
		while (tail != nullptr) {
			if (tail->key == k) {
				// Only reallocate list if the key must be removed.
				result = (void *)tail->value;
				LNode *ln = nullptr;
				tail = (LNode *)accessNode(oldln_);
				while (tail != nullptr) {
					if (tail->key != k) {
						ln = new (std::align_val_t(alignment)) LNode(tail->hash, tail->key, tail->value, ln);
						maskLNode(ln);
					}
					tail = tail->next;
				}
	
				return ln;
			}
	
			tail = tail->next;
		}
	
		return oldln_;
	}

	V fast_lookup(const K key) noexcept {
		const uint64_t hash = H(key);
		if constexpr (!useCache) {
			return lookup(key, hash, 0, rawRoot, cache_ptr);
		}

		if (cache_ptr == nullptr) {
			return lookup(key, hash, 0, rawRoot, nullptr);
		}

		const uint32_t len = cache_size;
		const uint32_t mask = len - 1u - 1u;
		const uint32_t pos = 1u + (hash & mask);
		void *cachee = READ(cache_ptr, pos);
		const uint32_t level = 31u - __builtin_clz(len - 1);

		if (cachee == nullptr) {
			// nothing is cached at this location, do slow lookup
			return lookup(key, hash, 0, rawRoot, cache_ptr);
		} else if (isSNode(cachee)) {
			auto *oldsn = (SNode *) accessNode(cachee);
			void *txn = READ_TXN(oldsn);
			if (txn == nullptr) {
				if ((oldsn->hash == hash) && oldsn->key == key) {
					return oldsn->value;
				} else {
					return 0;// TODO
				}
			} else {
				// The single node is either frozen or scheduled for modification
				return lookup(key, hash, 0, rawRoot, cache_ptr);
			}
		} else if (isAANode(cachee)){
			auto *an = (ANode *) (accessNode(cachee));
			const uint32_t pos = (hash >> level) & mask;
			void *old = READ(an, pos);
			if (old == nullptr) {
				// the key is not present in the cache trie
				return 0;
			} else if (isSNode(cachee)) {
				auto *oldsn = (SNode *) accessNode(cachee);
				void *txn = READ_TXN(oldsn);
				if (txn == nullptr) {
					// The single node is up-to-date.
					// Check if the key is contained in the single node.
					if ((oldsn->hash == hash) && oldsn->key == key) {
						return oldsn->value;
					} else {
						return 0;// TODO probably std::make_pair<T, bool>{0, false};
					}
				} else {
					// The single node is either frozen or scheduled for modification
					return lookup(key, hash, 0, rawRoot, cache_ptr);
				}
			} else {
				// resume slow lookup from a specific level
				if (isANode(old)) {
					return lookup(key, hash, level + 4, (void *)old, cache_ptr);
				} else if (isLNode(old)) {
					// Check if the key is contained in the list node.
					auto *tail = (LNode *)accessNode(old);
					while (tail != nullptr) {
						if (tail->hash == hash && tail->key == key) {
							return tail->value;
						}

						tail = tail->next;
					}

					return 0; // TODO
				} else if (isFNode(old) || isFVNode(old)) {
					// Array node contains a frozen node, so it is obsolete -- do slow lookup.
					return lookup(key, hash, 0, rawRoot, cache_ptr);
				} else if (isENode(old)) {
					auto *en = (ENode *) accessNode(old);
					completeExpansion(cache_ptr, en);
					return fast_lookup(key);
				} else if (isXNode(old)) {
					// Help complete the transaction.
					auto *xn = (XNode *) accessNode(old);
					completeCompression(cache_ptr, xn);
					return fast_lookup(key);
				} else {
					// error
					ASSERT(false);
				} //if old
			} // if old == nullptr
		} // if cachee == nullptr

		return 0; // TODO
	}

	V lookup(const K key, const uint64_t hash, uint64_t level, void *cur_, void *cache= nullptr) noexcept {
		ASSERT(cur_ != nullptr);
		ASSERT(isNode(cur_));

		// TODO cache
		auto *cur = (uintptr_t *) accessNode(cur_);
		uint64_t mask = usedLength(cur_) - 1;
		const size_t pos = (hash >> level) & mask;
		void *old = READ(cur, pos);
		if ((old == nullptr) || isFVNode(old)) {
			return 0;// TODO not correct
		} else if (isANode(old) || isANNode(old)) {
			return lookup(key, hash, level + lW, old, cache);
		} else if (isSNode(old)) {
			// TODO
			const uint32_t cachelevel = (cache == nullptr) ? 0 : 32 - __builtin_clz(1);
			if ((level < cachelevel) || (level >= cachelevel + 8)){
				recordCacheMiss();
			}

			const SNode *oldsn = (SNode *) accessNode(old);
			if ((oldsn->key == key) && (oldsn->hash == hash)) {
				return oldsn->value;
			}

			return 0; // TODO not correct
		} else if (isLNode(old)) {
			// TODO cache

			const LNode *oldln = (LNode *)accessNode(old);
			if (oldln->hash == hash){
				auto *tail = (LNode *)oldln;
				while(tail != nullptr) {
					if ((oldln->key == key) && (oldln->hash == hash)) {
						return oldln->value;
					}

					tail = (LNode *)accessNode(tail->next);
				}
			}

			return 0; // TODO not correct
		} else if (isENode(old)) {
			const ENode *ptr = (ENode *) accessNode(old);
			return lookup(key, hash, level + 4, ptr->narrow, cache);
		} else if (isXNode(old)) {
			const XNode *ptr = (XNode *) accessNode(old);
			return lookup(key, hash, level + 4, ptr->stale, cache);
		} else if (isFNode(old)) {
			const FNode *ptr = (FNode *) accessNode(old);
			void *frozen = ptr->frozen;
			if (isSNode(frozen)) {
				// Unexpected case (should never be frozen):
				ASSERT(false);
			} else if (isLNode(frozen)) {
				auto *ln = (LNode *) accessNode(frozen);
				if (ln->hash != hash) {
					return 0; // TODO
				} else {
					LNode *tail = ln;
					while (tail != nullptr) {
						if ((tail->key == key) || (tail->key == key)) {
							return tail->value;
						}

						tail = (LNode *)accessNode(tail->next);
					}

					return 0; // TODO
				}
			} else if (isANode(frozen)) {
				lookup(key, hash, level + 4, frozen, cache);
			} else {
				ASSERT(false);
			}
			return lookup(key, hash, level + lW, ptr->frozen);
		} else {
			ASSERT(false);
		}
	}

	V lookup(const K key) noexcept {
		return lookup(key, H(key), 0, rawRoot, cache_ptr);
	}

	///
	void fast_insert(const K key, const V value, const uint64_t hash) {
		fast_insert(key, value, hash, cache_ptr, cache_ptr);
	}

	void fast_insert(const K key, const V value, const uint64_t hash, void *cache, void *prevCache) {
		if constexpr(!useCache) {
			insert(key, value);
		}

		if (cache == nullptr) {
			return insert(key, value);
		}

		const uint32_t len = cache_size;
		const uint32_t mask = len - 1u - 1u;
		const uint32_t pos = 1u + (hash & mask);
		void *cachee = READ(cache, pos);
		const uint32_t level = 31u - __builtin_clz(len - 1);
		if (cachee == nullptr) {
			// Inconclusive -- retry one cache layer above.
			auto *stats = (CacheNode *)READ(cache, 0);
			auto *parentCache = stats->parent;
			fast_insert(key, value, hash, parentCache, cache);
		} else if (isAANode(cachee)) {
			// Read from the array node.
			void *an = (void *) accessNode(cachee);
			const uint64_t mask = usedLength(cachee) - 1u;
			const uint32_t pos = (hash >> level) & mask;
			void *old = READ(an, pos);
			if (old == nullptr) {
				// Try to write the single node directly.
				auto *sn = new (std::align_val_t(alignment))SNode{hash, key, value, nullptr};
				if (CAS_(an, pos, old, sn)) {
					incrementCount(an);
					return;
				} else {
					fast_insert(key, value, hash, cache, prevCache);
				}
			} else if (isAANode(old)) {
				// Continue search recursively.
				const V res = insert(key, value, hash, level + 4, old, an, prevCache);
				if (res == 0) {
					fast_insert(key, value, hash, cache, prevCache);
				}
			} else if (isSNode(old)) {
				auto *oldsn = (SNode *) accessNode(old);
				void *txn = READ_TXN(oldsn);
				if (txn == nullptr) {
					// No other transaction in progress.
					if ((oldsn->hash == hash) && ((oldsn->key == key))) {
						// Replace this key in the parent.
						auto *sn = new (std::align_val_t(alignment)) SNode{hash, key, value, nullptr};
						void *np = nullptr;
						if (CAS_TXN(oldsn, &np, sn)) {
							CAS_(an, pos, oldsn, sn);
							// Note: must not increment the count here.
						} else {
							fast_insert(key, value, hash, cache, prevCache);
						}
					} else if (usedLength(an) == 4) {
						// Must expand, but cannot do so without the parent.
						// Retry one cache level above.
						auto *stats = (CacheNode *)READ(cache, 0);
						void* parentCache = stats->parent;
						fast_insert(key, value, hash, parentCache, cache);
					} else {
						// Create an array node at the next level and replace the single node.
						auto *nnode = newNarrowOrWideNode(oldsn->hash, oldsn->key, oldsn->value, hash, key, value, level + 4);
						void *np = nullptr;
						if (CAS_TXN(oldsn, &np, nnode)) {
							CAS_(an, pos, oldsn, nnode);
						} else {
							fast_insert(key, value, hash, cache, prevCache);
						}
					}
				} else if (txn == FSNode) { // TODO fsnode = NoTxn = nullpt
					// Must restart from the root, to find the transaction node, and help.
					insert(key, value);
				} else {
					// Complete the current transaction, and retry.
					CAS_(an, pos, old, txn);
					fast_insert(key, value, hash, cache, prevCache);
				}
			} else {
				// Must restart from the root, to find the transaction node, and help.
				insert(key, value);
			}
		} else if (isSNode(cachee)) {
			// Need a reference to the parent array node -- retry one cache level above.
			auto *stats = (CacheNode *)READ(cache, 0);
			void *parentCache = stats->parent;
			fast_insert(key, value, hash, parentCache, cache);
		} else {
			// sys.error(s"Unexpected case -- $cachee is not supposed to be cached.")
			ASSERT(false);
		}
	}

	bool insert(const K key, const V value, const uint64_t hash,
	            const uint64_t level, void *cur_, void *prev_,
	            void *cache= nullptr){
		if ((cache != nullptr) && ((1 << level) == (cache_size - 1))) {
			inhabitCache(cache, cur_, hash, level);
		}

		ASSERT(cur_ != nullptr);
		ASSERT(isNode(cur_));
		void *cur = (uintptr_t *) accessNode(cur_);
		const uint64_t mask = usedLength(cur_) - 1u;
		const size_t pos = (hash >> level) & mask;
		void *old = READ(cur, pos);

		insert_start:
		ASSERT(isNode(old) || (old == nullptr));

		if (old == nullptr) {
			// Fast-path -- CAS the node into the empty position.
			const uint32_t cacheLevel = cache == nullptr ? 0 : __builtin_clz(cache_size - 1u);
			if (level < cacheLevel || level >= cacheLevel + 8) {
				recordCacheMiss();
			}

			auto sn = new (std::align_val_t(alignment)) SNode{hash, key, value, nullptr};
			maskSNode(sn);

			if (CAS_(cur, pos, (uintptr_t)&old, sn)) {
				auto *k = (ANode *)cur;
				incrementCount(cur_);
				return true;
			}

			return insert(key, value, hash, level, cur_, prev_, cache);
		} else if (isANode(old) || isAANode(old)) {
			// repeat the search on the next level
			return insert(key, value, hash, level + lW, old, cur_, cache);
		} else if (isSNode(old)) {
			const SNode *o = (SNode *) accessNode(old);
			void *txn = READ_TXN(o);
			if (txn == nullptr){
				if (o->key == key){
					auto sn = new (std::align_val_t(alignment)) SNode{hash, key, value, nullptr};
					maskSNode(sn);

					const uintptr_t *ptr = nullptr;
					if (CAS_TXN(o, &ptr, sn)) {
						CAS_(cur, pos, (uintptr_t)&old, sn);
						return true;
					}

					return insert(key, value, hash, level, cur_, prev_);
				} else if (isANNode(cur_)) { // if narrow node
					ASSERT(level);

					const uint32_t prev_length = usedLength(prev_);
					const uint64_t pmask = prev_length - 1u;
					const uint32_t ppos = (hash >> (level - 4u)) & pmask;
					auto en = new (std::align_val_t(alignment)) ENode{(ANode *)prev_, ppos, (ANode *)cur_, hash, level, nullptr};
					ENode *en2 = en;
					maskENode(en);

					auto *tmp_parent = (ANode *) accessNode(prev_);
					if(CAS_(tmp_parent, ppos, (uintptr_t)&cur_, en)) {
						completeExpansion(cache, en);
						const auto wide = READ_WIDE(en2);
						return insert(key, value, hash, level, (void *)wide, prev_, cache);
					} else {
						return insert(key, value, hash, level, cur_, prev_, cache);
					}
				} else {
					// Replace the single node with a narrow node.
					void *nnode = newNarrowOrWideNode(o->hash, o->key, o->value,
					                                hash, key, value, level + 4);
					const uintptr_t *ptr = nullptr;
					if (CAS_TXN(o, &ptr, nnode)) {
						CAS_(cur, pos, (uintptr_t)&old, nnode);
						return true;
					}

					return insert(key, value, hash, level, cur_, prev_, cache);

				} // old->key == k
			} // txn == 0
			else if (txn == FSNode) {
				// We landed into the middle of another transaction.
				// We must restart from the top, find the transaction node and help.
				goto insert_start;
			} else {
				// The single node had been scheduled for replacement by some thread.
				// We need to help, then retry.
				CAS_(cur, pos, (uintptr_t)&old, txn);
				return insert(key, value, hash, level, cur_, prev_, cache);
			}
		} else if (isLNode(old)) {
			void *nn = newListNarrowOrWideNode((LNode *)old, hash, key, value, level + lW);
			if (CAS_(cur, pos, (uintptr_t)&old, nn)) { return true;}
			return insert(key, value, hash, level, cur_, prev_, cache);
		} else if (isENode(old)) {
			completeExpansion(cache, (ENode *)old);
			// goto insert_start;
			return false;
		} else if (isXNode(old)) {
			completeCompression(cache, (XNode *)old);
			// goto insert_start;
			return false;
		} else {
			ASSERT(false);
		}

		return false;
	}

	void insert(const K key, const V value) {
		if (!insert(key, value, H(value), 0, rawRoot, nullptr)) {
			insert(key, value);
		}
	}

	void* remove(const K key,
	             const uint64_t hash,
	             const uint32_t level,
	             void *current_,
	             void *parent_, void *cache) {
		const uint64_t mask = usedLength(current_) - 1;
		const uint32_t pos = (hash >> level) & mask;
		void *current = (void *) accessNode(current_);
		void *old = READ(current, pos);
		if (old == nullptr) {
			// the key does not exist
			return nullptr;
		}else if (isANode(old)) {
			return remove(key, hash, level + 4, old, current, cache);
		} else if (isSNode(old)) {
			const uint32_t cachelevel = cache == nullptr ? 0 : 31 - __builtin_clz(cache_size - 1u);
			if ((level < cachelevel) || level >= cachelevel + 8) { recordCacheMiss(); }

			auto *oldsn = (SNode *) accessNode(old);
			void *txn = READ_TXN(oldsn);
			if (txn == nullptr) {
				// There is no other transaction in progress.
				if ((oldsn->hash == hash) && (oldsn->key == key)) {
					// The same key, remove it.
					const uintptr_t *ptr = nullptr;
					if (CAS_TXN(oldsn, &ptr, nullptr)) {
						CAS_(current, pos, (uintptr_t)&oldsn, nullptr);
						decrementCount(current_);
						compressAscend(cache, current_, parent_, hash, level);
						return oldsn;
					} else {
						return remove(key, hash, level, current_, parent_, cache);
					}
				} else {
					// The target key does not exist.
					return nullptr;
				}
			} else if (isFSNode(txn)) {
				// We landed into a middle of another transaction.
				// We must restart from the top, find the transaction node and help.
				ASSERT(false);
			} else {
				// The single node had been scheduled for replacement by some thread.
				// We need to help and retry.
				CAS_(current, pos, (uintptr_t)&oldsn, txn);
				return remove(key, hash, level, current_, parent_, cache);
			}
		} else if (isLNode(old)) {
			auto *oldln = (LNode *) accessNode(old);
			void *nn = nullptr;
			void *result = newListNodeWithoutKey(nn, oldln, hash, key);
			if (CAS_(current, pos, (uintptr_t)&oldln, nn)) {
				return result;
			} else {
				return remove(key, hash, level, current_, parent_, cache);
			}
		} else if (isENode(old)) {
			completeExpansion(cache, (ENode *)old);
			return remove(key, hash, level, current_, parent_, cache);
		} else if (isXNode(old)) {
			completeCompression(cache, (XNode *)old);
			return remove(key, hash, level, current_, parent_, cache);
		} else if (isFVNode(old) || isFNode(old)) {
			return remove(key, hash, level, current_, parent_, cache);
		}

		return nullptr;
	}

	void* remove(const K key) {
		void *val = nullptr;
		while (val == nullptr) {
			val = remove(key, H(key), 0, rawRoot, nullptr, cache_ptr);
		}

		return val;
	}
};



#undef isFVNode
#undef isSNode
#undef isENode
#undef isFNode
#undef isANode
#undef isANNode
#undef maskFVNode
#undef maskSNode
#undef maskENode
#undef maskFNode
#undef maskANode
#undef maskANNode
#undef accessNode
#endif//CRYPTANALYSISLIB_CTRIE_H
