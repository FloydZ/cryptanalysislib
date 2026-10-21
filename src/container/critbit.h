#ifndef CRYPTANALYSISLIV_CONTAINER_CRITBIT_H
#define CRYPTANALYSISLIV_CONTAINER_CRITBIT_H

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>

#include "algorithm/countl.h"

/// Source
/// https://github.com/glk/critbit 
// struct critbit_key;
// critbit_ref; //  = T = thing to store

// NOTE: the first insertion stores the reference itself as the root leaf
// 	(no node is needed). Leaves are the user's `critbit_ref` objects (not
// 	owned by the tree), internal nodes are `malloc`ed by the caller and
// 	`free`d by the tree (remove, clear, destructor).
// NOTE: the functors `Hash`, `KeyCompare`, `KeyLen`, `KeyBinary` are not
// 	stored, every call constructs a temporary (so they can also have a
// 	static `operator()`).

template<typename critbit_ref,
         typename critbit_key,
         typename Hash,
         typename KeyCompare,
         typename KeyLen,
         typename KeyBinary>
class critbit_tree {
private:
    
    typedef const uint8_t *critbit_keybuf_t(const critbit_key *key);

	void *root;
	void *free_arg;

	struct critbit_key_ref_tuple {
		critbit_key key;
		critbit_ref ref;
	};

	[[nodiscard]] constexpr static inline int keycmp(const critbit_key *a, const critbit_key *b) noexcept {
		return KeyCompare{}(a, b);
	}
	[[nodiscard]] constexpr static inline size_t keylen(const critbit_key *a) noexcept {
		return KeyLen{}(a);
	}
	[[nodiscard]] constexpr static inline const critbit_key *hash(const critbit_ref *a) noexcept {
		return Hash{}(a);
	}
	[[nodiscard]] constexpr static inline const uint8_t *keybinary(const critbit_key *a) noexcept {
		return KeyBinary{}(a);
	}

public:
    struct critbit_node {
    	critbit_ref *child[2];
    	uint32_t byte;
    	uint8_t otherbits;
    };

public:
    /// creates an empty tree
    critbit_tree() noexcept : free_arg(nullptr) {
		root = nullptr;
	}

    critbit_tree(const critbit_tree &) = delete;
    critbit_tree &operator=(const critbit_tree &) = delete;

    /// frees all internal nodes (the references are not owned)
    ~critbit_tree() noexcept {
        clear();
    }

    /// removes all elements: frees the internal nodes, the references
    /// (leaves) are owned by the caller
    void clear() noexcept {
        clear(root);
        root = nullptr;
    }

    /// calls `handle(ref)` for every stored reference whose key starts with
    /// the `prefix_len` bytes of `prefix`
    /// \return false if `handle` returned false (and stopped the traversal)
    template<typename F>
    bool allprefixed(const uint8_t *prefix,
                     const size_t prefix_len,
                     F &&handle) noexcept {
        critbit_ref *p = (critbit_ref *)this->root;
        critbit_ref *top = p;
        if (p == nullptr) { return true; }

        while (critbit_ref_is_internal(p)) {
            critbit_node *q = critbit_ref_get_node(p);
            uint8_t c = 0;
            if (q->byte < prefix_len) {
                c = prefix[q->byte];
            }
            const int direction = (1 + (q->otherbits | c)) >> 8;
            p = q->child[direction];
            if (q->byte < prefix_len) {
                top = p;
            }
        }

        // the best match must have the prefix, else no element has it
        const critbit_key *k = critbit_ref_get_key(p);
        if (keylen(k) < prefix_len) { return true; }
        const uint8_t *kb = keybinary(k);
        for (size_t i = 0; i < prefix_len; ++i) {
            if (kb[i] != prefix[i]) { return true; }
        }
        return allprefixed_traverse(top, handle);
    }

private:
    /// frees all internal nodes below `top`
    static void clear(void *top) noexcept {
        critbit_ref *p = (critbit_ref *)top;
        if ((p == nullptr) || !critbit_ref_is_internal(p)) { return; }
        critbit_node *q = critbit_ref_get_node(p);
        clear(q->child[0]);
        clear(q->child[1]);
        free(q);
    }

    template<typename F>
    static bool allprefixed_traverse(critbit_ref *top, F &handle) noexcept {
        if (critbit_ref_is_internal(top)) {
            critbit_node *q = critbit_ref_get_node(top);
            for (int direction = 0; direction < 2; ++direction) {
                if (!allprefixed_traverse(q->child[direction], handle)) {
                    return false;
                }
            }
            return true;
        }
        return handle((const critbit_ref *)top);
    }

    constexpr static inline
    int critbit_ref_is_internal(const critbit_ref *ref) noexcept {
    	return (((intptr_t)ref) & 1);
    }
   

    constexpr static inline 
    critbit_node * critbit_ref_get_node(critbit_ref *ref) noexcept {
    	assert(critbit_ref_is_internal(ref));
    	return ((critbit_node *)(void *)(((uint8_t *)ref) - 1));
    }
    
    constexpr static inline
    void critbit_ref_set_node(critbit_ref **ref,
                              critbit_node *node) noexcept {
    	*ref = (critbit_ref *)(((uint8_t *)node) + 1);
    	assert(critbit_ref_is_internal(*ref));
    }
    
    constexpr static inline
    void critbit_ref_set_key(critbit_ref **ref,
                             const critbit_key *key) noexcept {
    	*ref = (critbit_ref *)key;
    	assert(!critbit_ref_is_internal(*ref));
    }
    
    constexpr inline
    critbit_key * critbit_ref_get_key(const critbit_ref *ref) const noexcept {
    	const critbit_key *key = hash(ref);
    	assert(!critbit_ref_is_internal(ref));
    	return (critbit_key *)key;
    }

public:

    inline critbit_ref *
    critbit_get_impl(const critbit_key *key) noexcept {
        const size_t keyLen = keylen(key);
    	const uint8_t *ubytes = (uint8_t *)keybinary(key);
    	critbit_ref *ref = (critbit_ref *)this->root;
    	uint8_t c = 0;;
    
    	if (ref == nullptr) {
    		return nullptr;
        }
    
    	while (critbit_ref_is_internal(ref)) {
    		critbit_node *node = critbit_ref_get_node(ref);
    
    		c = 0;
    		if (node->byte < keyLen) {
    			c = ubytes[node->byte];
            }
    
    		const int direction = (1 + (node->otherbits | c)) >> 8;
    		ref = (critbit_ref *)node->child[direction];
    	}
    
    	if (keycmp(critbit_ref_get_key(ref), key) == 0) {
    		// return critbit_ref_get_key(ref);
    		return ref;
        }
    
    	return nullptr;
    }

    /// inserts the reference `ref` (its key is `Hash(ref)`)
    /// \param newnode[in]: an internal node allocated with `malloc`/`calloc`,
    /// 	which the tree takes over (and frees if it is not needed)
    /// \param ref[in]: reference to insert, owned by the caller
    /// \return nullptr if inserted, else the key of the already stored
    /// 	reference with the same key (nothing is inserted)
    inline critbit_key *
    critbit_insert_impl(critbit_node *newnode,
    				    const critbit_ref *ref) noexcept {
    	const critbit_key *key = hash(ref);
    	// const uint8_t *const ubytes = (const uint8_t *const)&key;
    	const uint8_t *const ubytes = keybinary(key);
        const size_t keyLen = keylen(key);
    	critbit_ref *p = (critbit_ref *)this->root;
    	const uint8_t *pkey;
    	uint32_t newbyte;
    	uint32_t newotherbits;
    	uint8_t c;
    
    	if (p == nullptr) {
            // case where the first element is inserted
    		critbit_ref_set_key((critbit_ref **)&this->root, (const critbit_key *)ref);
    		free(newnode);
    		return nullptr;
    	}
    
    	while (critbit_ref_is_internal(p)) {
    		critbit_node *q = critbit_ref_get_node(p);

    		c = 0;
    		if (q->byte < keyLen) {
    			c = ubytes[q->byte];
            }
    
    		const int direction = (1 + (q->otherbits | c)) >> 8;
    		p = (critbit_ref *)q->child[direction];
    	}
    	// NOTE: from this point on `p` is a pointer `T`
    
    	// pkey = (const uint8_t *)(critbit_ref_get_key(p));
    	pkey = keybinary(critbit_ref_get_key(p));
    	for (newbyte = 0; newbyte < keyLen; ++newbyte) {
    		if (pkey[newbyte] != ubytes[newbyte]) {
    			newotherbits = pkey[newbyte] ^ ubytes[newbyte];
    			goto different_byte_found;
    		}
    	}
    
    	if (pkey[newbyte] != 0) {
    		newotherbits = pkey[newbyte];
    		goto different_byte_found;
    	}
    
    	free(newnode);
    	return (critbit_ref_get_key(p));
    
    different_byte_found:
    
    	newotherbits = cryptanalysislib::bit_floor(uint8_t(newotherbits)) ^ 255;
    	const uint32_t newdirection = (1 + (newotherbits | pkey[newbyte])) >> 8;
    
    	newnode->byte = newbyte;
    	newnode->otherbits = newotherbits;
    	critbit_ref_set_key((critbit_ref **)&newnode->child[1 - newdirection], (critbit_key *)ref);
    
    	critbit_ref **wherep = (critbit_ref **)&this->root;
    	for (;;) {
    		p = *wherep;
    		if (!critbit_ref_is_internal(p))
    			break;
    		critbit_node *q = critbit_ref_get_node(p);
    		if (q->byte > newbyte)
    			break;
    		if (q->byte == newbyte && q->otherbits > newotherbits)
    			break;
    		c = 0;
    		if (q->byte < keyLen)
    			c = ubytes[q->byte];
    		const int direction = (1 + (q->otherbits | c)) >> 8;
    		wherep = (critbit_ref **)q->child + direction;
    	}
    
    	newnode->child[newdirection] = *wherep;
    	critbit_ref_set_node(wherep, newnode);
    	return nullptr;
    }
    
public:
    inline
    critbit_key *critbit_remove_impl(const critbit_key *key) noexcept {
    	const uint8_t *ubytes = (const uint8_t *)keybinary(key);
    	const size_t keyLen = keylen(key);
    	critbit_ref *p = (critbit_ref *)this->root;
    	critbit_node *q = nullptr;
    	critbit_ref **wherep = (critbit_ref **)&this->root;
    	critbit_ref **whereq = nullptr;
    	int direction = 0;

    	if (p == nullptr) {
    		return nullptr;
    	}

    	while (critbit_ref_is_internal(p)) {
    		whereq = wherep;
    		q = critbit_ref_get_node(p);
    		uint8_t c = 0;
    		if (q->byte < keyLen) {
    			c = ubytes[q->byte];
    		}
    		direction = (1 + (q->otherbits | c)) >> 8;
    		wherep = (critbit_ref **)q->child + direction;
    		p = *wherep;
    	}

    	// NOTE: full key comparison (as in `critbit_get_impl`), a `memcmp` of
    	// 	`keyLen` bytes also matched every stored key with prefix `key`
    	if (keycmp(critbit_ref_get_key(p), key) != 0) {
    		return nullptr;
    	}

    	// Remove p
    	if (whereq == nullptr) {
    		this->root = nullptr;
    		return (critbit_ref_get_key(p));
    	}

    	assert(q);
    	*whereq = (critbit_ref *)(q->child[1 - direction]);
    	free(q);

    	return critbit_ref_get_key(p);
    }
};




#endif
