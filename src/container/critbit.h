#ifndef CRYPTANALYSISLIV_CONTAINER_CRITBIT_H
#define CRYPTANALYSISLIV_CONTAINER_CRITBIT_H

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>

/// Source
/// https://github.com/glk/critbit 
// struct critbit_key;
// critbit_ref; //  = T = thing to store

// TODO: the insertion is not working for the first element, where only the key is inserted into the tree, but not the node.


/// TODO move somewhere useful
/// \returns the msb
constexpr static inline uint8_t ms1b8(uint8_t v) noexcept {
	uint32_t value = v;
	return value ? 1 << (31 - __builtin_clz(value)) : 0;
}

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

	// TODO static () Operator
    KeyCompare keycmp = KeyCompare();
    KeyLen keylen = KeyLen(); 
	Hash hash{};
	KeyBinary keybinary{};

	struct critbit_key_ref_tuple {
		critbit_key key;
		critbit_ref ref;
	};

public:
    struct critbit_node {
    	critbit_ref *child[2];
    	uint32_t byte;
    	uint8_t otherbits;
    };
    
private: 
    /// TODO move somewhere usefull
    static inline void
    critbit_node_free(critbit_node *node) noexcept {
    	// assert(t->node_free != NULL);
    	// t->node_free(t, node);
        free(node);
    }

public:
    /// TODO doc
    critbit_tree() noexcept : free_arg(nullptr) {
		root = nullptr;
	}

private:
    // /// TODO doc
    // inline size_t critbit_buf_keylen(const uint8_t *a) noexcept {
    //     (void)a;
    // 	return keylen;
    // }
   
    // /// TODO doc
    // inline size_t critbit_str_keylen(const uint8_t *a) noexcept {
    // 	return (strlen((char *)a));
    // }
  

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

    constexpr static inline const uint8_t *
    critbit_buf_keybuf(const critbit_key *key) noexcept {
    	return ((const uint8_t *)(key));
    }
    
    constexpr static inline const uint8_t *
    critbit_str_keybuf(const critbit_key *key) noexcept {
    	return (*(uint8_t **)(key));
    }
   
    /// TODO abstract away as class template function
    constexpr static inline 
    int critbit_str_keycmp(const uint8_t *a, 
                           const uint8_t *b, 
                           const size_t blen) noexcept {
    	size_t alen;
    	int rv;
    
    	alen = strlen((char *)a);
    	rv = alen - blen;
    	if (rv != 0)
    		return (rv);
    	return (memcmp(a, b, blen));
    }
    
    constexpr static inline
    int critbit_buf_keycmp(const uint8_t *a,
                           const uint8_t *b,
                           size_t blen) noexcept {
    	return (memcmp(a, b, blen));
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

	/// TODO returns?
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
    		critbit_node_free(newnode);
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
    
    	critbit_node_free(newnode);
    	return (critbit_ref_get_key(p));
    
    different_byte_found:
    
    	newotherbits = ms1b8(newotherbits) ^ 255;
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

    	/// TODO simplify `keybinary` is not needed
    	if (memcmp(keybinary(critbit_ref_get_key(p)), ubytes, keyLen) != 0) {
    		return nullptr;
    	}

    	// Remove p
    	if (whereq == nullptr) {
    		this->root = nullptr;
    		return (critbit_ref_get_key(p));
    	}

    	assert(q);
    	*whereq = (critbit_ref *)(q->child[1 - direction]);
    	critbit_node_free(q);

    	return critbit_ref_get_key(p);
    }
};

#if 0  // TODO
static void
traverse(void *top)
{
	uint8_t *p = top;

	if (1 & (intptr_t)p) {
		critbit_node *q = (void *)(p - 1);
		traverse(q->child[0]);
		traverse(q->child[1]);
		free(q);
	} else {
		free(p);
	}

}

void
critbit0_clear(critbit_tree *t)
{
	if (t->root)
		traverse(t->root);
	t->root = NULL;
}

static int
allprefixed_traverse(uint8_t *top,
    int(*handle)(const char *, void *), void *arg)
{
	if (1 & (intptr_t)top) {
		critbit_node *q = (void *)(top - 1);
		for (int direction = 0; direction < 2; ++direction)
			switch (allprefixed_traverse(q->child[direction], handle, arg)) {
			case 1:
				break;
			case 0:
				return 0;
			default:
				return -1;
			}
		return 1;
	}

	return handle((const char *)top, arg);
}

int
critbit0_allprefixed(critbit_tree *t, const char *prefix,
    int(*handle)(const char *, void *), void *arg)
{
	const uint8_t *ubytes = (const void *)prefix;
	const size_t keylen = strlen(prefix);
	critbit_node *q;
	uint8_t *p = t->root;
	uint8_t *top = p;

	if (p == NULL)
		return 1;

	while (1 & (intptr_t)p) {
		critbit_node *q = (void *)(p - 1);
		uint8_t c = 0;
		if (q->byte < keylen)
			c = ubytes[q->byte];
		const int direction = (1 + (q->otherbits | c)) >> 8;
		p = q->child[direction];
		if (q->byte < keylen)
			top = p;
	}

	for (size_t i = 0; i < keylen; ++i) {
		if (p[i] != ubytes[i])
			return 1;
	}


	return allprefixed_traverse(top, handle, arg);
}
#endif // end 0



#endif
