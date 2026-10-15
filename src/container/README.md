Overview
========

| header | type | description |
|---|---|---|
| `aa_tree.h` | `AATreeSet<T>` | ordered set, AA tree (balanced binary search tree) |
| `array.h` | `const_array<T, N>` | `constexpr` array |
| `avl_tree.h` | `AvlTreeList<T>` | list with O(log n) index access/insert/remove (AVL tree) |
| `binary_heap.h` | | forwards to `heap.h` |
| `binary_indexed_tree.h` | `BIT`, `rangeBIT` | Fenwick tree: prefix sums, range updates |
| `binary_packed_vector.h` | `FqPackedVector<n, 2, T>` | bit vector of `n` bits |
| `bk_tree.h` | `BKTree` | Burkhard-Keller tree, nearest neighbours in a metric (e.g. Hamming) |
| `bplus_tree.h` | `bplus_tree<Key, Value, Order>` | in-memory B+ tree map, linked leaves |
| `btree_set.h` | `BTreeSet<T>` | ordered set, B-tree |
| `cartesian_tree.h` | `cartesian_tree<T>` | treap: ordered set with order statistics (`kth`) |
| `critbit.h` | `critbit_tree<...>` | crit-bit tree (binary radix trie) over byte strings |
| `ctrie.h` | `CacheTrie` | concurrent lock-free hash trie |
| `dancing_links.h` | `dancing_links` | Knuth's algorithm X, exact cover |
| `deque.h` | `Deque<T>` | double ended queue |
| `eer_tree.h` | `eer_tree<T>` | palindromic tree: distinct palindromic substrings, online |
| `fq_packed_vector.h` | `FqPackedVector<n, q, T>` | vector `mod q`, several elements per limb |
| `fq_vector.h` | `FqNonPackedVector<n, q, T>` | vector `mod q`, one element per limb |
| `hashmap.h` | | the hashmaps, see [here](./hashmap/README.md) |
| `heap.h` | `Heap<T>`, `Heap2<T>` | binary heap with fixed capacity, indexed binary heap (`update_key`) |
| `imap.h` | `imap_tree_t` | integer map (radix tree over 32 bit keys) |
| `kAry_type.h` | `FqElement<q>` | element `mod q` |
| `kd_tree.h` | `kd_tree<...>` | k-d tree, nearest neighbour search (euclidean/manhattan) |
| `left_right_array.h` | `left_right_array` | set/free indices of [0, n): k-th free/set index in O(log n) |
| `linkedlist.h` | | the linked lists, see [here](./linkedlist/README.md) |
| `misof_tree.h` | `misof_tree` | multiset of 15 bit integers: insert/erase/n-th smallest |
| `priorityqueue.h` | `priority_queue<T>` | priority queue |
| `queue.h` | `concurrent_queue`, `thread_safe_queue` | queues for threads, see also `queue/spsc_fixed_queue.h` |
| `rb_tree.h` | | red-black tree (unfinished port) |
| `ringbuffer.h` | `RingBuffer<T>` | ring buffer |
| `segment_tree.h` | `SegmentTree`, `lazy_segment_tree` | range queries/updates |
| `signal_tree/` | `tree<N>` | signal tree: counters for task schedulers |
| `sparse_table.h` | `sparse_table` | O(1) range minimum queries |
| `stack.h` | `Stack<T>` | stack |
| `trie.h` | `trie` | prefix tree |
| `triple.h` | | triple (3-tuple) |
| `vector.h` | | vector with an allocator for consecutive pages |
| `vector_queue.h` | `ConstVectorQueue` | queue on a fixed size vector |

Number containers
=================

Each of the number containers implement the following functions. The variable 
`i` represents a coordinate index within the container. Thus `c.is_zero(i, i)`
means than a function that checks if the container is zero between a lower and 
an upper limit is implemented.

```C++
# Access operators
c[i];
c.get(i);
c.set(i, i);

#  rng setter
c.rng();
c.zero();

# get a pointer to the underlying raw array
c.ptr();

# or just a limb of the arrau
c.ptr(s);

# all needed comparison operators
c.is_equal(c, i, i);
c.is_equal(c);
c.is_greater(c, i, i);
c.is_greater(c);
c.is_lower(c, i, i);
c.is_lower(c);
c.is_zero(i, i);
c.is_zero();
c.neg(i, i);

# printing stuff
c.print_binary(i, i);
c.print(i, i);

# arithmetic on the full length
Container::add(c, c, c, i, i);
Container::sub(c, c, c, i, i);
Container::set(c, c, i, i);
Container::cmp(c, c, i, i);

# arithmetic only on a limb
Container::add_T(a, a);
Container::sub_T(a, a);
Container::mod_T(a);
Container::mul_T(a, a);
Container::neg_T(a);
Container::scalar_T(a, a);
Container::popcnt_T(a);

# arithmetic on a limb with 256 bits
Container::add256_T(b, b);
Container::sub256_T(b, b);
Container::mod256_T(b);
Container::mul256_T(b, b);
Container::neg256_T(b);
```

### BinaryContainer_T<T, n>
Container holding `n` bits in limbs of type `T`. Each limb will hold 
`sizeof(T)*8` bits.

### kAryType<T, T2, n>
represents a value `mod q`. The second type `T2` is needed to sanely implement 
the multiplication.

### FqNonPackedVector<T, n>
holds `len` elements `mod q` and each element is  saved in its own limb of 
type `T`. 

### FqPackedVector<T, n>
same as `kAryContainer<T, len>` but the implementations stores as much as 
possible elements `mod q` in one limb of type `T`.

Array:
======

A custom `constexpr` array implementation is provided. Not really tested

Vector:
=======
A "custom" `vector` class is provided. This differs only in the memory 
allocator its using. This allocator only allocates consecutive memory pages.

Queue:
======
A (non-lock-free) queue is provided

LinkedList:
More on the different implemented `linkedlists` is [here](./linkedlist/README.md)

HashMaps:
More on the different implemented `HashMaps` is [here](./hashmap/README.md)
