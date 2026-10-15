General:
========

All hashmaps implemented have the following features:
- const size. You need to specify the size within the object instantiation, and afterward it's impossible to change the 
  size.
- No deletion. Once an element is inserted, you cannot delete it. There is only a `clear()` function which clears the 
  whole hashmap.


Simple_Hashmap:
===============
As the name suggests, this is a rather simple hashmap, with only limited features. Well basically only a single feature,
it can handle multithreaded inserts. 

Usage:
-----

```C
using K = uint32_t; // just an example, can be whatever you want
using V = uint64_t; // just an example, can be whatever you want

// needed hashfunction, which only returns the lower `l` bits
template<const uint32_t l>
size_t H(const K k) {
	constexpr K mask = (1u << l) - 1u;
	return k & mask;
}

// number of bits to match on
constexpr uint32_t l = 12;

// max number of elements in each bucket
constexpr uint32_t bucketsize = 10;

// total number of threas
constexpr uint32_t threads = 2;
constexpr static SimpleHashMapConfig s = SimpleHashMapConfig{bucketsize, 1u << l, threads};
using HM = SimpleHashMap<K, V, s, &H<l>>;

HM hm = HM{};
hm.info();

const K key = 100;
const V value = 12937123;
hm.insert(key, value);
```


ParallelBucketSort:
===================
`sort.h`: `ParallelBucketSort<ConfigParallelBucketSort, List, LimbType, IndexType, HashFunction>`,
a bucket hashmap for the label parts `[b0, b2)` of a `List`, filled by
multiple threads. The bits `[b0, b1)` select the bucket, the buckets are
sorted on `[b1, b2)`. Tests: `tests/binary/sort.cpp`.

thashmap:
=========
`thashmap.h`: port of https://github.com/glk/thashmap. Unlike the hashmaps
above it grows, supports deletion, and is ordered: an intrusive radix trie for
30 bit keys (6 levels of 5 bit sub keys, nodes only store their present
children), whose nodes come from a `thm_pool` of 4 KiB pages.

```C
struct elem { uint32_t key; thm_entry entry; };
using map_t = thashmap<elem, offsetof(elem, entry), offsetof(elem, key)>;

thm_pool pool;
map_t m(pool);
elem e{42, {}};
while (m.insert(&e) == nullptr) { pool.new_block(); } // the pool is full
thm_bucket *b = m.find(42);                           // all elements with key 42
for (elem *x = map_t::bucket_first(b); x; x = map_t::bucket_next(x)) { ... }
thm_cursor c;                                         // ordered iteration
for (b = m.first(&c); b; b = map_t::next(&c)) { ... }
m.remove(&e);
```
Tests: `tests/container/thashmap.cpp`.

# IMAP

```C

```
