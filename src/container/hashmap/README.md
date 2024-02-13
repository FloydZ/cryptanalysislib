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

