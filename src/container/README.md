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

#  random setter
c.random();
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

### kAryContainer_T<T, n>
holds `len` elements `mod q` and each element is  saved in its own limb of 
type `T`. 

### kAryPackedContainer_T<T, n>
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
