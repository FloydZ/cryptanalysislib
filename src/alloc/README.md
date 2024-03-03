Idea:
=====

The `cryptanalysislib` comes with its own memory allocation implementation. 
Currently the following allocators are implemented:
- StackAllocator
- FreeListAllocator
- FallbackAllocator
- AffixAllocator
- Segregator
- PageMallocator
- FreeListPageMallocator

- STDAllocatorWrapper

The reason a custom allocation library is deployed in this library is memory 
safety. Instead of returning a `void *`-pointer a memory `Blk` is returned. 
The structure `Blk` is defined as:
```C 
struct {
    void *ptr,
    const size_t len;
}
```
So instead of just returning the pointer, the length is also returned. Thus,
every memory access can trivial be checked for correctness.


Usage:
======

```C 
#include <cryptanalysislib/alloc/alloc.h>

StackAllocator<1024> s;
Blk b = s.allocate(1024);
```

Configuration:
--------------

Each allocator class can be configured with the `AllocatorConfig` class. Which
is defined [here](TODO). One can use it like this:

```C 
constexpr static AllocatorConfig config {
    .base_alignment=16,
    .alignment=16,
    ...
};
StackAllocator<1024, config> s;
Blk b = s.allocate(1024);
```

