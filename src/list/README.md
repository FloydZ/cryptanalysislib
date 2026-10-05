All lists are implemented in such a way that wait/lock free access to the elements is possible. But not all threads have write access to all elements. Each list is designed is such a way that each threads operates on its portion of the list.

Each list assumes that the base container implements some basic functionalities
like comparison/add/sub/size... .

Each list comes with a load factor which tracks how many elements per thread 
where already inserted. Once the max number is reached, new elements are discarded.
This makes the resetting of the list trivial, simply overwrite the load array
with zero.

### `MetaList`
Main class. All other classes inherit from this one. This class implements all 
the main functionalities like:
    - rng generation, access operators, size(), base types, 
    - iterators, ...
implemented in `list/common.h`

### `List`,
    implements: sorting, searching

### `Parallel_List`
### `Parallel_List`
### `Parallel_List_FullElements`
### `Parallel_List_Index`
### `Parallel_List_Simple`
### `Parallel_List_Limb`
