This is the backbone implementation of our paper [McEliece needs a Break](https://eprint.iacr.org/2021/1634)
and our second paper [New Time-Memory Trade-Offs for Subset-Sum](https://eprint.iacr.org/2022/1329).

Ideas:
======

This repository aims to provide a STL which a few unique tweakes:
- Most (if not all) datastructures implemented in this library do not have a 
    delete/remove operation. Meaning you cannot delete an element once inserted
    into the datastructure. But its possible to clear (or reset) the whole 
    datastructure at once. Which doesnt super useful in general is very useful 
    in the setting of cryptanalysis, where we only insert "useful" elements.
    And after we checked if there is no (partial-) solution somewhere, we can
    simply clear everything and start from the beginnning.
- constexpr: All datastructure assume that you know beforehand how many elements
    you need to insert. Or many elements you have to store at max. E.g. resizing
    of any datastructure is not possible. Also memory will never freed.
- HPC: All datastructure are optimized or implemented in a way to reduce 
    cache-misses.

Requirements
============
Basically you need a `C++20` rdy compiler, `cmake 3.10`.For testing and 
benchmarking you need `gtest` and `googlebenchmark`.

## Arch Linux:
```bash
sudo pacman -S cmake make gtest benchmark clang
```

## Ubuntu 22.04:
```bash
sudo apt install libgtest-dev googletest cmake make 
```

It could be that `googletest` is not correctly installed, if so try;
```bash
# gtest is somehow quite difficult:
sudo cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp *.a /usr/lib
```

## NixOS
```bash
nix-shell
mkdir build
cd build
cmake ..
make
```

## MacOS
```bash
brew install cmake make tbb gcc googletest autoconf automake libtool google-benchmark gcc libomp
```
Follow this (Guide)[https://mac.r-project.org/openmp/] to install openmp on osx.

Somehow Apple is not supporting static linking? so you need to execute the following command once:
```bash
cd deps/m4ri
sudo make install
```

## Windows: 
I wish you luck with this one.

How to build
------
```bash
git clone --recurse-submodules -j4 https://github.com/FloydZ/cryptanalysislib
cd cryptanalysislib && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
```

A few notes on the cmake flags:
- for debugging you can also pass `-DCMAKE_BUILD_TYPE=Debug`.
- if you do not pass any flag (so neither `Debug`, nor `Release`) and optimized build without SIMD will be compiled


Label, Value, Element, Matrix and Lists: 
========================================
The core concept of the main data containers of this library are `Label` and 
`Value`. A `Label` is the result of the multiplication of an (error-) vector, 
called `Value` with a `Matrix`. Or mathematical speaking: `Label = Matrix*Value`.
If you are familiar with ISD algorithms or the decoding of codewords, this looks
a lot like the syndrome equation `s = H*e`, where `s` is the sybdrome, `e` a 
(probably) unknown error and `H` the parity check matrix of your code. 

This design is chosen for the case in which the error vector `e` (which is 
saved in a `Value`) is unknown. Hence one wants to iterate a lot of Values which 
matches the correct `Label` you can order them in a [List](TODO). As a matter 
of fact, this libraries offers of lot of different implementations. 

Internally a set of a `Value, Label` and `Matrix` is called an `Element`, which 
maps each `Value` via a `Matrix` to an unique `Value`. More on each of those 
containers you find here: [Label](TODO), [Value](TODO), [Matrix](TODO), 
[Element]().


Implementation Details:
=======================

The following things are implemented:
- Custom [allocators](./src/alloc/README.md) which do not return a memory 
    allocation, but memory blocks which do not allow for memory missuses.
- Binary- and Fq-[Enumeration](./src/combination/README.md) which enumerate vectors of length `n` and weight 
    `w` in a loop-less efficient way. Either a [Chase](TODO)-Sequence or a 
    [Gray-Code](TODO) or a combination of both. 
- [LinkedList](./src/container/linkedlist/README.md)
- [HashMaps](./src/container/hashmap/README.md)
- [Permutation](./src/permutation/README.md)
- [Search](./src/search/README.md)
- [SIMD](./src/simd/README.md)

A lot of different data containers are implemented:
- `BinaryContainer<T, len>` is able to hold `len` bits in `len/(sizeof(T)*8)` 
    limbs of type `T`. Additionally, all important `add,sub,compare` functions 
    are implemented
- `kAryType<T, T2, q>` represents a value `mod q`. The second type `T2` is 
    needed to sanely implement the multiplication.
- `kAryContainer<T, len>` holds `len` elements `mod q` and each element is 
    saved in its own limb of type `T`. 
- `kAryPackedContainer<T, len>` same as `kAryContainer<T, len>` but the 
    implementations stores as much as possible elements `mod q` in one limb of 
    type `T`.

These datatypes can be used to instantiate a `Label` and `Value` 
(which form together an `Element<Value_T, Label_T, Matrix>`), where 
`Label = H \times Value` for any Matrix `H` of any type (binary, ternary, 
kAry, ...).
Note: only for certain primes and prime-powers are speciallized arithmetics 
implemented. If you chose an unsupported one, a slow generic backup 
implementation will be used. If so you will be warned.

As well as this core datatypes the following list-containers are implemented:
- `Parallel_List` A list which separates `Value` from `Label` in two different
    lists to allow faster enumeration of one of types while one does not care 
    about the other. Additionally, each of the two lists is split into chucks 
    on which threads can operate independent of each other. Note: sorting is 
    currently not possible in this list.
- `Parallel_List_FullElement` Same as `Parallel_List`, but `Values` and 
    `Labels` are together saved in one list of `Elements`, which allows sorting. 
- `Parallel_List_IndexElement` TODO
- `List` generic list implementation.
Range checks are performed in every implementation.

All matrices are represented with the matrix class
- [Matrix](./src/matrix/README.md)

The following sorting algorithms are available.
-ska_sort 
-timsort

Currently `ska_sort` (Link)[https://github.com/skarupke/ska_sort] is used as 
the main sorting algorithm. Note that sorting is avoided in the code as much 
as possible. Do not sort if you want fast code.



Benchmarks
===
Can be found [here](https://floydz.github.io/cryptanalysislib/dev/bench/)


TODO
===
explain:
- List generators,
- triple
- mccd mem tracker
- mccl: hashmap neu streiben ohne load factor, indem die hash funk ein rotate einbaut
- matrix: more tests via constexpr loops
- binary_matrix aufräumen
- Die sorting algorithmen in `list`, davon die hashfunktionen zusammenfassen 
    und die #ifdefs weg. Wahrscheinlich `parallel.h` weg? verstehe nicht so ganz was die implementierung soll, wenn ist ListT gibt.
