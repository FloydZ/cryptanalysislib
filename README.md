This is the backbone implementation of our [Paper](https://eprint.iacr.org/2021/1634)
and our second [paper](https://eprint.iacr.org/2022/1329).

Requirements
-----
Basically you need a `C++20` rdy compiler, `cmake 3.10` and `autoconf` for building `fplll, m4ri`. For testing and benchmarking you need `gtest` and `googlebenchmark`.

## Arch Linux:
```bash
sudo pacman -S cmake make autoconf automake fplll gtest mpfr bazel gperftools benchmark clang
```

## Ubuntu:
```bash
sudo apt install autoconf automake libfplll-dev libfplll4 fplll-tools libgtest-dev googletest cmake make libmpfrc++-dev bazel libmpfr-dev libmpfr-doc libmpfr6 libmpfrc++-dev libm4ri-dev libpng-dev libpng++-dev libtbb-dev

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
brew install cmake make tbb gcc googletest autoconf automake libtool googlebenchmark
```
Follow this (Guide)[https://mac.r-project.org/openmp/] to install openmp on osx.

## Windows: 
I wish you luck with this one.

How to build
------
```bash
git clone --recurse-submodules -j4 https://github.com/FloydZ/cryptanalysislib
cd cryptanalysislib && mkdir build && cd build && cmake ..
```

List of compiler flags
-------
A list of used preprocessor flags. Use them with care, some of them can break the code.
```
#define SORT_INCREASING_ORDER | SORT_INCREASING_ORDER   
#define SORT_PARALLEL                                   untested
#define SEARCH_PARALLEL                                 untested
#define USE_LOOP_UNROLL                                 in most of the add/sub/cmp functions the loops will be unrolled
#define USE_PREFETCH                                    untested
#define USE_BRANCH_PREDICTION                           untested
```

Implementation Details
=======

The following datatypes are implemented:
- `BinaryContainer<T, len>` is able to hold `len` bits in `len/(sizeof(T)*8)` limbs of type `T`. Additionally, all important `add,sub,compare` functions are implemented
- `kAryType<T, T2, q>` represents a value `mod q`. The second type `T2` is needed to sanely implement the multiplication.
- `kAryContainer<T, len>` holds `len` elements `mod q` and each element is saved in its own limb of type `T`. 
- `kAryPackedContainer<T, len>` same as `kAryContainer<T, len>` but the implementations stores as much as possible elements `mod q` in one limb of type `T`.

These datatypes can be used to instantiate a `Label` and `Value` (which form together an `Element<Value_T, Label_T, Matrix>`), where `Label = H \times Value` for any Matrix `H` of any type (binary, ternary, kAry, ....).

As well as this core datatypes the following containers are implemented:
- `Parallel_List` A list which separates `Value` from `Label` in two different lists to allow faster enumeration of one of types while one does not care about the other. Additionally, each of the two lists is split into chucks on which threads can operate independent of each other. Note: sorting is currently not possible in this list.
- `Parallel_List_FullElement` Same as `Parallel_List`, but `Values` and `Labels` are together saved in one list of `Elements`, which allows sorting. 
- `Parallel_List_IndexElement` unused
- `List` generic list implementation.
Range checks are performed in every implementation.

Currently `ska_sort` (Link)[https://github.com/skarupke/ska_sort] is used as the main sorting algorithm. Note that sorting is avoided in the code as much as possible. Do not sort if you want fast code.


Benchmarks
===
Can be found [here](https://floydz.github.io/cryptanalysislib/dev/bench/)


TODO
===
explain:
- List generators,
- cache sequence generators
- matrix
- triple
- hashmap

