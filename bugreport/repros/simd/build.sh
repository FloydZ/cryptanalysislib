#!/bin/sh
f=$1; shift
clang++ -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fsanitize=address,undefined -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/src -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/build/_deps/reflect-cpp-src/include -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/deps/b63/include/b63 "$@" $f.cpp -o $f 2>&1 | grep -E "error|warning: shift" | head -30 && ./$f
