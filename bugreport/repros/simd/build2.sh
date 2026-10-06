#!/bin/sh
# same as build.sh but with neon.h duplicate lt/lt_ (lines 3277-3310) removed so the TU compiles
f=$1; shift
D=/private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_simd
clang++ -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fsanitize=undefined -I$D/patched -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/src -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/build/_deps/reflect-cpp-src/include -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/deps/b63/include/b63 "$@" $D/$f.cpp -o $D/$f 2>&1 | grep -E "error" -A3 | head -40 ; timeout 60 $D/$f
