#!/bin/bash
D=/private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_containers
f=$1; shift
clang++ -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fsanitize=undefined -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG -Wno-invalid-constexpr "-Dulong=unsigned long" -I$D/src -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/build/_deps/reflect-cpp-src/include "$@" $D/$f.cpp -o $D/$f 2>&1 | grep -E "error" -A3 | head -30
