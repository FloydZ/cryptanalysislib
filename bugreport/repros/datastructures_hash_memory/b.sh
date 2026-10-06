#!/bin/bash
# usage: b.sh src.cpp [extra flags]
R=/Users/ai/Downloads/crypto/lib/cryptanalysislib
GT=/nix/store/ih02qcciv6kl4khlk30682ca914ssl3s-gtest-1.17.0-dev/include
GL=/nix/store/9s69n8vhfzard8qv70dsk31lrd161ms4-gtest-1.17.0/lib
src=$1; shift
out=${src%.cpp}
clang++ -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fsanitize=undefined -fno-sanitize=alignment -I/private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_ds/src -I$R/build/_deps/reflect-cpp-src/include -I$GT -include /private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_ds/ulong.h "$@" $src -o $out -L$GL -lgtest -Wl,-rpath,$GL 2>&1 | grep -E "error|Error" | head -20
