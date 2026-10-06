#!/bin/bash
f=$1; shift
env -i PATH=/usr/bin:/bin HOME=$HOME /opt/homebrew/bin/g++-16 -std=gnu++23 -g -O0 -DUSE_ARM -DDEBUG -flax-vector-conversions -fsanitize=undefined -I/private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_algo/srcp -I/Users/ai/Downloads/crypto/lib/cryptanalysislib/build/_deps/reflect-cpp-src/include "$@" $f.cpp -o $f && timeout 60 ./$f
