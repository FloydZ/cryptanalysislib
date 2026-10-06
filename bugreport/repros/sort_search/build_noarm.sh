#!/bin/sh
D=/private/tmp/claude-503/-Users-ai-Downloads-crypto-lib-cryptanalysislib/fbab725c-4838-4aed-b795-5f43f37adbcf/scratchpad/agent_sort
eval clang++ $(cat $D/flags_noarm.txt) -O0 -fsanitize=undefined "$1.cpp" -o "$1" 2>&1 | grep -E "error" -A3 | head -40
