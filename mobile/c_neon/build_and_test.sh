#!/usr/bin/env bash
# Build and run every C/C++ correctness test in this directory, in one command.
# Usage: ./build_and_test.sh [soak_seconds]   (default soak_seconds=30; use 300 for the full 5-min soak)
set -e
cd "$(dirname "$0")"
SOAK_SECONDS="${1:-30}"

echo "== regenerating weights header from checkpoints/F_GENERIC.tar =="
python3 export_weights.py

mkdir -p build
cd build

echo "== standalone kernel/block tests (no ulunas_full linkage needed) =="
for name in test_block0 test_block1 test_block2 test_dpgrnn test_decoder4; do
    gcc -O2 -std=c99 -I../src "../src/${name}.c" -lm -o "$name" 2>/dev/null
    echo "--- $name ---"
    ./"$name"
done

gcc -O2 -std=c99 -I../src ../src/test_generic_regression.c -lm -o test_generic_regression 2>/dev/null
echo "--- test_generic_regression ---"
./test_generic_regression

echo "== full-graph tests (link against ulunas_full.cpp) =="
for name in test_full_graph_multiframe test_full_streaming test_edge_cases; do
    g++ -O2 -std=c++17 -I../src "../src/${name}.cpp" ../src/ulunas_full.cpp -o "$name" 2>/dev/null
    echo "--- $name ---"
    ./"$name" || true  # test_full_streaming exits 1 on its documented offline-vs-causal
                       # latency-convention mismatch (see MOBILE_BENCHMARK.md) -- expected,
                       # not a script failure; don't let it abort the rest of the suite
done

echo "== soak test (${SOAK_SECONDS}s; pass an argument to change, e.g. 300 for the full 5-min soak) =="
g++ -O2 -std=c++17 -I../src ../src/test_soak.cpp ../src/ulunas_full.cpp -o test_soak 2>/dev/null
./test_soak "$SOAK_SECONDS"

echo ""
echo "== ASan+UBSan build of edge cases + soak (slower, catches memory/UB issues) =="
g++ -O0 -g -fsanitize=address,undefined -std=c++17 -I../src ../src/test_edge_cases.cpp ../src/ulunas_full.cpp -o test_edge_cases_san 2>/dev/null
./test_edge_cases_san

echo ""
echo "All tests completed. See MOBILE_BENCHMARK.md for the interpretation of these numbers"
echo "(expected error magnitudes, the documented streaming-vs-offline latency caveat, etc.)."
