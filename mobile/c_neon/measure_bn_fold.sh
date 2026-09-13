#!/usr/bin/env bash
# Self-contained correctness + native-host timing; no /tmp/*.npy fixtures.
set -euo pipefail
cd "$(dirname "$0")"
python3 export_weights.py
python3 export_weights.py --fold-bn
mkdir -p build/bn_fold
CXX="${CXX:-g++}"
flags=(-O2 -std=c++17 -Isrc)
"$CXX" "${flags[@]}" -c src/ulunas_full.cpp -o build/bn_fold/reference.o
"$CXX" "${flags[@]}" -DULUNAS_BN_FOLDED -c src/ulunas_full.cpp -o build/bn_fold/folded.o
rename=(-DUlunasState=FoldedUlunasState)
for name in init destroy reset process_frame_spec process_hop state_size_bytes; do
    rename+=("-Dulunas_${name}=folded_ulunas_${name}")
done
"$CXX" "${flags[@]}" -DULUNAS_BN_FOLDED "${rename[@]}" -c src/ulunas_full.cpp -o build/bn_fold/folded_parity.o
"$CXX" "${flags[@]}" src/test_bn_fold_parity.cpp build/bn_fold/reference.o build/bn_fold/folded_parity.o -o build/bn_fold/test_parity
build/bn_fold/test_parity | tee build/bn_fold/parity.log
for variant in reference folded; do
    "$CXX" "${flags[@]}" src/test_edge_cases.cpp "build/bn_fold/$variant.o" -o "build/bn_fold/edge_$variant"
    "build/bn_fold/edge_$variant" | tee "build/bn_fold/edge_$variant.log"
    "$CXX" "${flags[@]}" src/benchmark_latency.cpp "build/bn_fold/$variant.o" -o "build/bn_fold/benchmark_$variant"
done
# Sequential, alternating order to expose host timing variability; every run is 2000/100.
# Optional CPU=N pins only benchmarks to one allowed logical CPU.
runner=()
if [[ -n "${CPU:-}" ]]; then runner=(taskset -c "$CPU"); fi
for pair in 1 2 3; do
    variants=(reference folded)
    if [[ "$pair" == 2 ]]; then variants=(folded reference); fi
    for variant in "${variants[@]}"; do
        echo "== pair $pair: $variant (native host, not ARM/iPhone) =="
        "${runner[@]}" "build/bn_fold/benchmark_$variant" 2000 100 | tee "build/bn_fold/latency_${pair}_$variant.log"
    done
done
