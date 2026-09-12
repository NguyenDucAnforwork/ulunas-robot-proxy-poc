/* Compatibility header: the iOS bridge (ios/UlunasDemo/.../UlunasBridge.mm) was written
 * against a slightly different type/name convention (`UlunasContext`) than the actual
 * runtime-free engine implementation in ../src/ulunas_full.h (`UlunasState`), since the two
 * were developed in parallel. This header reconciles them with a type alias so both
 * workstreams' code compiles unchanged against the real implementation. Function signatures
 * now match exactly (ulunas_process_hop returns int, per the original bridge contract). */
#pragma once
#include "../src/ulunas_full.h"

typedef UlunasState UlunasContext;
