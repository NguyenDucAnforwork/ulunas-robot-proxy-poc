// Objective-C++ implementation. Wraps either:
//   (a) the runtime-free C engine (mobile/c_neon/, header at mobile/c_neon/include/ulunas_api.h)
//   (b) the ONNX Runtime iOS reference backend (onnxruntime-objc), gated behind
//       ULUNAS_ENABLE_ONNXRUNTIME_BACKEND so the primary build has zero ONNX Runtime linkage,
//       matching the project requirement that the runtime-free path not depend on it.
#import "UlunasBridge.h"

extern "C" {
#include "ulunas_api.h"
}

#if ULUNAS_ENABLE_ONNXRUNTIME_BACKEND
#import <onnxruntime/onnxruntime_cxx_api.h>
#endif

@implementation UlunasEngine {
    UlunasBackend _backend;
    UlunasContext *_ctx; // runtime-free C engine handle, NULL if using the ONNX backend
#if ULUNAS_ENABLE_ONNXRUNTIME_BACKEND
    std::unique_ptr<Ort::Env> _ortEnv;
    std::unique_ptr<Ort::Session> _ortSession;
    // Streaming state for the ONNX backend mirrors ulunas_api's internal state: an analysis
    // window ring, the packed conv/tfa/inter caches, and the overlap-add accumulator. Kept
    // separate from the C engine's state deliberately, so the two backends can run
    // side-by-side on the same input for the on-device comparison harness (see
    // BackendComparisonHarness.swift) without sharing mutable state.
#endif
}

- (nullable instancetype)initWithBackend:(UlunasBackend)backend {
    self = [super init];
    if (!self) return nil;
    _backend = backend;

    switch (backend) {
        case UlunasBackendRuntimeFreeC: {
            if (ulunas_init(&_ctx) != 0) {
                NSLog(@"[UlunasEngine] ulunas_init failed");
                return nil;
            }
            break;
        }
        case UlunasBackendOnnxRuntime: {
#if ULUNAS_ENABLE_ONNXRUNTIME_BACKEND
            @try {
                _ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ulunas_ios_ref");
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(1);
                NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"ulunas_finetuned_stream_simple" ofType:@"onnx"];
                if (!modelPath) {
                    NSLog(@"[UlunasEngine] bundled ONNX model not found -- add it to the app bundle's Resources");
                    return nil;
                }
                _ortSession = std::make_unique<Ort::Session>(*_ortEnv, [modelPath UTF8String], opts);
            } @catch (NSException *e) {
                NSLog(@"[UlunasEngine] ONNX Runtime init failed: %@", e.reason);
                return nil;
            }
#else
            NSLog(@"[UlunasEngine] ONNX Runtime backend requested but not compiled in (set ULUNAS_ENABLE_ONNXRUNTIME_BACKEND=1 and link onnxruntime-objc)");
            return nil;
#endif
            break;
        }
    }
    return self;
}

- (void)dealloc {
    if (_ctx) {
        ulunas_destroy(_ctx);
        _ctx = nullptr;
    }
}

- (NSInteger)hopSize {
    return ULUNAS_HOP_SIZE;
}

- (UlunasBackend)backend {
    return _backend;
}

- (BOOL)processHopWithInput:(const float *)input output:(float *)output {
    switch (_backend) {
        case UlunasBackendRuntimeFreeC:
            return ulunas_process_hop(_ctx, input, output) == 0;
        case UlunasBackendOnnxRuntime:
#if ULUNAS_ENABLE_ONNXRUNTIME_BACKEND
            // NOTE: the ONNX reference backend still needs the SAME host-side STFT/ISTFT
            // framing + cache-packing logic the runtime-free engine has internally (see
            // mobile/onnx_export/export_finetuned_stream.py and mobile/android_ref/src/main.cpp
            // for the reference implementation of that framing). For the on-device debug/
            // comparison backend, port that exact logic here rather than re-deriving it, to
            // keep both backends numerically comparable. Not implemented in this pass --
            // wiring the runtime-free C path was the priority; this is a documented follow-up,
            // not a silent gap.
            NSLog(@"[UlunasEngine] ONNX backend processHop: STFT/cache framing not yet ported to this bridge -- see comment above");
            return NO;
#else
            return NO;
#endif
    }
}

- (void)reset {
    if (_ctx) {
        ulunas_reset(_ctx);
    }
    // ONNX backend: reset its host-side streaming state here once implemented (see note above).
}

@end
