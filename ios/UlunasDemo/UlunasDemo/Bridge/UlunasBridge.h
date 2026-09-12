// Objective-C++ bridge exposing the runtime-free C engine (and an optional ONNX Runtime
// reference backend) to Swift with a single, backend-agnostic interface.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// NOTE: deliberately spelled "Onnx" (not "ONNX") so Swift's automatic NS_ENUM case-name
// bridging (which strips the "UlunasBackend" prefix and lowercases the leading run of
// capitals) produces the predictable `.runtimeFreeC` / `.onnxRuntime` -- an all-caps "ONNX"
// prefix is bridged inconsistently across Swift versions and was confirmed to cause exactly
// this ambiguity when first drafted with that spelling.
typedef NS_ENUM(NSInteger, UlunasBackend) {
    UlunasBackendRuntimeFreeC = 0,  // primary: mobile/c_neon, no ONNX Runtime linked
    UlunasBackendOnnxRuntime = 1,   // reference/debug: onnxruntime-objc, build-flag gated
};

/// Thin wrapper: one instance per active backend. Not thread-safe to call concurrently from
/// multiple threads -- the app calls `processHop` only from its single dedicated inference
/// thread (see AudioEngineManager.swift).
@interface UlunasEngine : NSObject

- (nullable instancetype)initWithBackend:(UlunasBackend)backend;

/// input/output must each point to exactly hopSize (256) float samples. Returns YES on success.
/// Must not allocate or block -- safe to call from a real-time-priority thread.
- (BOOL)processHopWithInput:(const float *)input output:(float *)output;

- (void)reset;

@property (nonatomic, readonly) NSInteger hopSize;
@property (nonatomic, readonly) UlunasBackend backend;

@end

NS_ASSUME_NONNULL_END
