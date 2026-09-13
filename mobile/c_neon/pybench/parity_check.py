import numpy as np

HOP = 256
FREQ = 257

print("=== END-TO-END (PCM) parity: C vs Python ===")
c = np.fromfile("output_pcm_c_run1.bin", dtype=np.float32).reshape(-1, HOP)
py = np.fromfile("output_pcm_py_run2.bin", dtype=np.float32).reshape(-1, HOP)
assert c.shape == py.shape
diff = np.abs(c - py)
print("shape:", c.shape)
print("max_abs_error (all hops):", float(diff.max()))
print("rmse (all hops):", float(np.sqrt(np.mean((c - py) ** 2))))
print("C has NaN/Inf:", bool(not np.all(np.isfinite(c))))
print("Python has NaN/Inf:", bool(not np.all(np.isfinite(py))))
for skip in [0, 1, 2, 4, 8]:
    d = diff[skip:]
    print(f"  excluding first {skip} hops: max_abs_error={d.max():.6e} rmse={np.sqrt(np.mean((c[skip:]-py[skip:])**2)):.6e}")

print()
print("=== MODEL-ONLY (spectral frame) parity: C vs Python ===")
c2 = np.fromfile("output_spec_c_run1.bin", dtype=np.float32).reshape(-1, FREQ, 2)
py2 = np.fromfile("output_spec_py_run2.bin", dtype=np.float32).reshape(-1, FREQ, 2)
assert c2.shape == py2.shape
diff2 = np.abs(c2 - py2)
print("shape:", c2.shape)
print("max_abs_error (all frames):", float(diff2.max()))
print("rmse (all frames):", float(np.sqrt(np.mean((c2 - py2) ** 2))))
print("C has NaN/Inf:", bool(not np.all(np.isfinite(c2))))
print("Python has NaN/Inf:", bool(not np.all(np.isfinite(py2))))
for skip in [0, 1, 2, 4, 8]:
    d = diff2[skip:]
    print(f"  excluding first {skip} frames: max_abs_error={d.max():.6e} rmse={np.sqrt(np.mean((c2[skip:]-py2[skip:])**2)):.6e}")
