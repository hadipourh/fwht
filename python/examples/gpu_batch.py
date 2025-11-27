"""
GPU batch transform example using pyfwht (if CUDA is available).

Demonstrates:
- Querying device info
- Enabling profiling and reading metrics
- Running in-place batch transforms for int32 and float64
- Using a persistent GPU context for repeated transforms
"""

import numpy as np
import pyfwht as fwht


def main():
    if not fwht.gpu.available:
        print("GPU/CUDA not available. Skipping example.")
        return

    print("Device:", fwht.gpu.device_name())
    print("Compute capability:", fwht.gpu.compute_capability())
    print("SM count:", fwht.gpu.sm_count())
    print("SMEM banks:", fwht.gpu.smem_banks())

    # Prepare batch data (int32)
    batch_i32 = np.random.randint(-10, 10, size=(8, 256), dtype=np.int32)

    # Enable profiling
    fwht.gpu.set_profiling(True)

    # In-place batch transform
    fwht.gpu.batch_transform_i32(batch_i32)
    metrics = fwht.gpu.get_last_metrics()
    print(f"Batch i32 kernel: {metrics['kernel_ms']:.3f} ms (n={metrics['n']}, batch={metrics['batch_size']})")

    # Float64 batch
    batch_f64 = np.random.randn(4, 512).astype(np.float64)
    fwht.gpu.batch_transform_f64(batch_f64)
    metrics = fwht.gpu.get_last_metrics()
    print(f"Batch f64 kernel: {metrics['kernel_ms']:.3f} ms (n={metrics['n']}, batch={metrics['batch_size']})")

    # Persistent context
    with fwht.gpu.Context(max_n=512, batch_size=8) as ctx:
        vec = np.random.randn(512).astype(np.float64)
        ctx.transform_f64(vec)
        metrics = fwht.gpu.get_last_metrics()
        print(f"Context f64 kernel: {metrics['kernel_ms']:.3f} ms")

    # Reset profiling
    fwht.gpu.set_profiling(False)


if __name__ == "__main__":
    main()
