#!/usr/bin/env python3
"""
Test dispatch logging to see which kernel is used for each size.
"""
import os
os.environ['FWHT_DISPATCH_LOGGING'] = '1'

import pyfwht
import numpy as np

print("Testing kernel dispatch with logging enabled")
print("=" * 60)

ctx = pyfwht.gpu.Context(max_n=8192, batch_size=1)

for n in [512, 1024, 2048, 4096, 8192]:
    print(f"\n=== n={n} ===")
    x = np.random.randint(-100, 100, size=(n,), dtype=np.int32)
    ctx.transform_i32(x)
    
ctx.close()
