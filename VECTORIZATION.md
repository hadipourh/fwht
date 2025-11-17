# Vectorized Memory Access Implementation

## Changes Made:

### 1. Added Vec4Type Helper
```cpp
template <typename T> struct Vec4Type { using type = T; };
template <> struct Vec4Type<int32_t> { using type = int4; };
template <> struct Vec4Type<float> { using type = float4; };
template <> struct Vec4Type<double> { using type = double2; };
```

This maps scalar types to their vectorized equivalents for 16-byte loads/stores.

### 2. Vectorized Load (lines ~475-505)
- Uses `int4`/`float4` for 16-byte aligned loads
- 2 vectorized loads per thread (EPT=8, VEC_SIZE=4)
- Falls back to scalar loads if unaligned or boundary conditions
- Checks alignment at runtime: `(ptr % 16) == 0`

### 3. Vectorized Store (lines ~582-605)
- Uses `int4`/`float4` for 16-byte aligned stores
- 2 vectorized stores per thread
- Falls back to scalar stores if needed
- Ensures same `can_vectorize` condition as load

## Expected Performance Gain:
- **Memory bandwidth**: 2-4× improvement (fewer transactions)
- **Cache efficiency**: Better coalescing (128-byte cacheline utilization)
- **Overall speedup**: 1.5-2× for memory-bound kernels

## Testing:
```bash
# On server:
cd /workspace/libfwht/python
pip install -e . --force-reinstall --no-deps

# Run benchmark
cd /workspace/libfwht
python3 tools/compare_libs.py --powers 10 12 14 --batches 100 1000 --repeats 10
```

Compare new results vs previous:
- Previous: 9.45-14.12 GOps/s
- Target: 15-25 GOps/s (approaching Meta's 44-83 GOps/s)
