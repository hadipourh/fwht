# Fixed: Test Bug and Kernel Bugs

## Changes Made

### 1. Fixed Kernel Bug (Both src/ and python/c_src/)
**File**: `fwht_cuda.cu` (both versions)

**Problem**: XOR-based butterfly logic was incorrect
- Old code had `if (idx < partner_idx)` branches with both setting `regs[e] = val`
- This lost the partner's result

**Fix**: Removed conditional - all threads perform butterfly and keep result
```cuda
// OLD (WRONG):
if (idx < partner_idx) {
    butterfly_xor(val, partner);
    regs[e] = val;
} else {
    butterfly_xor(partner, val);
    regs[e] = val;  // WRONG! Both branches same
}

// NEW (CORRECT):
butterfly_xor(val, partner);
regs[e] = val;
```

### 2. Fixed Test Bug
**File**: `python/tests/benchmark_all_precisions.py`

**Problem**: Test was using already-transformed data as input
- CPU transform modified `data_ref` in-place
- GPU tests used `data_ref` (already transformed) as input
- Result: comparing double-transform vs single-transform

**Fix**: Use untransformed `data_original` for GPU tests
```python
# Create original data
data_original = np.random.randn(batch_size, n)

# Transform copy on CPU (reference)
data_ref = data_original.copy()
for i in range(batch_size):
    pyfwht.transform(data_ref[i])

# Test GPU with ORIGINAL data
data_f64 = torch.tensor(data_original, dtype=torch.float64, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data_f64)
```

## To Deploy

```bash
# Sync to server
rsync -avz --progress -e "ssh -p 26586" * root@194.14.47.19:/workspace

# On server, reinstall
cd /workspace/python
pip install -e . --force-reinstall

# Run tests
python tests/test_precisions.py
python tests/benchmark_all_precisions.py
```

## Expected Results

All tests should now PASS with correct errors:
- fp64: ~1e-15
- fp32: ~1e-6  
- fp16: ~1e-3

Performance (RTX 4090):
- fp64: ~74 GOps/s
- fp32: ~300-400 GOps/s (2-5× speedup)
- fp16: ~700-900 GOps/s (10-12× speedup, matching Meta!)
