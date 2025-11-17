# Installing Meta Library for Benchmarking

## The Problem
The diagnostic shows Meta library is not installed in your venv:
```
Python executable: /venv/main/bin/python3
Fast_hadamard_transform: ModuleNotFoundError
```

## Quick Fix (On Server)

```bash
# 1. Activate the venv that your benchmark uses
source /venv/main/bin/activate

# 2. Verify you're in the right environment
which python3
# Should show: /venv/main/bin/python3

# 3. Install Meta library
cd /tmp
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .

# 4. Verify installation
python3 -c "import fast_hadamard_transform; print(fast_hadamard_transform.__version__)"

# 5. Run diagnostic again
cd /workspace/libfwht
python3 tools/debug_imports.py

# 6. Run benchmark
python3 tools/compare_libs.py --powers 10 12 --batches 1 100 --repeats 5
```

## Alternative: Use System Python

If you don't want to use the venv:
```bash
# Deactivate venv
deactivate

# Install into system python (or create new venv)
pip3 install torch
cd /tmp
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip3 install -e .

# Run benchmark with system python
cd /workspace/libfwht
python3 tools/compare_libs.py --powers 10 12 --batches 1 100
```

## What This Will Fix
Once Meta library is installed, the benchmark will show:
```
META gpu n=1024 b=1: X.XXX ms, XX.XX GOps/s
META gpu n=4096 b=1: X.XXX ms, XX.XX GOps/s
```

Instead of:
```
META gpu n=1024 b=1: skipped (unavailable)
```
