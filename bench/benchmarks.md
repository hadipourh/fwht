# libfwht Benchmark Log

This document records reproducible benchmark runs, including system configuration, command output, and a short description of each experiment.


## 2025-11-30 — SymPy FWHT vs pyfwht (CPU)
**Description:** CPU comparison between pyfwht's optimized C backend and SymPy's reference Walsh–Hadamard transform for powers of two (2⁶…2¹⁹). Demonstrates both correctness (exact agreement) and the speed gap versus a pure-Python baseline.

**System configuration:**
- macOS 15.7.1 (24G231)
- Apple M4 (arm64)
- Kernel: Darwin 24.6.0 (arm64)
- Python environment: `myenv` (Python 3.14)

```
/Users/hoseinhadipour/Projects/libfwht/.venv/bin/python bench/compare_sympy_fwht.py
pyfwht vs SymPy FWHT (CPU)
--------------------------------------------------
[libfwht] CPU backend: NEON vector path active
Running n=64 ... ok
Running n=128 ... ok
Running n=256 ... ok
Running n=512 ... ok
Running n=1024 ... ok
Running n=2048 ... ok
Running n=4096 ... ok
Running n=8192 ... ok
Running n=16384 ... ok
Running n=32768 ... ok
Running n=65536 ... ok
Running n=131072 ... ok
Running n=262144 ... ok
Running n=524288 ... ok

   n (2^k) |  SymPy (s) |  pyfwht (s) |  speedup |  max |diff|
--------------------------------------------------------------
2^ 6=   64 |    0.00106 |     0.00000 |   338.92 |   0.000e+00
2^ 7=  128 |    0.00213 |     0.00000 |   624.46 |   0.000e+00
2^ 8=  256 |    0.00373 |     0.00000 |   993.42 |   0.000e+00
2^ 9=  512 |    0.00661 |     0.00000 |  2367.17 |   0.000e+00
2^10= 1024 |    0.01282 |     0.00000 |  4186.07 |   0.000e+00
2^11= 2048 |    0.02691 |     0.00001 |  5037.23 |   0.000e+00
2^12= 4096 |    0.05789 |     0.00001 |  7095.51 |   0.000e+00
2^13= 8192 |    0.12862 |     0.00002 |  6615.72 |   0.000e+00
2^14=16384 |    0.26529 |     0.00003 |  8364.44 |   0.000e+00
2^15=32768 |    0.57513 |     0.00006 |  9193.54 |   0.000e+00
2^16=65536 |    1.23334 |     0.00013 |  9589.29 |   0.000e+00
2^17=131072 |    2.65330 |     0.00028 |  9611.67 |   0.000e+00
2^18=262144 |    5.58323 |     0.00063 |  8793.19 |   0.000e+00
2^19=524288 |   11.93385 |     0.00135 |  8843.22 |   0.000e+00

Largest observed speedup: 9611.67×
```


## 2025-11-30 — pyfwht vs Hadamard.jl (CPU)
**Description:** Direct comparison between pyfwht's CPU backend (NEON-optimized) and the Julia Hadamard.jl implementation for the same random inputs (2⁶…2²⁷). Both sides run a fixed 10 repetitions per size, Julia is granted all 10 available threads via `JULIA_NUM_THREADS`, and Hadamard.jl outputs are scaled by *n* to match pyfwht's unnormalized convention. The harness defaults still allow extending to 2³⁰ when memory allows.

**System configuration:**
- macOS 15.7.1 (24G231)
- Apple M4 (arm64)
- Kernel: Darwin 24.6.0 (arm64)
- Python environment: `myenv` (Python 3.14)
- Julia: 1.12.2 (Homebrew) with project `otherlibs/Hadamard.jl`

```
/Users/hoseinhadipour/Projects/libfwht/.venv/bin/python bench/compare_pyfwht_vs_hadamardjl.py --julia-bin /opt/homebrew/bin/julia --max-power 27 --repeat 10
pyfwht vs Hadamard.jl (CPU)
------------------------------------------------------------------------
Julia project : /Users/hoseinhadipour/Projects/libfwht/otherlibs/Hadamard.jl
Julia threads : 10
pyfwht backend: CPU
Repeat count  : 10
[libfwht] CPU backend: NEON vector path active
Running n=64 (repeat=10) ... ok
Running n=128 (repeat=10) ... ok
Running n=256 (repeat=10) ... ok
Running n=512 (repeat=10) ... ok
Running n=1024 (repeat=10) ... ok
Running n=2048 (repeat=10) ... ok
Running n=4096 (repeat=10) ... ok
Running n=8192 (repeat=10) ... ok
Running n=16384 (repeat=10) ... ok
Running n=32768 (repeat=10) ... ok
Running n=65536 (repeat=10) ... ok
Running n=131072 (repeat=10) ... ok
Running n=262144 (repeat=10) ... ok
Running n=524288 (repeat=10) ... ok
Running n=1048576 (repeat=10) ... ok
Running n=2097152 (repeat=10) ... ok
Running n=4194304 (repeat=10) ... ok
Running n=8388608 (repeat=10) ... ok
Running n=16777216 (repeat=10) ... ok
Running n=33554432 (repeat=10) ... ok
Running n=67108864 (repeat=10) ... ok
Running n=134217728 (repeat=10) ... ok

     n (2^k) |  pyfwht (s) | Hadamard.jl (s) |       Faster (×) |  max |diff|
-----------------------------------------------------------------------------
2^ 6=     64 |    0.000008 |        0.001738 |   pyfwht ×225.65 |   0.000e+00
2^ 7=    128 |    0.000003 |        0.001646 |   pyfwht ×579.94 |   0.000e+00
2^ 8=    256 |    0.000003 |        0.001836 |   pyfwht ×562.73 |   0.000e+00
2^ 9=    512 |    0.000004 |        0.001852 |   pyfwht ×523.61 |   0.000e+00
2^10=   1024 |    0.000004 |        0.002074 |   pyfwht ×557.96 |   0.000e+00
2^11=   2048 |    0.000005 |        0.002151 |   pyfwht ×405.93 |   0.000e+00
2^12=   4096 |    0.000008 |        0.003149 |   pyfwht ×389.52 |   0.000e+00
2^13=   8192 |    0.000015 |        0.002754 |   pyfwht ×189.20 |   0.000e+00
2^14=  16384 |    0.000031 |        0.002892 |    pyfwht ×94.76 |   0.000e+00
2^15=  32768 |    0.000056 |        0.003004 |    pyfwht ×53.57 |   0.000e+00
2^16=  65536 |    0.000117 |        0.003293 |    pyfwht ×28.12 |   0.000e+00
2^17= 131072 |    0.000270 |        0.003880 |    pyfwht ×14.35 |   0.000e+00
2^18= 262144 |    0.000535 |        0.003962 |    pyfwht × 7.41 |   0.000e+00
2^19= 524288 |    0.001168 |        0.004544 |    pyfwht × 3.89 |   0.000e+00
2^20=1048576 |    0.002546 |        0.005599 |    pyfwht × 2.20 |   0.000e+00
2^21=2097152 |    0.005300 |        0.007424 |    pyfwht × 1.40 |   0.000e+00
2^22=4194304 |    0.011232 |        0.011155 |  Hadamard × 1.01 |   0.000e+00
2^23=8388608 |    0.023838 |        0.019100 |  Hadamard × 1.25 |   0.000e+00
2^24=16777216 |    0.050218 |        0.034588 |  Hadamard × 1.45 |   0.000e+00
2^25=33554432 |    0.105886 |        0.066366 |  Hadamard × 1.60 |   0.000e+00
2^26=67108864 |    0.223484 |        0.135599 |  Hadamard × 1.65 |   0.000e+00
2^27=134217728 |    0.472922 |        0.279899 |  Hadamard × 1.69 |   0.000e+00

Average ratios — py/jl: 0.47×, jl/py: 165.66× (max diff 0.000e+00)
```

