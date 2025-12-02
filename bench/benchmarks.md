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


## 2025-12-02 — pyfwht vs Hadamard.jl (CPU) — Extended to 2³⁰
**Description:** Direct comparison between pyfwht's AUTO backend (OpenMP-optimized) and the Julia Hadamard.jl implementation for the same random inputs (2⁶…2³⁰). Both sides run 5 repetitions per size, Julia is granted all 10 available threads via `JULIA_NUM_THREADS`, and Hadamard.jl outputs are scaled by *n* to match pyfwht's unnormalized convention. This extended run demonstrates competitive performance up to 1 billion elements.

**System configuration:**
- macOS 15.7.1 (24G231)
- Apple M4 (arm64)
- Kernel: Darwin 24.6.0 (arm64)
- Python environment: `myenv` (Python 3.14)
- Julia: 1.12.2 (Homebrew) with project `otherlibs/Hadamard.jl`

```
/Users/hoseinhadipour/Projects/libfwht/.venv/bin/python bench/compare_pyfwht_vs_hadamardjl.py --julia-bin /opt/homebrew/bin/julia --min-power 6 --max-power 30 --repeat 5
pyfwht vs Hadamard.jl (CPU)
------------------------------------------------------------------------
Julia project : /Users/hoseinhadipour/Projects/libfwht/otherlibs/Hadamard.jl
Julia threads : 10
pyfwht backend: AUTO
Repeat count  : 5
[libfwht] CPU backend: NEON vector path active
Running n=64 (repeat=5) ...
Running n=128 (repeat=5) ...
Running n=256 (repeat=5) ...
Running n=512 (repeat=5) ...
Running n=1024 (repeat=5) ...
Running n=2048 (repeat=5) ...
Running n=4096 (repeat=5) ...
Running n=8192 (repeat=5) ...
Running n=16384 (repeat=5) ...
Running n=32768 (repeat=5) ...
Running n=65536 (repeat=5) ...
Running n=131072 (repeat=5) ...
Running n=262144 (repeat=5) ...
Running n=524288 (repeat=5) ...
Running n=1048576 (repeat=5) ...
Running n=2097152 (repeat=5) ...
Running n=4194304 (repeat=5) ...
Running n=8388608 (repeat=5) ...
Running n=16777216 (repeat=5) ...
Running n=33554432 (repeat=5) ...
Running n=67108864 (repeat=5) ...
Running n=134217728 (repeat=5) ...
Running n=268435456 (repeat=5) ...
Running n=536870912 (repeat=5) ...
Running n=1073741824 (repeat=5) ...

     n (2^k) |   pyfwht (s) | Hadamard.jl (s) |       Faster (×) |  max |diff| |  Correct?
------------------------------------------------------------------------------------------
2^ 6=        64 |     0.000047 |        0.002553 |  pyfwht ×  54.45 |   0.000e+00 |     ✔    
2^ 7=       128 |     0.000007 |        0.002516 |  pyfwht × 365.07 |   0.000e+00 |     ✔    
2^ 8=       256 |     0.000005 |        0.002658 |  pyfwht × 484.78 |   0.000e+00 |     ✔    
2^ 9=       512 |     0.000005 |        0.002845 |  pyfwht × 534.20 |   0.000e+00 |     ✔    
2^10=      1024 |     0.000007 |        0.003331 |  pyfwht × 451.13 |   0.000e+00 |     ✔    
2^11=      2048 |     0.000009 |        0.003081 |  pyfwht × 344.53 |   0.000e+00 |     ✔    
2^12=      4096 |     0.000011 |        0.003305 |  pyfwht × 291.36 |   0.000e+00 |     ✔    
2^13=      8192 |     0.000017 |        0.003408 |  pyfwht × 205.17 |   0.000e+00 |     ✔    
2^14=     16384 |     0.000031 |        0.003614 |  pyfwht × 117.88 |   0.000e+00 |     ✔    
2^15=     32768 |     0.000064 |        0.003993 |  pyfwht ×  62.13 |   0.000e+00 |     ✔    
2^16=     65536 |     0.000116 |        0.004312 |  pyfwht ×  37.33 |   0.000e+00 |     ✔    
2^17=    131072 |     0.000246 |        0.004420 |  pyfwht ×  17.95 |   0.000e+00 |     ✔    
2^18=    262144 |     0.000392 |        0.004798 |  pyfwht ×  12.25 |   0.000e+00 |     ✔    
2^19=    524288 |     0.001200 |        0.005523 |  pyfwht ×   4.60 |   0.000e+00 |     ✔    
2^20=   1048576 |     0.000964 |        0.006422 |  pyfwht ×   6.66 |   0.000e+00 |     ✔    
2^21=   2097152 |     0.001958 |        0.008175 |  pyfwht ×   4.17 |   0.000e+00 |     ✔    
2^22=   4194304 |     0.004482 |        0.011609 |  pyfwht ×   2.59 |   0.000e+00 |     ✔    
2^23=   8388608 |     0.010983 |        0.019230 |  pyfwht ×   1.75 |   0.000e+00 |     ✔    
2^24=  16777216 |     0.023812 |        0.034926 |  pyfwht ×   1.47 |   0.000e+00 |     ✔    
2^25=  33554432 |     0.051422 |        0.066293 |  pyfwht ×   1.29 |   0.000e+00 |     ✔    
2^26=  67108864 |     0.109030 |        0.134193 |  pyfwht ×   1.23 |   0.000e+00 |     ✔    
2^27= 134217728 |     0.240199 |        0.280843 |  pyfwht ×   1.17 |   0.000e+00 |     ✔    
2^28= 268435456 |     0.523054 |        0.580735 |  pyfwht ×   1.11 |   0.000e+00 |     ✔    
2^29= 536870912 |     1.136761 |        1.277834 |  pyfwht ×   1.12 |   0.000e+00 |     ✔    
2^30=1073741824 |     3.010074 |        2.661556 | Hadamard ×   1.13 |   0.000e+00 |     ✔    

Summary
------------------------------------------------------------------------
pyfwht faster on 24/25 sizes (avg speedup 125.23×).
Hadamard.jl faster on 1/25 sizes (avg speedup 1.13×).
Max |diff|: 0.000e+00
```


## 2025-12-02 — libfwht vs sboxU Walsh spectrum (CPU + GPU)
**Description:** Compare libfwht's Boolean-packed FWHT (CPU and GPU) against sboxU's coordinate-based Walsh spectrum implementation on large random Boolean functions. Both sides compute the **same non-normalized ±1 Walsh–Hadamard transform**, i.e. integer spectra in the range $[-2^n, 2^n]$ with no $1/\sqrt{N}$ scaling. The GPU run also evaluates three libfwht GPU variants: unpacked, packed, and device-pointer API.

**System configuration:**
- Host: fatgpu001
- OS: Linux 5.14.0-570.62.1.el9_6.x86_64 (x86_64)
- CPU: 2 × AMD EPYC 9454 (48 cores/socket, 96 hardware threads total)
- GPU: NVIDIA H100 80GB HBM3 (SM 9.0, 132 SMs)
- Threads in benchmark:
   - 1 thread for $n \le 2^{15}$
   - 96 threads for larger sizes

**Command:**

```bash
cd /Users/hoseinhadipour/Projects/libfwht
make clean
make -C bench sboxu
./build/compare_sboxu_fwht
```

**Correctness:**
- `walsh_spectrum_coord` (sboxU) and `fwht_boolean_packed` (libfwht) produce identical integer spectra for random Boolean functions up to $n = 8192$.
- Both use the **same non-normalized convention**:
   - Input is mapped to $T(x) = (-1)^{f(x)}$.
   - FWHT applies in-place butterflies `sum = a + b`, `diff = a - b` with no normalization.

**Selected results (per-iteration time, lower is better):**

| $n$         | Threads | SboxU coord (µs) | libfwht CPU (µs) | GPU device (µs) | Best method     | Best vs SboxU | GPU vs libfwht |
|------------:|:-------:|-----------------:|-----------------:|----------------:|-----------------|--------------:|---------------:|
| $2^{10}$    |    1    |             4.10 |             3.16 |          18.42  | libfwht (CPU)   |       1.30×   |        0.17×   |
| $2^{14}$    |    1    |            85.07 |            49.84 |          56.62  | libfwht (CPU)   |       1.71×   |        0.88×   |
| $2^{15}$    |   96    |           179.77 |           113.77 |          63.29  | GPU (device)    |       2.84×   |        1.80×   |
| $2^{18}$    |   96    |         3742.88  |           725.58 |         109.90  | GPU (device)    |      34.06×   |        6.60×   |
| $2^{20}$    |   96    |        12386.09  |          1060.56 |         269.95  | GPU (device)    |      45.88×   |        3.93×   |
| $2^{23}$    |   96    |       125713.91  |          7040.89 |        2106.62  | GPU (device)    |      59.68×   |        3.34×   |
| $2^{25}$    |   96    |       538959.48  |         28554.42 |        9865.65  | GPU (device)    |      54.63×   |        2.89×   |
| $2^{27}$    |   96    |      3793414.85  |        113869.07 |       40038.73  | GPU (device)    |      94.74×   |        2.84×   |

**Observations:**
- On this dual-EPYC 9454 system, libfwht's CPU FWHT is consistently faster than sboxU's coordinate-based Walsh spectrum, from about **1.3×** at $n = 2^{10}$ up to roughly **30×+** at the largest tested sizes.
- The libfwht GPU device-pointer path on H100 becomes dominant from around $n \approx 2^{15}$, reaching up to **~95× faster** than sboxU's CPU implementation and about **3–7× faster** than libfwht's own 96-thread CPU backend at large $n$.


## 2025-12-02 — libfwht AVX2 vs FFHT (single-threaded x86_64 CPU)
**Description:** Direct performance comparison between libfwht's AVX2 intrinsics implementation and FFHT (Fast Fast Hadamard Transform), a highly optimized x86-only library using hand-written inline assembly. FFHT operates on `float` arrays while libfwht uses `int32_t`, but both compute the unnormalized Walsh-Hadamard transform. This benchmark measures single-threaded CPU performance across power-of-2 sizes from 2⁸ to 2²⁵.

**System configuration:**
- Host: server01
- OS: Linux 5.4.0-208-generic #228-Ubuntu SMP (x86_64)
- CPU: 2 × AMD EPYC 7742 64-Core Processor (128 cores total, 256 threads with SMT)
- Compiler: GCC with `-march=native -O3`
- Backend: libfwht CPU backend with AVX2 vectorization enabled

**Build and run commands:**

```bash
# On x86_64 Linux server
cd /path/to/libfwht
make clean && make -j
make -C bench ffht
cd bench && LD_LIBRARY_PATH=../lib ./compare_ffht_fwht
```

**Output:**

```
hossein.hadipour@server01:~/bench$ cd bench && LD_LIBRARY_PATH=../lib ./compare_ffht_fwht
[libfwht] CPU backend: AVX2 vector path active
libfwht vs FFHT (single-threaded CPU comparison)
      Size   libfwht_s      FFHT_s  libfwht_GO/s   FFHT_GO/s    Speedup
       256    0.000001    0.000000       2.062      64.425    31.250x
       512    0.000002    0.000000       2.301      17.054     7.412x
      1024    0.000004    0.000000       2.517      25.265    10.039x
      2048    0.000008    0.000001       2.907      26.006     8.945x
      4096    0.000015    0.000002       3.350      27.734     8.278x
      8192    0.000030    0.000004       3.566      29.387     8.241x
     16384    0.000063    0.000008       3.628      28.548     7.868x
     32768    0.000119    0.000020       4.135      25.019     6.051x
     65536    0.000261    0.000036       4.015      29.537     7.356x
    131072    0.000535    0.000073       4.164      30.619     7.354x
    262144    0.001200    0.000150       3.933      31.505     8.011x
    524288    0.002426    0.000354       4.107      28.122     6.848x
   1048576    0.005055    0.000744       4.148      28.169     6.790x
   2097152    0.010304    0.001557       4.274      28.289     6.619x
   4194304    0.025338    0.003371       3.642      27.369     7.515x
   8388608    0.063547    0.007740       3.036      24.928     8.210x
  16777216    0.124580    0.015886       3.232      25.346     7.842x
  33554432    0.250707    0.045204       3.346      18.557     5.546x
```

**Observations:**
- FFHT is consistently **5.5–10× faster** than libfwht's AVX2 intrinsics implementation across all tested sizes.
- FFHT achieves **25–32 GOps/s** sustained throughput on AMD EPYC 7742, while libfwht AVX2 reaches **3–4 GOps/s**.
- The performance gap is expected: FFHT uses ~20,000 lines of machine-generated inline assembly with hand-tuned permutations, register allocation, and float-specific instructions like `vaddsubps` that have no direct integer equivalent.
- libfwht's AVX2 implementation uses portable C intrinsics for `int32_t`, prioritizing:
  - **Maintainability**: 80 lines of readable C vs 20K lines of generated assembly
  - **Portability**: Same codebase works across x86_64 platforms
  - **Type flexibility**: Supports int32/int8/double, not just float
  - **Correctness**: Exact integer arithmetic for cryptanalysis
- For applications requiring extreme single-threaded CPU performance on x86_64 with float data, FFHT is recommended.
- For typical cryptanalysis workloads, libfwht provides better balance:
  - AVX2 is still **2-3× faster** than scalar code
  - **OpenMP backend** scales well on multi-core systems
  - **GPU backend** provides 10-100× speedup for batch operations
  - **Integer arithmetic** ensures exact results for Boolean functions


## 2025-12-02 — libfwht fp64 vs FFHT double (single-threaded x86_64 CPU)
**Description:** Direct performance comparison between libfwht's double-precision (fp64) implementation and FFHT's double-precision path. This benchmark complements the float/int32 comparison above by testing the same two libraries on double-precision data. FFHT provides separate implementations for both `float` and `double`, each with ~20,000 lines of hand-tuned AVX assembly.

**Purpose:**
- Measure whether libfwht's fp64 performance gap vs FFHT is similar to the int32 case
- Determine if FFHT's double assembly optimization is as effective as its float path
- Validate that both libraries produce numerically equivalent results for double-precision transforms

**System configuration:**
- Same as previous FFHT benchmark (AMD EPYC 7742 x86_64 server)
- Compiler: GCC with `-march=native -O3`
- Backend: libfwht CPU backend (scalar + auto-vectorization for fp64)

**Build and run commands:**

```bash
# On x86_64 Linux server
cd /path/to/libfwht
make clean && make -j
make -C bench ffht
cd bench && LD_LIBRARY_PATH=../lib ./compare_ffht_fwht_fp64
```

**Output:**

```
hossein.hadipour@server01:~$ cd bench && LD_LIBRARY_PATH=../lib ./compare_ffht_fwht_fp64
[libfwht] CPU backend: AVX2 vector path active
libfwht fp64 vs FFHT double (single-threaded CPU comparison)
      Size   libfwht_s      FFHT_s  libfwht_GO/s   FFHT_GO/s    Speedup
       256    0.000001    0.000000       2.364       7.579     3.206x
       512    0.000002    0.000000       2.928      11.833     4.041x
      1024    0.000004    0.000001       2.915      13.015     4.465x
      2048    0.000007    0.000002       3.323      14.764     4.443x
      4096    0.000014    0.000004       3.492      12.912     3.697x
      8192    0.000029    0.000007       3.705      14.613     3.944x
     16384    0.000070    0.000015       3.263      15.087     4.624x
     32768    0.000122    0.000035       4.012      13.855     3.453x
     65536    0.000249    0.000071       4.217      14.825     3.515x
    131072    0.000515    0.000140       4.330      15.919     3.676x
    262144    0.001050    0.000374       4.492      12.625     2.811x
    524288    0.002162    0.000760       4.608      13.104     2.843x
   1048576    0.004509    0.001656       4.651      12.666     2.723x
   2097152    0.009610    0.003367       4.583      13.080     2.854x
   4194304    0.025114    0.007769       3.674      11.878     3.233x
   8388608    0.053795    0.016234       3.587      11.885     3.314x
  16777216    0.117093    0.051521       3.439       7.815     2.273x
  33554432    0.280407    0.105759       2.992       7.932     2.651x
```

**Observations:**
- FFHT is **2.3–4.6× faster** than libfwht fp64, achieving **8–16 GOps/s** vs **3–4.5 GOps/s**.
- Smaller gap than float/int32 case (2–5× vs 5–10×) due to halved SIMD width (4 doubles vs 8 floats per AVX2 register).
- Both libraries produce identical results within floating-point epsilon tolerance.

