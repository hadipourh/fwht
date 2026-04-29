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


## 2026-04-18 — pyfwht vs Hadamard.jl (CPU) — Extended to 2³⁰
**Description:** Direct comparison between pyfwht's AUTO backend (OpenMP-optimized) and the Julia Hadamard.jl implementation for the same random inputs (2⁶…2³⁰). Both sides run 5 repetitions per size, Julia is granted all 10 available threads via `JULIA_NUM_THREADS`, and Hadamard.jl outputs are scaled by *n* to match pyfwht's unnormalized convention. This section was rerun after the OpenMP-path fixes in libfwht to refresh the CPU benchmark with current behavior.

**System configuration:**
- macOS 15.7.1 (24G231)
- Apple M4 (arm64)
- Kernel: Darwin 24.6.0 (arm64)
- Python environment: `.venv` (Python 3.14)
- Julia: 1.12.2 (Homebrew) with project `otherlibs/Hadamard.jl`

```
/Users/hoseinhadipour/Projects/libfwht/.venv/bin/python bench/compare_pyfwht_vs_hadamardjl.py --julia-bin /opt/homebrew/bin/julia --min-power 6 --max-power 30 --repeat 5
[libfwht] CPU backend: NEON vector path active
pyfwht vs Hadamard.jl (CPU)
------------------------------------------------------------------------
Julia project : /Users/hoseinhadipour/Projects/libfwht/otherlibs/Hadamard.jl
Julia threads : 10
pyfwht backend: AUTO
Repeat count  : 5
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
2^ 6=        64 |     0.000066 |        0.002451 |  pyfwht ×  37.30 |   0.000e+00 |     ✔    
2^ 7=       128 |     0.000008 |        0.002520 |  pyfwht × 315.93 |   0.000e+00 |     ✔    
2^ 8=       256 |     0.000006 |        0.002762 |  pyfwht × 461.02 |   0.000e+00 |     ✔    
2^ 9=       512 |     0.000007 |        0.002814 |  pyfwht × 429.68 |   0.000e+00 |     ✔    
2^10=      1024 |     0.000007 |        0.002969 |  pyfwht × 444.29 |   0.000e+00 |     ✔    
2^11=      2048 |     0.000008 |        0.003052 |  pyfwht × 400.74 |   0.000e+00 |     ✔    
2^12=      4096 |     0.000010 |        0.003225 |  pyfwht × 312.09 |   0.000e+00 |     ✔    
2^13=      8192 |     0.000017 |        0.003347 |  pyfwht × 199.29 |   0.000e+00 |     ✔    
2^14=     16384 |     0.000031 |        0.003501 |  pyfwht × 112.69 |   0.000e+00 |     ✔    
2^15=     32768 |     0.000060 |        0.003754 |  pyfwht ×  63.02 |   0.000e+00 |     ✔    
2^16=     65536 |     0.000122 |        0.004011 |  pyfwht ×  32.89 |   0.000e+00 |     ✔    
2^17=    131072 |     0.000242 |        0.004257 |  pyfwht ×  17.56 |   0.000e+00 |     ✔    
2^18=    262144 |     0.000405 |        0.004656 |  pyfwht ×  11.49 |   0.000e+00 |     ✔    
2^19=    524288 |     0.000585 |        0.005272 |  pyfwht ×   9.01 |   0.000e+00 |     ✔    
2^20=   1048576 |     0.000962 |        0.006275 |  pyfwht ×   6.53 |   0.000e+00 |     ✔    
2^21=   2097152 |     0.001915 |        0.007891 |  pyfwht ×   4.12 |   0.000e+00 |     ✔    
2^22=   4194304 |     0.005022 |        0.011950 |  pyfwht ×   2.38 |   0.000e+00 |     ✔    
2^23=   8388608 |     0.012731 |        0.019199 |  pyfwht ×   1.51 |   0.000e+00 |     ✔    
2^24=  16777216 |     0.028820 |        0.035726 |  pyfwht ×   1.24 |   0.000e+00 |     ✔    
2^25=  33554432 |     0.063132 |        0.068104 |  pyfwht ×   1.08 |   0.000e+00 |     ✔    
2^26=  67108864 |     0.137383 |        0.139017 |  pyfwht ×   1.01 |   0.000e+00 |     ✔    
2^27= 134217728 |     0.295287 |        0.297419 |  pyfwht ×   1.01 |   0.000e+00 |     ✔    
2^28= 268435456 |     0.621371 |        0.637335 |  pyfwht ×   1.03 |   0.000e+00 |     ✔    
2^29= 536870912 |     1.367685 |        1.417856 |  pyfwht ×   1.04 |   0.000e+00 |     ✔    
2^30=1073741824 |     5.676305 |        4.155913 | Hadamard ×   1.37 |   0.000e+00 |     ✔    

Summary
------------------------------------------------------------------------
pyfwht faster on 24/25 sizes (avg speedup 119.50×).
Hadamard.jl faster on 1/25 sizes (avg speedup 1.37×).
Max |diff|: 0.000e+00
```


## 2026-04-18 — libfwht vs FFTW DHT (CPU, fair dense-WHT comparison)
**Description:** Apple-to-apple comparison between libfwht's dedicated double-precision FWHT and FFTW's best all-real realization of the same dense unnormalized transform: a rank-*n* size-2 `FFTW_DHT` plan reused across iterations. The harness verifies correctness against a direct $O(N^2)$ WHT up to $n = 4096$, reports FFTW planning cost separately from steady-state execution time, and measures both single-thread execution and a forced-OpenMP run where **both** libraries use threading for every size. This avoids the earlier apples-to-oranges artifact where libfwht silently stayed serial for small inputs while FFTW paid full thread-startup overhead.

Three library-side fixes that made the multi-threaded comparison honest and competitive:

1. **Explicit-request semantics**: when a caller explicitly selects `FWHT_BACKEND_OPENMP`, the library now honours the request at any size instead of silently falling back to the single-threaded CPU path for sizes below an internal heuristic threshold.  The threshold remains active for the `AUTO` backend, which is the right engineering trade-off.
2. **Cache-oblivious bootstrap kernel**: the per-tile bootstrap phase inside the stage-parallel path now calls the same recursive cache-oblivious CPU butterfly used by the single-threaded path, instead of a flat iterative loop.  This improves L1/L2 reuse during the bootstrap.
3. **Larger default tile (L2-sized)**: the bootstrap tile was increased from $2^{12}$ doubles (32 KiB, L1-resident) to $2^{17}$ doubles (1 MiB, L2-resident).  Larger tiles mean fewer memory-bound merge stages at the top of the butterfly network, which is where FFTW previously had the edge.

**System configuration:**
- macOS 26.3.1 (build 25D771280a)
- Apple M4 (arm64)
- FFTW: 3.3.10_2 (Homebrew)
- libomp: 21.1.6 (Homebrew)
- Maximum thread count used in the benchmark: 10

**Command:**

```bash
cd /Users/hoseinhadipour/Projects/libfwht
make -C bench fftw
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$(pwd)/lib ./bench/compare_fftw_fwht
```

**Memory note:** At $2^{30}$, each runtime uses one 8 GiB double-precision work buffer, so the largest comparison point needs comfortably more than 16 GiB of host memory once FFTW planning and ordinary process overhead are included.

**Selected results:**

| mode | $n$ | threads | libfwht exec (ms) | FFTW exec (ms) | FFTW plan (ms) | winner |
|-----:|----:|--------:|------------------:|---------------:|---------------:|:-------|
| single-thread | $2^{20}$ | 1 | 2.266 | 7.504 | 0.270 | libfwht × 3.31 |
| single-thread | $2^{23}$ | 1 | 21.857 | 66.788 | 0.369 | libfwht × 3.06 |
| single-thread | $2^{28}$ | 1 | 941.417 | 2597.823 | 0.544 | libfwht × 2.76 |
| single-thread | $2^{30}$ | 1 | 4246.796 | 11538.150 | 1.094 | libfwht × 2.72 |
| forced OpenMP | $2^{20}$ | 10 | 0.753 | 1.585 | 3.490 | libfwht × 2.10 |
| forced OpenMP | $2^{23}$ | 10 | 11.163 | 14.426 | 4.261 | libfwht × 1.29 |
| forced OpenMP | $2^{27}$ | 10 | 261.440 | 317.180 | 6.085 | libfwht × 1.21 |
| forced OpenMP | $2^{30}$ | 10 | 2645.266 | 3646.188 | 10.127 | libfwht × 1.38 |

**Observations:**
- The comparison is like-for-like: both libraries compute the same unnormalized dense Walsh-Hadamard transform on the same double-precision inputs, and the logged harness checks exact agreement (within floating-point tolerance) for every benchmarked size.
- On this Apple M4 system, libfwht is consistently faster in single-thread steady-state execution across the full logged sweep $2^8 \ldots 2^{30}$, typically by about **2.7–4.2×**.
- In the forced-OpenMP comparison (both threading at every size), libfwht wins from $2^{15}$ upward, typically by **1.06–2.15×**. FFTW leads by a small margin only for tiny sizes ($2^{10}$–$2^{13}$) where both libraries pay comparable thread-startup overhead.
- The three library-side improvements closed and reversed the gap that earlier runs showed at the large end. The previous version used a 32 KiB iterative bootstrap and silently downgraded small explicit-OPENMP requests to the CPU path; after fixing those issues, libfwht leads at every practically useful size.
- FFTW planning overhead remains small in absolute terms even at the large end, but it is still reported separately because libfwht does not rely on a comparable heavy planning phase.


## 2026-04-18 — libfwht vs sboxU Walsh spectrum (CPU + GPU)
**Description:** Compare libfwht's Boolean-packed FWHT (CPU and GPU) against sboxU's coordinate-based Walsh spectrum implementation on large random Boolean functions. Both sides compute the **same non-normalized ±1 Walsh–Hadamard transform**, i.e. integer spectra in the range $[-2^n, 2^n]$ with no $1/\sqrt{N}$ scaling. The GPU run also evaluates three libfwht GPU variants: unpacked, packed, and device-pointer API. This section was rerun after the OpenMP-path fixes in libfwht to refresh the CPU numbers on the dual-EPYC + H100 system.

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
mkdir -p bench/tmp
make -C bench -f Makefile.sboxu CXX=g++ clean
make -C bench -f Makefile.sboxu CXX=g++ -j
./build/compare_sboxu_fwht | tee bench/tmp/compare_sboxu_fwht_20260418.txt
```

**Correctness:**
- `walsh_spectrum_coord` (sboxU) and `fwht_boolean_packed` (libfwht) produce identical integer spectra for random Boolean functions up to $n = 8192$.
- Both use the **same non-normalized convention**:
   - Input is mapped to $T(x) = (-1)^{f(x)}$.
   - FWHT applies in-place butterflies `sum = a + b`, `diff = a - b` with no normalization.

**Selected results (per-iteration time, lower is better):**

| $n$         | Threads | SboxU coord (µs) | libfwht CPU (µs) | GPU device (µs) | Best method     | Best vs SboxU | GPU vs libfwht |
|------------:|:-------:|-----------------:|-----------------:|----------------:|-----------------|--------------:|---------------:|
| $2^{10}$    |    1    |             4.11 |             2.78 |          18.41  | libfwht (CPU)   |       1.47×   |        0.15×   |
| $2^{14}$    |    1    |            84.94 |            43.84 |          57.86  | libfwht (CPU)   |       1.94×   |        0.76×   |
| $2^{15}$    |    1    |           181.57 |           102.45 |          63.83  | GPU (device)    |       2.84×   |        1.60×   |
| $2^{18}$    |   96    |         3693.63  |           179.07 |         109.83  | GPU (device)    |      33.63×   |        1.63×   |
| $2^{20}$    |   96    |        11937.22  |          1003.55 |         268.62  | GPU (device)    |      44.44×   |        3.74×   |
| $2^{23}$    |   96    |       123623.23  |          6719.55 |        1979.35  | GPU (device)    |      62.46×   |        3.39×   |
| $2^{25}$    |   96    |       529463.21  |         27681.58 |        9410.59  | GPU (device)    |      56.26×   |        2.94×   |
| $2^{27}$    |   96    |      3756961.52  |        110153.20 |       38221.70  | GPU (device)    |      98.29×   |        2.88×   |

**Observations:**
- On this refreshed dual-EPYC 9454 system run, libfwht's CPU FWHT is consistently faster than sboxU's coordinate-based Walsh spectrum, from about **1.4×** at $n = 2^{10}$ up to about **34.1×** at $n = 2^{27}$.
- The libfwht GPU device-pointer path on H100 becomes dominant from around $n \approx 2^{15}$, reaching up to **98.29× faster** than sboxU's CPU implementation and roughly **1.6–3.7× faster** than libfwht's own CPU backend depending on size.


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

