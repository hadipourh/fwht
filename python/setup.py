"""
Build configuration for pyfwht Python package.

This setup.py compiles the C library and creates Python bindings via pybind11.
Note: C files (.c) are compiled as C++ to avoid flag conflicts.
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Paths relative to python/ directory
PYTHON_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = PYTHON_DIR.parent  # Go up to libfwht/
INCLUDE_DIR = PROJECT_ROOT / "include"
SRC_DIR = PROJECT_ROOT / "src"

# Validate paths
if not INCLUDE_DIR.exists():
    raise RuntimeError(f"C library include directory not found: {INCLUDE_DIR}")
if not SRC_DIR.exists():
    raise RuntimeError(f"C library source directory not found: {SRC_DIR}")

# Use a C++ wrapper file that includes the C sources
# This avoids compiler flag conflicts between C and C++ files
wrapper_sources = [
    str(PYTHON_DIR / "src" / "fwht_wrapper.cpp"),
]

# Verify wrapper exists
for src in wrapper_sources:
    if not Path(src).exists():
        raise RuntimeError(f"Wrapper file not found: {src}")
        
# Verify the actual C sources exist (they're included by the wrapper)
c_sources_check = [
    str(SRC_DIR / "fwht_core.c"),
    str(SRC_DIR / "fwht_backend.c"),
    str(SRC_DIR / "fwht_batch.c"),
]
for src in c_sources_check:
    if not Path(src).exists():
        raise RuntimeError(f"C source file not found: {src}")

# Check for CUDA availability
def has_cuda():
    """Check if CUDA compiler (nvcc) is available."""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

# Check for OpenMP availability
def has_openmp():
    """Check if OpenMP is available (simple heuristic)."""
    # On most Unix systems with GCC/Clang, OpenMP is available
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        return True
    return False

# Detect features
# Auto-enable CUDA if nvcc is available (unless explicitly disabled)
use_cuda_env = os.environ.get('USE_CUDA', 'auto').lower()
if use_cuda_env == '0' or use_cuda_env == 'false' or use_cuda_env == 'no':
    cuda_available = False
elif use_cuda_env == '1' or use_cuda_env == 'true' or use_cuda_env == 'yes':
    cuda_available = has_cuda()
else:  # 'auto' or not set - auto-detect
    cuda_available = has_cuda()

openmp_available = has_openmp()

print("=" * 70)
print("pyfwht build configuration")
print("=" * 70)
print(f"Python package directory: {PYTHON_DIR}")
print(f"C library include: {INCLUDE_DIR}")
print(f"C library sources: {SRC_DIR}")
print(f"OpenMP support: {'YES' if openmp_available else 'NO'}")
print(f"CUDA support: {'YES' if cuda_available else 'NO (set USE_CUDA=1 to enable)'}")
print("=" * 70)

# Compiler and linker flags - use C++ flags as base
extra_compile_args = ['-O3', '-std=c++17', '-fPIC']
extra_link_args = []
define_macros = []
include_dirs_list = [str(INCLUDE_DIR), str(PYTHON_DIR / "include")]  # Add DLPack include path

# Platform-specific optimizations
if sys.platform == 'darwin':  # macOS
    # Avoid -march=native on macOS CI runners (causes issues with newer Apple Silicon)
    # Use -mtune=native for safe optimization without breaking compatibility
    if os.environ.get('CI'):
        # On CI, use conservative flags
        extra_compile_args.append('-mtune=generic')
    else:
        # Local builds can use aggressive optimization
        extra_compile_args.append('-march=native')
    extra_compile_args.append('-stdlib=libc++')
elif sys.platform.startswith('linux'):  # Linux
    # On Linux CI, be conservative with march
    if os.environ.get('CI'):
        extra_compile_args.append('-mtune=generic')
    else:
        extra_compile_args.append('-march=native')

# OpenMP support
if openmp_available:
    if sys.platform == 'darwin':
        # macOS with clang might need special handling
        # Try to use libomp if available
        extra_compile_args.append('-Xpreprocessor')
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-lomp')
        # Add libomp include and lib paths
        libomp_prefix = '/opt/homebrew/opt/libomp'  # Homebrew on Apple Silicon
        if Path(libomp_prefix).exists():
            include_dirs_list.append(f'{libomp_prefix}/include')
            extra_link_args.append(f'-L{libomp_prefix}/lib')
    elif sys.platform.startswith('linux'):
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
    
    print("Enabled OpenMP support")

# CUDA support
cuda_objects = []
if cuda_available:
    define_macros.append(('USE_CUDA', '1'))
    
    # Find CUDA paths
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        # Try common locations
        for path in ['/usr/local/cuda', '/opt/cuda']:
            if Path(path).exists():
                cuda_home = path
                break
    
    if cuda_home:
        cuda_include = Path(cuda_home) / 'include'
        cuda_lib = Path(cuda_home) / 'lib64'
        if not cuda_lib.exists():
            cuda_lib = Path(cuda_home) / 'lib'
        
        include_dirs_list.append(str(cuda_include))
        extra_link_args.extend([f'-L{cuda_lib}', '-lcudart'])
        
        # Compile CUDA source files with nvcc
        cuda_sources = ["fwht_cuda.cu", "fwht_cuda_lat.cu"]
        cuda_objects = []
        extra_nvcc = os.environ.get('FWHT_NVCC_FLAGS', '').strip()

        for cuda_source in cuda_sources:
            cuda_src = SRC_DIR / cuda_source
            cuda_obj = PYTHON_DIR / "build" / (cuda_source.replace('.cu', '.o'))
            cuda_obj.parent.mkdir(exist_ok=True)

            print(f"Compiling CUDA source: {cuda_src}")
            nvcc_cmd = [
                'nvcc',
                '-c',
                str(cuda_src),
                '-o', str(cuda_obj),
                f'-I{INCLUDE_DIR}',
                '-DUSE_CUDA=1',
                '-Xcompiler', '-fPIC',
                '--std=c++17',
                '-arch=sm_80',
                '-O3'
            ]
            if extra_nvcc:
                nvcc_cmd.extend(extra_nvcc.split())
            print('NVCC command:', ' '.join(nvcc_cmd))

            result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if result.stdout:
                    print("NVCC stdout:\n" + result.stdout)
                if result.stderr:
                    print("NVCC stderr:\n" + result.stderr)
                raise RuntimeError(f"Failed to compile CUDA source {cuda_source}")

            cuda_objects.append(str(cuda_obj))
            print(f"CUDA object file: {cuda_obj}")

        print("Enabled CUDA support")
    else:
        print("Warning: CUDA_HOME not found, disabling CUDA support")
        cuda_available = False

# Define extension module (use relative paths for sources)
ext_modules = [
    Pybind11Extension(
        "pyfwht._pyfwht",
        sources=[
            "src/bindings.cpp",
            "src/fwht_wrapper.cpp",
        ],
        include_dirs=include_dirs_list,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=cuda_objects,
        define_macros=define_macros,
        language='c++',
    ),
]

# Run setup
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["pyfwht"],
    package_dir={"pyfwht": "pyfwht"},
)
