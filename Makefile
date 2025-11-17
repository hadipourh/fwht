# Fast Walsh-Hadamard Transform (FWHT) Library
# Makefile
#
# Copyright (C) 2025 Hosein Hadipour
#
# Author: Hosein Hadipour <hsn.hadipour@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# ============================================================================
# Configuration
# ============================================================================

CC = gcc
CXX = g++
NVCC = nvcc
BASE_CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -pthread -I$(INCLUDE_DIR)
CFLAGS = $(BASE_CFLAGS)
BASE_CXXFLAGS = -O3 -Wall -Wextra -pedantic -pthread -I$(INCLUDE_DIR)
CXXFLAGS = $(BASE_CXXFLAGS)
ifeq ($(NO_SIMD),1)
	CFLAGS +=
	CXXFLAGS +=
else
	CFLAGS += -march=native
	CXXFLAGS += -march=native
endif
NVCCFLAGS = -O3 -I$(INCLUDE_DIR) --compiler-options -fPIC -std=c++14
LDFLAGS =

# Directories
SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests
BUILD_DIR = build
LIB_DIR = lib
TOOLS_DIR = tools
BENCH_DIR = bench
EXAMPLES_DIR = examples

# Output
LIB_NAME = libfwht
STATIC_LIB = $(LIB_DIR)/$(LIB_NAME).a
SHARED_LIB = $(LIB_DIR)/$(LIB_NAME).so
TEST_BIN = $(BUILD_DIR)/test_correctness
CLI_SRC = $(TOOLS_DIR)/fwht_cli.c
CLI_BIN = $(BUILD_DIR)/fwht_cli
EXAMPLE_SRC = $(EXAMPLES_DIR)/example_basic.c
EXAMPLE_BIN = $(EXAMPLES_DIR)/example_basic
EXAMPLE2_SRC = $(EXAMPLES_DIR)/example_boolean_packed.c
EXAMPLE2_BIN = $(EXAMPLES_DIR)/example_boolean_packed

# Source files (CPU)
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# CUDA source files
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))

# Test files
TEST_SRCS = $(TEST_DIR)/test_correctness.c
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

# ============================================================================
# Platform Detection
# ============================================================================

UNAME_S := $(shell uname -s)

# Detect CUDA availability
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)
ifdef CUDA_AVAILABLE
    HAS_CUDA = 1
else
    HAS_CUDA = 0
endif

ifeq ($(UNAME_S),Darwin)
    # macOS
    SHARED_FLAGS = -dynamiclib
    SHARED_EXT = .dylib
    # OpenMP on macOS (Homebrew)
    ifneq ($(NO_OPENMP),1)
        OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
        OPENMP_CXXFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
        OPENMP_LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
        # Apply OpenMP flags by default
        CFLAGS += $(OPENMP_CFLAGS)
        CXXFLAGS += $(OPENMP_CXXFLAGS)
        LDFLAGS += $(OPENMP_LDFLAGS)
    endif
else ifeq ($(UNAME_S),Linux)
    # Linux
    SHARED_FLAGS = -shared -fPIC
    SHARED_EXT = .so
    # OpenMP on Linux
    ifneq ($(NO_OPENMP),1)
        OPENMP_CFLAGS = -fopenmp
        OPENMP_CXXFLAGS = -fopenmp
        OPENMP_LDFLAGS = -fopenmp
        # Apply OpenMP flags by default
        CFLAGS += $(OPENMP_CFLAGS)
        CXXFLAGS += $(OPENMP_CXXFLAGS)
        LDFLAGS += $(OPENMP_LDFLAGS)
    endif
endif

# CUDA configuration
ifeq ($(HAS_CUDA),1)
    ifndef NO_CUDA
        USE_CUDA = 1
        CFLAGS += -DUSE_CUDA
        CXXFLAGS += -DUSE_CUDA
		NVCCFLAGS += -DUSE_CUDA
        CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
        # Try to detect CUDA path
        CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
        ifneq ($(CUDA_PATH),)
            CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart
        endif
    endif
endif

# ============================================================================
# Build Targets
# ============================================================================

.PHONY: all clean test install lib static shared directories bench cli

all: directories lib test

# Convenience alias (common typo)
example: examples

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)

# Build both static and shared libraries
lib: static shared

# Build command-line utility
cli: directories lib $(CLI_BIN)

# Static library
static: directories $(STATIC_LIB)

ifeq ($(USE_CUDA),1)
$(STATIC_LIB): $(OBJS) $(CUDA_OBJS)
	@echo "Creating static library (with CUDA): $@"
	ar rcs $@ $^
else
$(STATIC_LIB): $(OBJS)
	@echo "Creating static library: $@"
	ar rcs $@ $^
endif

# Shared library
shared: directories $(SHARED_LIB)

ifeq ($(USE_CUDA),1)
$(SHARED_LIB): $(OBJS) $(CUDA_OBJS)
	@echo "Creating shared library (with CUDA): $@"
	$(CC) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
else
$(SHARED_LIB): $(OBJS)
	@echo "Creating shared library: $@"
	$(CC) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS)

$(CLI_BIN): $(CLI_SRC) $(SHARED_LIB)
	@echo "Building CLI tool: $@"
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lfwht -lm -o $@ -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
endif

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Compile CUDA files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	@echo "Compiling test: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Build test executable
test: directories $(STATIC_LIB) $(TEST_OBJS)
	@echo "Building test suite..."
ifeq ($(USE_CUDA),1)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(CUDA_OBJS) -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(TEST_BIN) $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
else
	$(CC) $(CFLAGS) $(TEST_OBJS) -L$(LIB_DIR) -lfwht -o $(TEST_BIN) $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
endif
	@echo "Running CPU tests..."
	@$(TEST_BIN)
ifeq ($(USE_CUDA),1)
	@echo ""
	@echo "Building GPU test suite..."
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_gpu.c $(CUDA_OBJS) -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(BUILD_DIR)/test_gpu -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
	@echo "Running GPU tests..."
	@$(BUILD_DIR)/test_gpu
endif

# Build benchmark executable
bench: directories $(STATIC_LIB)
ifeq ($(USE_CUDA),1)
	@echo "Building benchmark (CUDA enabled)..."
	$(CXX) $(CXXFLAGS) $(BENCH_DIR)/fwht_bench.c -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) $(LDFLAGS) -lm -o $(BUILD_DIR)/fwht_bench -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
else
	@echo "Building benchmark..."
	$(CC) $(CFLAGS) $(BENCH_DIR)/fwht_bench.c -L$(LIB_DIR) -lfwht $(LDFLAGS) -lm -o $(BUILD_DIR)/fwht_bench -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
endif
	@echo "Run with: ./build/fwht_bench [options]"

# Build example programs
examples: directories lib $(EXAMPLE_BIN) $(EXAMPLE2_BIN)
	@echo "Example binaries available in $(EXAMPLES_DIR)/"

$(EXAMPLE_BIN): $(EXAMPLE_SRC) $(STATIC_LIB)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lfwht -lm -o $@ -Wl,-rpath,$(CURDIR)/$(LIB_DIR)

$(EXAMPLE2_BIN): $(EXAMPLE2_SRC) $(STATIC_LIB)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lfwht -lm -o $@ -Wl,-rpath,$(CURDIR)/$(LIB_DIR)

# Build and run GPU-specific tests (only if CUDA available)
test-gpu: lib
ifeq ($(USE_CUDA),1)
	@echo "Building GPU test suite..."
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_gpu.c $(CUDA_OBJS) -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(BUILD_DIR)/test_gpu -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
	@echo "Running GPU tests..."
	@$(BUILD_DIR)/test_gpu
else
	@echo "GPU tests skipped (CUDA not available)"
endif

# Run only CPU tests
test-cpu: directories $(STATIC_LIB) $(TEST_OBJS)
	@echo "Building CPU test suite..."
ifeq ($(USE_CUDA),1)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(TEST_BIN) $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
else
	$(CC) $(CFLAGS) $(TEST_OBJS) -L$(LIB_DIR) -lfwht -o $(TEST_BIN) $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
endif
	@echo "Running CPU tests..."
	@$(TEST_BIN)

# Build and run batch FWHT tests
test-batch: directories lib
	@echo "Building batch test suite..."
ifeq ($(USE_CUDA),1)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_batch.c -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(BUILD_DIR)/test_batch $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
else
	$(CC) $(CFLAGS) $(TEST_DIR)/test_batch.c -L$(LIB_DIR) -lfwht -o $(BUILD_DIR)/test_batch $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
endif
	@echo "Running batch tests..."
	@$(BUILD_DIR)/test_batch

# Build and run GPU callback tests
test-gpu-callbacks: directories lib
	@echo "Building GPU callback test suite..."
ifeq ($(USE_CUDA),1)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_gpu_callbacks.c -L$(LIB_DIR) -lfwht $(CUDA_LDFLAGS) -o $(BUILD_DIR)/test_gpu_callbacks $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
	@echo "Running GPU callback tests..."
	@$(BUILD_DIR)/test_gpu_callbacks
else
	$(CC) $(CFLAGS) $(TEST_DIR)/test_gpu_callbacks.c -L$(LIB_DIR) -lfwht -o $(BUILD_DIR)/test_gpu_callbacks $(LDFLAGS) -lm -Wl,-rpath,$(CURDIR)/$(LIB_DIR)
	@echo "Running GPU callback tests (CUDA features disabled)..."
	@$(BUILD_DIR)/test_gpu_callbacks
endif

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(LIB_DIR)
	rm -f $(EXAMPLE_BIN) $(EXAMPLE2_BIN)

# Install library (requires sudo on most systems)
install: lib
	@echo "Installing library to /usr/local..."
	install -d /usr/local/lib
	install -m 644 $(STATIC_LIB) /usr/local/lib/
	install -m 755 $(SHARED_LIB) /usr/local/lib/
	install -d /usr/local/include
	install -m 644 $(INCLUDE_DIR)/fwht.h /usr/local/include/
	@echo "Installation complete!"

# Uninstall
uninstall:
	@echo "Uninstalling library..."
	rm -f /usr/local/lib/$(LIB_NAME).a
	rm -f /usr/local/lib/$(LIB_NAME).so
	rm -f /usr/local/lib/$(LIB_NAME).dylib
	rm -f /usr/local/include/fwht.h
	@echo "Uninstallation complete!"

# ============================================================================
# Development Targets
# ============================================================================

# Build with debug symbols
debug: CFLAGS += -g -O0 -DDEBUG
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: all

# Build with OpenMP support
openmp: CFLAGS += $(OPENMP_CFLAGS)
openmp: CXXFLAGS += $(OPENMP_CXXFLAGS)
openmp: LDFLAGS += $(OPENMP_LDFLAGS)
openmp: all

# Build with address sanitizer (debugging)
asan: CFLAGS += -fsanitize=address -g
asan: CXXFLAGS += -fsanitize=address -g
asan: LDFLAGS += -fsanitize=address
asan: all

# Format code (requires clang-format)
format:
	@echo "Formatting code..."
	clang-format -i $(SRC_DIR)/*.c $(INCLUDE_DIR)/*.h $(TEST_DIR)/*.c

# Check for memory leaks with valgrind
memcheck: test
	@echo "Running memory check..."
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_BIN)

# Show build configuration
info:
	@echo "Build Configuration:"
	@echo "  Platform: $(UNAME_S)"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  SIMD: $(if $(NO_SIMD),disabled (NO_SIMD=1),auto -march=native)"
	@echo "  OpenMP: $(if $(OPENMP_CFLAGS),enabled,disabled)"
	@echo "  CUDA: $(if $(USE_CUDA),enabled ($(shell which nvcc)),disabled)"

# Help target
help:
	@echo "FWHT Library Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build library and run tests (default)"
	@echo "  lib       - Build both static and shared libraries"
	@echo "  static    - Build static library only"
	@echo "  shared    - Build shared library only"
	@echo "  test      - Build and run test suite"
	@echo "  cli       - Build the fwht_cli command-line tool"
	@echo "  examples  - Build example programs under $(EXAMPLES_DIR)/"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install library to /usr/local (requires sudo)"
	@echo "  uninstall - Remove installed library"
	@echo ""
	@echo "Development:"
	@echo "  debug     - Build with debug symbols"
	@echo "  asan      - Build with address sanitizer"
	@echo "  format    - Format source code"
	@echo "  memcheck  - Check for memory leaks"
	@echo "  info      - Show build configuration"
	@echo ""
	@echo "Options:"
	@echo "  NO_OPENMP=1  - Disable OpenMP support (enabled by default)"
	@echo "  NO_SIMD=1    - Disable auto SIMD optimization flags"
	@echo "  NO_CUDA=1    - Disable CUDA support"
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Build with OpenMP (default)"
	@echo "  make bench            # Build benchmark tool"
	@echo "  make NO_OPENMP=1      # Build without OpenMP"
	@echo "  make NO_SIMD=1        # Build without auto SIMD flags"
	@echo "  make debug            # Debug build"
	@echo "  sudo make install     # Install library"
