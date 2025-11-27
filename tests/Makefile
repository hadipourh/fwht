# Helper Makefile to build tests/compare_sboxu_fwht.cpp with platform-specific
# OpenMP settings. Usage:
#   make -C tests                  # build using defaults
#   make -C tests LIBOMP_LIBDIR=... LIBOMP_INCDIR=...  # override libomp paths

ROOT_DIR := ..
BUILD_DIR := $(ROOT_DIR)/build
TARGET := $(BUILD_DIR)/compare_sboxu_fwht
PLATFORM_FILE := $(BUILD_DIR)/.platform
FWHT_STATIC := $(ROOT_DIR)/lib/libfwht.a

COMPARE_SRC := $(ROOT_DIR)/tests/compare_sboxu_fwht.cpp
SBOXU_DIR := $(ROOT_DIR)/sboxU/sboxU/sboxU_cython
SBOXU_SRCS := \
	$(SBOXU_DIR)/sboxu_cpp_diff_lin.cpp \
	$(SBOXU_DIR)/sboxu_cpp_utils.cpp

SOURCES := $(COMPARE_SRC) $(SBOXU_SRCS)

CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3
INCFLAGS := -I$(ROOT_DIR)/include -I$(SBOXU_DIR)

CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)
ifdef CUDA_AVAILABLE
	CXXFLAGS += -DUSE_CUDA
	CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart
	CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
	ifneq ($(CUDA_PATH),)
	CUDA_LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart
	endif
else
	CUDA_LDFLAGS :=
endif

UNAME_S := $(shell uname -s)
PREV_PLATFORM := $(strip $(shell cat $(PLATFORM_FILE) 2>/dev/null))
ifeq ($(PREV_PLATFORM),)
PLATFORM_MISMATCH := 1
else ifneq ($(PREV_PLATFORM),$(UNAME_S))
PLATFORM_MISMATCH := 1
else
PLATFORM_MISMATCH := 0
endif

ifeq ($(UNAME_S),Darwin)
LIBOMP_INCDIR ?= /opt/homebrew/opt/libomp/include
LIBOMP_LIBDIR ?= /opt/homebrew/opt/libomp/lib
OMP_CFLAGS := -Xpreprocessor -fopenmp
else
LIBOMP_INCDIR ?=
LIBOMP_LIBDIR ?=
OMP_CFLAGS := -fopenmp
endif

LINUX_GXX := $(and $(findstring Linux,$(UNAME_S)),$(findstring g++,$(notdir $(CXX))))

ifeq ($(UNAME_S),Linux)
ifeq ($(LIBOMP_LIBDIR),)
ifneq ($(LINUX_GXX),)
LIBOMP_LIBDIR :=
else
AUTO_LIBOMP_LIBDIR := $(shell sh -c 'path=$$(ldconfig -p 2>/dev/null | awk -F"=>" "/libomp\.so/{gsub(/^[ \t]+|[ \t]+$$/,\"\",$$2); print $$2; exit}"); if [ -n "$$path" ]; then dirname $$path; fi')
LIBOMP_LIBDIR := $(AUTO_LIBOMP_LIBDIR)
endif
endif
endif

ifneq ($(LIBOMP_INCDIR),)
OMP_CFLAGS += -I$(LIBOMP_INCDIR)
endif

OMP_LINK_LIB := -lomp
ifneq ($(LINUX_GXX),)
OMP_LINK_LIB :=
endif

OMP_LDFLAGS := $(OMP_LINK_LIB)
ifneq ($(LIBOMP_LIBDIR),)
OMP_LDFLAGS := -L$(LIBOMP_LIBDIR) $(OMP_LINK_LIB)
endif

.PHONY: all clean help

all: $(TARGET)

FWHT_LIBDIR := $(abspath $(ROOT_DIR)/lib)

$(TARGET): ensure_libfwht $(SOURCES) $(FWHT_STATIC)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(OMP_CFLAGS) $(SOURCES) -L$(FWHT_LIBDIR) -lfwht $(CUDA_LDFLAGS) $(OMP_LDFLAGS) -Wl,-rpath,$(FWHT_LIBDIR) -o $@

.PHONY: ensure_libfwht
ensure_libfwht:
	@mkdir -p $(BUILD_DIR)
	@if [ ! -f $(PLATFORM_FILE) ]; then echo $(UNAME_S) > $(PLATFORM_FILE); fi
	@if [ $(PLATFORM_MISMATCH) -eq 1 ] || [ ! -f $(BUILD_DIR)/fwht_core.o ]; then \
		if [ $(PLATFORM_MISMATCH) -eq 1 ]; then \
			$(MAKE) -C $(ROOT_DIR) clean; \
		fi; \
		$(MAKE) -C $(ROOT_DIR); \
		echo $(UNAME_S) > $(PLATFORM_FILE); \
	fi

clean:
	rm -f $(TARGET)

help:
	@echo "Targets:"
	@echo "  all (default)    Build compare_sboxu_fwht"
	@echo "  clean            Remove the comparison binary"
	@echo
	@echo "Variables:"
	@echo "  CXX              Compiler (default: clang++)"
	@echo "  CXXFLAGS         Extra compiler flags (default: -std=c++17 -O3)"
	@echo "  LIBOMP_INCDIR    Override OpenMP include path"
	@echo "  LIBOMP_LIBDIR    Override OpenMP library path"
