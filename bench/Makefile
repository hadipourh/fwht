UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -pthread -I../include
LDFLAGS ?=

ifeq ($(UNAME_S),Darwin)
ifneq ($(NO_OPENMP),1)
CFLAGS += -Xpreprocessor -fopenmp -isystem /opt/homebrew/opt/libomp/include
LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif
else ifeq ($(UNAME_S),Linux)
ifneq ($(NO_OPENMP),1)
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif
endif

FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null)
FFTW_LIBS := $(shell pkg-config --libs fftw3 2>/dev/null)
FFTW_LIBDIR := $(shell pkg-config --variable=libdir fftw3 2>/dev/null)
FFTW_THREAD_LIBS :=
FFTW_THREAD_CFLAGS :=
ifneq ($(FFTW_LIBDIR),)
ifneq ($(wildcard $(FFTW_LIBDIR)/libfftw3_threads.*),)
FFTW_THREAD_LIBS += -L$(FFTW_LIBDIR) -lfftw3_threads
FFTW_THREAD_CFLAGS += -DUSE_FFTW_THREADS
endif
endif
HAVE_FFTW := $(if $(FFTW_LIBS),1,0)

.PHONY: all sboxu ffht fftw clean help

# Default target: do nothing by default to avoid building external benches
all:
	@echo "bench: no default build; use 'make ffht' or 'make sboxu' explicitly."

# Explicit target to build the sboxU comparison benchmark
sboxu:
	$(MAKE) -f Makefile.sboxu

# Build standalone FFHT vs libfwht comparison benchmarks
ffht: compare_ffht_fwht compare_ffht_fwht_fp64

# Build standalone FFTW vs libfwht comparison benchmark
fftw: compare_fftw_fwht

FFHT_SRCS = \
	../otherlibs/FFHT/fht.c \
	../otherlibs/FFHT/fast_copy.c

ifeq ($(HAVE_FFTW),1)
compare_fftw_fwht: compare_fftw_fwht.c
	$(CC) $(CFLAGS) $(FFTW_CFLAGS) $(FFTW_THREAD_CFLAGS) \
		compare_fftw_fwht.c -L../lib -lfwht $(FFTW_LIBS) $(FFTW_THREAD_LIBS) $(LDFLAGS) -lm -o compare_fftw_fwht
else
compare_fftw_fwht:
	@echo "FFTW development files were not found via pkg-config; skipping build."
endif

ifeq ($(UNAME_M),x86_64)
compare_ffht_fwht: compare_ffht_fwht.c $(FFHT_SRCS)
	$(CC) $(CFLAGS) -march=native -I../otherlibs/FFHT \
		compare_ffht_fwht.c $(FFHT_SRCS) -L../lib -lfwht $(LDFLAGS) -lm -o compare_ffht_fwht

compare_ffht_fwht_fp64: compare_ffht_fwht_fp64.c $(FFHT_SRCS)
	$(CC) $(CFLAGS) -march=native -I../otherlibs/FFHT \
		compare_ffht_fwht_fp64.c $(FFHT_SRCS) -L../lib -lfwht $(LDFLAGS) -lm -o compare_ffht_fwht_fp64
else
compare_ffht_fwht:
	@echo "FFHT benchmark is only supported on x86_64; skipping build."

compare_ffht_fwht_fp64:
	@echo "FFHT benchmark is only supported on x86_64; skipping build."
endif

clean:
	$(MAKE) -f Makefile.sboxu clean || true
	rm -f compare_fftw_fwht compare_ffht_fwht compare_ffht_fwht_fp64

help:
	@echo "Bench targets:"
	@echo "  make sboxu    - build compare_sboxu_fwht benchmark"
	@echo "  make fftw     - build FFTW comparison benchmark"
	@echo "  make ffht     - build FFHT comparison benchmarks (float and fp64)"
	@echo "  make clean    - clean benchmark artifacts"
	@echo "  make help     - show this help"
