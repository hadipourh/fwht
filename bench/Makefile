UNAME_M := $(shell uname -m)

.PHONY: all sboxu ffht clean help

# Default target: do nothing by default to avoid building external benches
all:
	@echo "bench: no default build; use 'make ffht' or 'make sboxu' explicitly."

# Explicit target to build the sboxU comparison benchmark
sboxu:
	$(MAKE) -f Makefile.sboxu

# Build standalone FFHT vs libfwht comparison benchmarks
ffht: compare_ffht_fwht compare_ffht_fwht_fp64

FFHT_SRCS = \
	../otherlibs/FFHT/fht.c \
	../otherlibs/FFHT/fast_copy.c

ifeq ($(UNAME_M),x86_64)
compare_ffht_fwht: compare_ffht_fwht.c $(FFHT_SRCS)
	$(CC) -O3 -march=native -std=c11 -I../include -I../otherlibs/FFHT \
		compare_ffht_fwht.c $(FFHT_SRCS) -L../lib -lfwht -lm -o compare_ffht_fwht

compare_ffht_fwht_fp64: compare_ffht_fwht_fp64.c $(FFHT_SRCS)
	$(CC) -O3 -march=native -std=c11 -I../include -I../otherlibs/FFHT \
		compare_ffht_fwht_fp64.c $(FFHT_SRCS) -L../lib -lfwht -lm -o compare_ffht_fwht_fp64
else
compare_ffht_fwht:
	@echo "FFHT benchmark is only supported on x86_64; skipping build."

compare_ffht_fwht_fp64:
	@echo "FFHT benchmark is only supported on x86_64; skipping build."
endif

clean:
	$(MAKE) -f Makefile.sboxu clean || true
	rm -f compare_ffht_fwht compare_ffht_fwht_fp64

help:
	@echo "Bench targets:"
	@echo "  make sboxu    - build compare_sboxu_fwht benchmark"
	@echo "  make ffht     - build FFHT comparison benchmarks (float and fp64)"
	@echo "  make clean    - clean benchmark artifacts"
	@echo "  make help     - show this help"
