.DEFAULT_GOAL := all

CONFIG_MK ?= config.mk

# Cleaning and host-only tests do not require a CUDA configuration.
ifeq ($(filter-out clean test,$(MAKECMDGOALS)),)
ifeq ($(strip $(MAKECMDGOALS)),)
include $(CONFIG_MK)
endif
else
include $(CONFIG_MK)
endif

NVCCFLAGS ?= -O3
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

TARGET := build/cuda_state_vector
OBJECT := build/main.o

.PHONY: all clean test test-cuda
all: $(TARGET)

$(TARGET): $(OBJECT)
	$(NVCC) $(GENCODE_FLAGS) $(LDFLAGS) $^ $(LDFLAGS_VENDOR) $(LDLIBS) -o $@

$(OBJECT): src/main.cu $(CONFIG_MK) Makefile | build
	$(NVCC) $(CPPFLAGS) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

test: | build
	$(CXX) -O2 -Wall -Wextra -Werror -std=c++17 tests/indexing.cpp -o build/test_indexing
	./build/test_indexing
	$(CXX) -O2 -Wall -Wextra -Werror -std=c++17 tests/qubit_order.cpp -o build/test_qubit_order
	./build/test_qubit_order

test-cuda: | build
	$(NVCC) $(CPPFLAGS) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(NVCCFLAGS) -std=c++17 tests/kernels.cu $(LDFLAGS) $(LDFLAGS_VENDOR) $(LDLIBS) -o build/test_kernels
	./build/test_kernels

build:
	mkdir -p $@

clean:
	rm -rf build

-include $(OBJECT:.o=.d)
