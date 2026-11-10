.DEFAULT_GOAL := all

CONFIG_MK ?= config.mk

# Cleaning does not require a compiler configuration.
ifneq ($(strip $(MAKECMDGOALS)),clean)
include $(CONFIG_MK)
endif

NVCCFLAGS ?= -O3
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

TARGET := build/cuda_state_vector
OBJECT := build/main.o

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJECT)
	$(NVCC) $(GENCODE_FLAGS) $(LDFLAGS) $^ $(LDFLAGS_VENDOR) $(LDLIBS) -o $@

$(OBJECT): src/main.cu $(CONFIG_MK) Makefile | build
	$(NVCC) $(CPPFLAGS) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build

-include $(OBJECT:.o=.d)
