################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
	XCODE_GE_5 = $(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5)
endif

# Take command line flags that override any of these settings
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE    = 32
	OS_ARCH    = armv7l
	ARCH_FLAGS = -target-cpu-arch ARM
endif
ifeq ($(aarch64),1)
	OS_SIZE    = 64
	OS_ARCH    = aarch64
	ARCH_FLAGS = -target-cpu-arch ARM
endif

# Common binaries
ifneq ($(DARWIN),)
ifeq ($(XCODE_GE_5),1)
  CC ?= clang
  GCC ?= clang
else
  CC ?= gcc
  GCC ?= g++
endif
else
ifeq ($(ARMv7),1)
  GCC ?= arm-linux-gnueabihf-g++
else
  GCC ?= g++
endif
endif
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS} -arch=sm_35 -rdc=true -lineinfo
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),)
  LDFLAGS += -rpath $(CUDA_PATH)/lib -framework Accelerate
  # CCFLAGS += -arch $(OS_ARCH)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),androideabi)
      NVCCFLAGS += -target-os-variant Android
    else
      ifeq ($(abi),gnueabi)
        CCFLAGS += -mfloat-abi=softfp
      else
        # default to gnueabihf
        override abi := gnueabihf
        LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
        CCFLAGS += -mfloat-abi=hard
      endif
    endif
  endif
endif

ifeq ($(ARMv7),1)
ifneq ($(TARGET_FS),)
GCCVERSIONLTEQ46 := $(shell expr `$(GCC) -dumpversion` \<= 4.6)
ifeq ($(GCCVERSIONLTEQ46),1)
CCFLAGS += --sysroot=$(TARGET_FS)
endif
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
	CCFLAGS += -g
	NVCCFLAGS += -G
	TARGET := debug
else
	CCFLAGS += -O3
	TARGET := release
endif

# Logging build flags
ifeq ($(log),1)
	CCFLAGS += -DDETAILED_LOGGING
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  :=
LIBRARIES := -lcublas -lcublas_device -lcudadevrt -llapack -lblas -lgomp

ifneq ($(DARWIN),)
  INCLUDES +=  -I/usr/local/opt/openblas/include -I/usr/local/opt/lapack/include
  LIBRARIES += -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -L/usr/local/opt/gcc/lib/gcc/4.9
else
  LIBRARIES += -L/usr/lib64/atlas
endif

################################################################################

# Target rules
all: build

run: build
	$(EXEC) ./gaussian

clean:
	rm -f *.o
	rm -f bench gaussian inverse inverse_bench cholesky_gpu
	rm -rf inverse.dSYM
	rm -rf bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/gaussian

c-test: src/inverse.c
	clang -g -o inverse $+
	echo "\
	11  8   12  13  9\n\
	4   5   6   7   14\n\
	15  10  2   16  17\n\
	18  19  20  21  3\n\
	22  23  24  25  26\n" | ./inverse

build: gaussian

bench.o: src/bench.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench: bench.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

helper.o: src/helper.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

inverse_gauss.o: src/gauss/inverse_gpu.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

inverse_gauss_batched.o: src/gauss/batched_invert.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

inverse_cpu.o: src/inverse.c
	$(EXEC) $(CC) $(INCLUDES) $(CCFLAGS) -fopenmp -o $@ -c $<

inverse_bench.o: src/inverse_bench.c
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

inverse_bench: inverse_bench.o inverse_cpu.o cholesky_gpu.o inverse_gauss.o inverse_gauss_batched.o helper.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

gauss_bench.o: src/gauss_bench.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gauss_bench: gauss_bench.o inverse_cpu.o cholesky_gpu.o inverse_gauss.o inverse_gauss_batched.o helper.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench-all: bench
	./bench 10 134217728 10

cholesky_gpu.o: src/inverse_cholesky_gpu.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cholesky_gpu: cholesky_gpu.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	./cholesky_gpu

cholesky_cpu: src/inverse_cholesky_cpu.c
	clang -g -o cholesky_cpu $+
	echo "\
	18 22 54 42\n\
	22 70 86 62\n\
	54 86 174 134\n\
	42 62 134 106\n" | ./cholesky_cpu

clobber: clean
