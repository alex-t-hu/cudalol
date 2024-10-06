# Compilers and flags
NVCC := /usr/local/cuda/bin/nvcc
CXX := g++
CXXFLAGS := -O3 -std=c++20
NVCCFLAGS := -O3 -arch=sm_89

ifeq ($(DEBUG), 1)
	CXXFLAGS := -g -O0 -std=c++20
	NVCCFLAGS := -g -O0 -arch=sm_89
endif

# Libraries and paths
LIBS := -lcublas -lcudart
CUDA_PATH := /usr/local/cuda
INCLUDE := -I$(CUDA_PATH)/include
LIBRARY_PATH := -L$(CUDA_PATH)/lib64

# Target executable
TARGET := main

# Source files
CU_SRC := $(wildcard *.cu)
CPP_SRC := $(wildcard *.cpp)
CU_OBJ := $(CU_SRC:.cu=.o)
CPP_OBJ := $(CPP_SRC:.cpp=.o)

PTX := $(CU_SRC:.cu=.ptx)

# Combine object files
OBJ := $(CU_OBJ) $(CPP_OBJ)

# Rules
all: $(TARGET) $(PTX)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS) $(LIBRARY_PATH)

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) $(PTX)

.PHONY: clean
