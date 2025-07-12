# Compiler and flags
NVCC = nvcc
CUDA_ARCH = -arch=sm_80
CUDAFLAGS = -std=c++17 $(CUDA_ARCH) -rdc=true


# Targets
TARGET = Builds/rt
OUTPUT_PPM = Outputs/out.ppm

# Default target
all: $(OUTPUT_PPM)

# Rule to show the output image
show: $(OUTPUT_PPM)
	eog $(OUTPUT_PPM)

# Rule to generate the output PPM file
$(OUTPUT_PPM): build
	rm -f $(OUTPUT_PPM)
	time ./$(TARGET) > $(OUTPUT_PPM)

# Build target
build: Scripts/main.cu
	$(NVCC) $(CUDAFLAGS) -o $(TARGET) Scripts/main.cu $(LDFLAGS)

# Clean target
clean:
	rm -rf $(TARGET) $(OUTPUT_PPM)
