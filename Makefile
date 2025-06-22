show: out.ppm
	eog Outputs/out.ppm

out.ppm: build
	rm -f Outputs/out.ppm
	time ./Builds/rt > Outputs/out.ppm

NVCC=nvcc
CUDAFLAGS= -std=c++17 -arch=sm_89

build: Scripts/main.cu
	$(NVCC) $(CUDAFLAGS) -o Builds/rt Scripts/main.cu

clean:
	rm -rf Builds/rt Outputs/out.ppm