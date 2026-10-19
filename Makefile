NVCC ?= nvcc
TARGET := gpu_heat_monitor

.PHONY: all clean

all: $(TARGET)

$(TARGET): gpu_heat_monitor.cu
	$(NVCC) -O3 -std=c++17 -Xcompiler=-Wall $< -lnvidia-ml -o $@

clean:
	$(RM) $(TARGET)
