NVCC ?= nvcc
TARGET := thermal_config_experiment

.PHONY: all clean

all: $(TARGET)

$(TARGET): thermal_config_experiment.cu
	$(NVCC) -O3 -std=c++17 -Xcompiler=-Wall $< -lnvidia-ml -o $@

clean:
	$(RM) $(TARGET)
