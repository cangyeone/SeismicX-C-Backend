# SeismicX-C-Backend native inference

C11 CPU inference for PNSN v3 and the selected China PhaseNet-style UNet, with fused BatchNorm, internal float64 preprocessing, FP32/INT8 models, exact min-heap NMS and CMake shared/static builds. Optional cuDNN FP32/W8A8 and MLX/Metal W8A32 backends support the same two strategies: arbitrary history `[3,N]` and independent realtime batches `[B,3,L]`.

See the [repository overview](../README.md), [Chinese usage](README.zh.md), [complete device matrix](reports/DEVICE_MATRIX.md), [China UNet validation](reports/PHASENET.md), [PNSN INT8 validation](reports/INT8_CPU.md) and [C API](include/seismicx.h).

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DPNSN_OPENMP=OFF
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure
```

Pre-exported models are included in the release repository. CPU inference needs no Python or third-party inference library. Export/reference tests require Python/PyTorch; the MLX adapter is optional. INT8 accuracy and speed are measured separately from FP32; quantization changes some predictions and does not guarantee GPU acceleration.
