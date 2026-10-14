# seismicx-picker-c-backend

Licensed under **GPLv3 only (GPL-3.0-only)**. See [LICENSE](LICENSE).

Native C inference for seismic phase picking, with FP32 and 8-bit weights, internal float64 preprocessing, exact min-heap suppression, and CMake builds.

| Model | Reference checkpoint | Probabilities | Fixed window |
|---|---|---|---|
| PNSN v3 | `ckpt/pnsn.v3.pt` | Noise/Pg/Sg/Pn/Sn | multiples of 128, default 10240 |
| China PhaseNet-style UNet | `ckpt/china.unet.pt` | Noise/P/S | multiples of 256, default 10240 |

The China model is the nearest-neighbor upsampling UNet from [cangyeone/pnsn](https://github.com/cangyeone/pnsn), as requested. It is a PhaseNet-style variant, with different layers from the original AI4EPS or SeisBench PhaseNet. Both checkpoints are exported without retraining. All BatchNorm layers are fused: 14 in PNSN and 10 in China UNet.

Two explicit input strategies are supported: history `[3,N]` with arbitrary N and global sliding-window picking, and independent realtime batches `[B,3,L]`. Each realtime record has its own preprocessing, model state and min-heap NMS. Python probabilities are `[B,C,L]`; C probabilities are `[B,L,C]`, with C read from the loaded model.

The CPU backend is C11 and needs the C standard library and libm, with NEON/DOTPROD or AVX2 acceleration when available. cuDNN is optional; MLX/Metal is an optional Python adapter. Pure C applications and CLI inference need no Python, PyTorch, ONNX Runtime or TensorRT.

## Build and run

Pre-exported FP32/INT8 `.bin` models and reference checkpoints are included.

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DPNSN_OPENMP=OFF
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure

native/build/seismicx_cli --model native/models/china-unet-int8.bin \
  --history-planar --input history-3n-f64.bin --dtype f64 --output picks.csv
```

The CLI input is raw native-endian E/N/Z numerical data. SAC/miniSEED decoding and sampling-rate conversion are handled by the caller. Reference checkpoints use 100 Hz input in the provided evaluations.

```python
import sys
sys.path.insert(0, 'native/python')
from seismicx import Picker

with Picker('native/models/china-unet-int8.bin', threads=1) as picker:
    history_picks = picker.pick_history(history_3n)      # [3,N] -> [K,3]
    probabilities = picker.predict_batch(realtime_b3l) # [B,3,10240] -> [B,3,10240]
    independent_picks = picker.pick_batch(realtime_b3l)
    print(picker.model_name, picker.classes)
```

C callers include [seismicx.h](native/include/seismicx.h), link `pnsn` / CMake `SeismicX::CPU`, and use `sx_open`, `sx_classes`, `sx_predict_batch`, `sx_pick_history`, and `sx_pick_batch`. The original `pnsn_*` ABI remains available. One context owns reusable workspaces and is used by one thread at a time.

## CUDA and actual INT8 arithmetic

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DPNSN_CUDNN=ON -DCUDNN_ROOT=/path/to/cudnn \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build native/build --parallel
```

`120` is the tested RTX5090/CUDA12.8 architecture; choose the architecture of your GPU. cuDNN9.10.2 was tested. cuDNN and its CUDA dependencies must be installed; NVIDIA libraries are not redistributed here.

- `pnsn_cudnn` / `SeismicX::CuDNN`: FP32 arithmetic, including when Q8 weights are dequantized at loading.
- `pnsn_cudnn_int8` / `SeismicX::CuDNN_INT8`: true INT8×INT8 convolution arithmetic with INT32 accumulation and float activations. China UNet uses per-patch activation quantization and tiled cuDNN 1×1 convolutions, matching the CPU quantization granularity. PNSN uses per-record activation quantization and polyphase INT8 transposed convolutions; its two bidirectional LSTMs remain FP32 because the cuDNN RNN interface does not support INT8.
- `pnsn_cuda`: optional custom PNSN CUDA/DP4A reference backend; China UNet is supported through cuDNN.
- CPU INT8 uses W8A8 matrix arithmetic, including PNSN LSTM matrix products; recurrent states and nonlinearities remain float.
- MLX Q8 uses W8A32: quantized weights with floating activations. This is a distinct precision mode and is measured separately.

```python
with Picker('native/models/china-unet-int8.bin',
            library='native/build/libpnsn_cudnn_int8.so') as picker:
    probabilities = picker.predict_batch(realtime_b3l)
```

## Measurements and validation

[Device matrix](native/reports/DEVICE_MATRIX.md) compares every model on the available M4 Max CPU, M4 Max MLX GPU, Threadripper 3955WX CPU and RTX5090 cuDNN GPU. Tests include actual `4096×3×10240` batches, 6144/10240 single-window latency, and complete picks. Timings, precision and thread counts are recorded rather than inferred from language or model file size.

[China UNet results](native/reports/PHASENET.md) and [PNSN INT8/CPU results](native/reports/INT8_CPU.md) report numerical drift and held-out label comparisons. Original FP32 predictions are preserved within floating point tolerance; quantization changes some predictions. Accuracy is evaluated on the stated test cache and thresholds, with its phase-label coverage documented.

CPU portability CI covers Linux/macOS/Windows and shared/static builds. CUDA and MLX were tested on the hardware listed above; other GPUs and physical ARM edge boards have not been benchmarked. For edge builds use `-DBUILD_SHARED_LIBS=OFF -DPNSN_OPENMP=OFF`, a smaller supported window such as 6144, and an explicit candidate budget.

See [Chinese usage](native/README.zh.md), [model provenance](NOTICE.md), [C API](native/include/seismicx.h) and [GPLv3 license](LICENSE).
