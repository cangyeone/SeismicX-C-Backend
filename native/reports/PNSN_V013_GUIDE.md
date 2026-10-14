# PNSN 原生推理（v0.1.3）

基于上游 [cangyeone/pnsn](https://github.com/cangyeone/pnsn) 的 `BRNNPNSN` v3，源码提交 `1f955f4d8cffbc5c43584f02ef14a0da2b0aeb6b`，默认权重 `ckpt/pnsn.v3.pt`。不重新训练模型。CPU 推理库自己用 C11 实现；CUDA 提供 cuDNN 加速后端及自写内核后端；Apple GPU 使用 MLX 薄 Python 适配层和自写 Metal LSTM 内核。

## 已实现

- 历史策略：`[3,N]` 任意长度三分量输入，`pnsn_pick_history` 内部滑窗、全局 NMS；0/1/2 点也有定义。
- 实时策略：`[B,3,L]` 输入，`pnsn_predict_batch` 真正 CUDA batch，`pnsn_pick_batch` 逐条独立 NMS。Python 概率输出 `[B,5,L]`，C 输出 `[B,L,5]`。L 默认 10240，也支持 6144 等 128 的倍数。
- 实时批内每条记录的去均值、归一化和双向 LSTM 状态相互隔离；不可把 B 条记录拼成历史长序列。两种 CUDA 后端均使用 GPU float64 预处理；cuDNN 使用独立记录的批量双向 LSTM。
- 单条固定长度概率输出：`pnsn_predict`；默认 10240，6144 及其他 128 的整数倍均可，权重无需重新导出。
- 模型调用内部去均值、归一化：float64 移位后 Welford，逐分量样本标准差 `ddof=1`，分母加 `1e-6`；支持原始 float32、float64、int32，拒绝 NaN/Inf。
- 融合所有 14 个 BatchNorm：包括转置卷积的输出通道维度。运行时没有 BN 算子。
- FP32 和 INT8 两种版本；INT8 权重按输出通道对称量化，CPU/自写 CUDA 动态逐输入行激活量化，INT32 累加，LSTM 状态和非线性保持 float32。
- CPU：AArch64 NEON、运行时探测的 DOTPROD；x86 运行时探测 AVX2；其他平台标量回退。转置卷积按奇偶拆分，避免与插入零相乘。可选 OpenMP。
- 后处理：以 `(-confidence, sample, phase)` 为键的真正二叉最小堆、时间坐标索引和 successor/并查集区间删除；完整 confidence 优先贪心 NMS，无 top-K 截断。
- 模型严格校验，工作区复用，候选内存预算，静态库、共享库、纯 C CLI、Python 接口、可复现测试及基准脚本。

## 快速使用

以下命令在仓库根目录运行。

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DPNSN_OPENMP=OFF
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure

# 只有导出和对照测试需要 torch；纯 C 推理不需要 Python/torch/numpy。
python native/tools/export.py --output native/models/pnsn-fp32.bin --precision fp32
python native/tools/export.py --output native/models/pnsn-int8.bin --precision int8
```

现有交付目录和发布包包含已导出的两个 `.bin` 文件。模型文件使用版本化小端格式，附带 checkpoint/model SHA256 清单。C 加载器检查实际层结构、有限值、文件长度。

### Python（CPU）

```python
import sys
import numpy as np
sys.path.insert(0, 'native/python')
from pnsn import Picker

# E/N/Z 顺序；采样率应为 100 Hz；无需外部去均值/归一化。
history = np.load('history_enz.npy')  # [3,N]，推荐 float64 或 int32
realtime = np.load('batch_enz.npy')   # [B,3,10240]
with Picker('native/models/pnsn-int8.bin') as picker:
    arrivals = picker.pick_history(history) # [K,3]: phase/sample/confidence
    prob = picker.predict_batch(realtime)    # [B,5,10240]
    by_record = picker.pick_batch(realtime)  # 长度 B 的列表，各项 [Ki,3]

with Picker('native/models/pnsn-int8.bin', window=6144, overlap=1024) as picker:
    prob6144 = picker.predict_batch(realtime[:, :, :6144])
```

震相编号 `0=Pg, 1=Sg, 2=Pn, 3=Sn`；到时是相对输入起点的整数采样点，秒数为 `sample / sampling_rate`。拾取结果按时间、震相排序。原始 float32 如果已经丢失小信号精度，float64 累加无法恢复已丢失的数据。

### C

头文件 `native/include/pnsn.h` 提供不透明 context 和 C ABI。示例：

```c
#include "pnsn.h"
#include <stdio.h>

/* wave 指向调用方自己的 [3,n] int32 历史数据；output 容量由调用方提供。 */
int pick(const int32_t *wave, size_t n, pnsn_pick *output, size_t capacity) {
    pnsn_options opt = pnsn_default_options();
    opt.window = 10240;
    opt.max_candidates = 32768;
    pnsn_context *ctx = pnsn_open("pnsn-int8.bin", &opt);
    if (!ctx) { fprintf(stderr, "%s\n", pnsn_last_error()); return -1; }
    size_t count = 0;
    int status = pnsn_pick_history(ctx, wave, PNSN_I32, n, output, capacity, &count);
    if (status) fprintf(stderr, "%s\n", pnsn_last_error());
    pnsn_close(ctx);
    return status ? status : (int)count;
}
```

固定概率接口需要输出缓冲区包含 `window*5` 个 float；输入、输出缓冲区应独立。一个 context 复用所有固定窗口工作区；同一个 context 不应被并发调用，多线程使用各自 context。错误文本为线程局部存储。拾取容量不足返回 `-2`，`count` 给出所需数量，输出缓冲区保持原值；候选预算不足返回错误，不丢弃低分候选伪装为成功。

`predict_batch` / `pick_batch` 接受 `[B,3,L]` 的连续 E/N/Z 数组，窗口内到时按各自记录起点计数。C `pick_batch` 返回平坦数组与 `offsets[B+1]`，第 b 条结果位于 `[offsets[b], offsets[b+1])`，不会跨记录互相抑制。`pick_batch` 支持 raw+diff；概率接口只返回 raw。`normalized=True` 仅供已经规范化的 float32 输入做专家/基准测试，常规调用应使用默认内部预处理。

原有 `pick([N,3])` 和 `predict([L,3])` 保留为兼容接口。历史和实时的新接口不依赖自动猜测维度。

### 纯 C CLI

输入是当前主机字节序的原始数值，无文件头。`--history-planar` 对应 `[3,N]`；`--batch B` 对应 `[B,3,L]`；不加上述选项保留原交错 `[N,3]` 兼容模式。SAC/miniSEED 解码与重采样由数据采集端负责；核心库没有引入 ObsPy/miniSEED 库。

```sh
native/build/pnsn_cli --model native/models/pnsn-int8.bin \
  --history-planar --input history-3n-f64.bin --dtype f64 --output picks.csv

native/build/pnsn_cli --model native/models/pnsn-int8.bin \
  --batch 4096 --window 10240 --input batch-b3l-f32.bin --dtype f32 --output batch-picks.csv

native/build/pnsn_cli --model native/models/pnsn-fp32.bin \
  --input window-f64.bin --dtype f64 --window 6144 --overlap 1024 \
  --predict --output probabilities-f32.bin
```

`--predict` 输出 `[window,5]` float32 原始二进制；默认输出震相 CSV。`--difference` 为任意长度拾取增加一阶差分通道，再对两次结果一起 NMS；固定概率接口始终返回 raw 单次网络结果。

## Apple MLX

依赖 `mlx`、`numpy` 和编译好的 C 库（共享预处理和 NMS）；不依赖 torch 推理。Metal 内核直接完成每个方向的 LSTM 序列，避免逐时间步从 Python 提交 GPU 操作。INT8 使用原生容器里的同一份 Q8 权重，经无损打包交给 MLX `quantized_matmul`；激活保持 float32，因此是 **W8A32**，与 C/CUDA **W8A8** 的误差不同。

```sh
python -m pip install mlx numpy
```

```python
from pnsn_mlx import MLXPicker  # 先将 native/python 加入 sys.path
picker = MLXPicker('native/models/pnsn-fp32.bin', window=10240)
arrivals = picker.pick_history(history)
prob = picker.predict_batch(realtime)
```

首次调用包含 MLX 图和 Metal 内核编译，冷启动时间另见基准 JSON。实际高幅值波形会产生很大的 LSTM gate，Metal 的 tanh 输入先限幅至 ±10，避免内部指数实现的非有限结果；这个区域内 float32 的 tanh 已饱和至 ±1。已用重复真实波形调用回归验证。

## CUDA / cuDNN（大 batch 使用此后端）

`libpnsn_cudnn` 将卷积、转置卷积、池化和两个双向 LSTM 交给 NVIDIA cuDNN。保留公共 C ABI、内部 float64 预处理、14个 BN 权重融合与 C 最小堆后处理；推理无需 PyTorch/Python。内部特征为 `[B,C,T]`，LSTM 转换为时间优先布局；每次调用初始隐状态为零，批内记录独立。卷积按 cuDNN 启发式选算法并缓存 batch 形状，卷积工作区上限512 MiB；RNN 的必要工作区另计，所有算子复用最大工作区。

实测环境 CUDA12.8 / cuDNN9.10.2 / RTX5090。FP32禁用TF32。原生 `.bin` 格式沿用；Q8文件只在加载时还原权重，然后用FP32计算，属于 **Q8存储/FP32计算**，不等同于CPU/自写CUDA的W8A8。Q8文件小但GPU权重与激活并不因此变成INT8。该路径的标签结果单列在 [cuDNN报告](CUDNN.md)。

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DPNSN_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDNN_ROOT=/path/to/cudnn
cmake --build native/build --parallel
```

`CUDNN_ROOT` 下需有 `include/cudnn.h` 与 `lib`/`lib64` 中的 cuDNN库，也可显式设置 `CUDNN_INCLUDE_DIR`、`CUDNN_LIBRARY`。服务器已安装的前缀为 `/path/to/cuda-python/lib/python3.13/site-packages/nvidia/cudnn`；这里仅复用其头文件和动态库，推理不导入torch。运行时应能找到匹配版本的cuDNN及其CUDA依赖；安装后可用系统动态库路径或 `LD_LIBRARY_PATH`。cuDNN采用动态链接，发布包不附带NVIDIA库。

```python
with Picker('native/models/pnsn-fp32.bin',
            library='native/build/libpnsn_cudnn.so', window=10240) as picker:
    history_picks = picker.pick_history(history)      # [3,N]
    prob = picker.predict_batch(realtime)             # [B,3,10240]
    by_record = picker.pick_batch(realtime)
```

C程序链接 `pnsn_cudnn`，CLI为 `pnsn_cudnn_cli`，参数与CPU CLI一致。`PNSN_CUDNN` 与 `PNSN_CUDA` 可同时打开，生成两个独立库；一份程序选择其中一个库，不能混用各自context。服务器实测4096批FP32概率端到端0.2635秒，网络0.1047秒，较自写CUDA FP32端到端快5.90倍，完整拾取0.3307秒（单次）。详见 [批量报告](BATCH4096.md)。

参考 [NVIDIA cuDNN卷积接口](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/api/cudnn-cnn-library.html) 与 [RNN接口](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.0.0/api/cudnn-adv-library.html)。当前实现使用cuDNN传统C卷积API和v8 RNN API；传统卷积API已被标记deprecated，已在9.10.2实测，未来版本应运行回归后再升级。

## cuDNN INT8 算术后端

v0.1.3新增独立库 `libpnsn_cudnn_int8` 与 `pnsn_cudnn_int8_cli`，打开 `PNSN_CUDNN=ON` 时一起编译。加载 `pnsn-int8.bin` 时，所有17层卷积使用 cuDNN **INT8输入和权重、INT32累加**，输出还原为FP32并加融合后的bias。首层3通道补零到4通道，特征使用NHWC布局；激活按每条记录、每层的最大绝对值动态对称量化，权重沿用逐输出通道scale。每个转置卷积拆为两个奇偶相位的INT8普通卷积，额外尾部零与原output_padding对应，避免对整段插零后运算。

cuDNN9.10.2的RNN接口不支持INT8，所以双向LSTM使用FP32计算（从Q8文件还原其权重）；池化、非线性、层间输出、预处理和最小堆也保留原精度。这是 **卷积W8A8 / LSTM FP32** 混合后端；和CPU/自写CUDA中LSTM矩阵乘法也量化的路径不同。找不到可用INT8卷积算法时返回明确错误，Q8模型不会悄悄切换为FP32卷积。加载FP32模型则明确执行原FP32路径。

```python
with Picker('native/models/pnsn-int8.bin',
            library='native/build/libpnsn_cudnn_int8.so', window=10240) as picker:
    history_picks = picker.pick_history(history)  # [3,N]
    prob = picker.predict_batch(realtime)         # [B,3,10240]
    by_record = picker.pick_batch(realtime)
```

```sh
python native/tests/validate_cudnn.py --int8 --output native/reports/cudnn-int8-validation.json
python native/tests/validate_batch.py --library native/build/libpnsn_cudnn_int8.so
python native/tools/benchmark_batch_native.py --backend cudnn_int8 --precision int8 --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cudnn-int8.json
python native/tools/benchmark_cpu.py --output native/reports/cpu-single.json
python native/tools/benchmark_batch_native.py --backend cpu --precision int8 --batch 4096 --length 10240 --repeats 3 --output native/reports/batch4096-cpu-int8.json
```

整数算子校验（CMake的 `pnsn_cudnn_int8_ops` 测试）对17层、3条独立记录，使用原始完整转置卷积公式的CPU整数参考；整数累加误差为0。独立浮点模拟量化图与整网输出有差异，真实波形最大约0.012，因此准确率另用5730条既有标签评估：Pg/Sg F1为85.9518%，较原模型86.2243%下降0.2725个百分点。CPU的逐patch激活量化不同，不能套用这项GPU混合后端F1。最新速度及完整量化范围见 [INT8与CPU报告](INT8_CPU.md)。

## 自写 CUDA（可选 W8A8 路径）

只依赖 CUDA Runtime；没有 cuBLAS/cuDNN/TensorRT/ONNX Runtime。大 batch 中按层分块 im2col，保留完整 B 个神经网络状态；LSTM 为 B 个独立 block 并行，不是 B 次单窗循环。权重常驻设备，卷积、池化、双向 LSTM、解码器、softmax 在 GPU 上执行；历史路径的 float64 预处理和堆 NMS 在 CPU；实时 batch 的去均值和归一化在 CUDA GPU 用 float64 完成，NMS 仍在 CPU、逐记录独立执行。INT8 内积使用 `__dp4a`（支持时）。

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DPNSN_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build native/build --parallel
```

上面 `120` 对应实测 RTX 5090/CUDA 12.8；其他显卡填自己的计算能力。编译产生独立的 `libpnsn_cuda` 和 `pnsn_cuda_cli`，CPU 库也保留。CUDA Runtime 静态链接，无需在运行时寻找 `libcudart.so`。

```python
picker = Picker('native/models/pnsn-int8.bin',
                library='native/build/libpnsn_cuda.so')
arrivals = picker.pick_history(history)
prob = picker.predict_batch(realtime)
```

```sh
CUDA_VISIBLE_DEVICES=0 native/build/pnsn_cuda_cli \
  --model native/models/pnsn-int8.bin --input raw-f64.bin --output picks.csv
```

本次远程工作目录：remote CUDA server 上的 `/path/to/SeismicX-C-Backend`。使用现有 `/path/to/cuda-12.8/bin/nvcc`、`/path/to/cmake-env/bin/cmake` 和 `/path/to/cuda-python/bin/python`。工具链和 Python 环境没有改动。默认 anaconda3 的旧 torch 不支持该服务器 SM120，验证使用现有 cu128 环境。

batch 4096 测量见 [reports/BATCH4096.md](BATCH4096.md)。大批吞吐不能从单窗速度推算；cuDNN 后端现已实测优于本次 PyTorch 端到端参考，自写 CUDA 仍用于独立 W8A8 对照。

## 边缘设备

```sh
cmake -S native -B native/build-edge -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DPNSN_OPENMP=OFF -DPNSN_CUDA=OFF
cmake --build native/build-edge --parallel
cmake --install native/build-edge --prefix ./pnsn-install
```

- 静态 CPU 核心只需要 C 标准库和 libm，无第三方推理库。CMake 可传交叉编译 toolchain；ARM32/缺少 SIMD 时使用标量回退。
- `pnsn_memory_bytes(ctx)` 返回 CPU 模型和工作区字节数，不包括调用方输入/输出、NMS 临时内存或 CUDA 设备内存。10240 点 INT8 的实测计算值见 `reports/RESULTS.md`。
- 固定窗口调用不再分配工作区；历史任意长度逐窗计算，不保存整段概率张量。实时 batch 分配 B 条激活与概率张量，内存需求单独测量，不适用单窗 1.76 MiB 的数字。CPU batch 当前是逐条 SIMD 回退；CUDA/MLX 具备真正 batch 图。
- 全局精确 NMS 仍需保留候选：候选相关峰值内存上界约 `88*max_candidates` 字节（本次 64-bit ABI，另有分配器开销）。边缘端建议显式设置预算，例如 32768 个候选约 2.75 MiB，并按数据时长和阈值调整。
- 调用方持有整段输入，本接口不是自动管理 ring buffer 的实时流接口。逐块调用适合在线窗口处理，但跨块全局 NMS 应由上层合并后调用 `pnsn_nms`；不能声称简单块内 NMS 与全局 NMS 等价。
- 6144 更省内存，但比训练时 10240 的上下文更短；当前人工标签 F1 验证针对 10240。
- macOS ARM64、Linux x86_64 和 RTX 5090 已实测；Windows 构建矩阵已写入 CI，但此会话未执行 Windows CI，也未实测实体 ARM 边缘板。

## 兼容性与数值定义

任意长度允许 0 个采样点（返回空），短波形用最后真实采样点重复填充。超过一窗的输入步长为 `window-overlap`，覆盖真实输入后停止。剔除填充区的候选，避免产生原始时长外的到时。阈值为严格 `score > threshold`，同类 `|sample difference| <= min_gap` 被抑制，阈值默认 0.1、gap 默认 300。

上游 TorchScript 直接将索引限制在 `N-2`、预处理用 float32、取 `ceil(N/stride)` 窗口；本版本修复这些边界行为，**长波形边缘结果不保证与旧 JIT 逐点相同**。浮点后端的微小舍入差异也可能在平坦峰顶选择不同采样点。INT8 有真实量化误差，不应宣传完全无损。

## 测试

```sh
python native/tools/prepare_example.py  # 仅示例格式转换需要 obspy
python native/tests/validate.py
python native/tests/validate_batch.py --mlx
python native/tests/validate.py --mlx --output native/reports/local-mlx-validation.json
python native/tests/validate.py --library native/build/libpnsn_cuda.so
python native/tests/validate_cudnn.py --output native/reports/cudnn-validation.json
python native/tests/validate_batch.py --library native/build/libpnsn_cudnn.so
python native/tools/benchmark.py --mlx --repeats 50 \
  --input native/reports/real-waveform.npy --output native/reports/local-benchmark.json
python native/tools/benchmark.py --cuda --repeats 50 \
  --input native/reports/real-waveform.npy --output native/reports/remote-benchmark.json

cmake -S native -B native/build-asan -DPNSN_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build native/build-asan --parallel
ctest --test-dir native/build-asan --output-on-failure
```

基准包含 float64 预处理、GPU 输入/输出同步；固定概率计时不包含后处理，完整拾取计时另见 `benchmark_trace.py`。模型载入不计入暖态计时，首次推理单独报告。真实数据测试需要先用已有上游 SAC 文件生成 `reports/real-waveform.npy`。

人工标签验证使用已有 31 省测试缓存，只读，依据 `test.json` 的有效行数排除 `.npy` 尾部填充；2020–2022 测试，Pg/Sg 人工标签，四类预测同时与 FP32 对比。CUDA FP32 参考禁用 TF32。

```sh
python native/tools/evaluate_labeled.py --cache /path/to/existing/cache \
  --output native/reports/labeled-evaluation.json
```

结果、速度限制和准确率范围见 [reports/RESULTS.md](RESULTS.md)。参考接口文档：[MLX quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)、[NVIDIA 整数内积](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)。

## 批量性能复现

```sh
python native/tools/benchmark_batch_torch.py --batch 4096 --length 10240 --repeats 5 --output native/reports/batch4096-torch.json
python native/tools/benchmark_batch_native.py --backend cudnn --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cudnn.json
python native/tools/benchmark_batch_native.py --backend cuda --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cuda.json
python native/tools/benchmark_batch_native.py --backend mlx --batch 4096 --length 10240 --repeats 3 --post --output native/reports/batch4096-mlx.json
```

输入为 4096 条独立生成的 float32 波形；本测量不含造数/加载模型。`--post` 另外测一次从主机原始数据到逐记录最终拾取的完整链路。概率阶段不含 NMS。CUDA event 给出单独 GPU 预处理与网络时间，端到端 wall time 包含主机传输与输出分配。CPU 回退结果标明实际逐条处理，避免与 GPU 真正 batch 混淆。
