Latest v0.1.3 INT8 convolution / FP32 LSTM and CPU measurements: [INT8_CPU.md](INT8_CPU.md). The tables below preserve v0.1.2 results.

# BatchSize=4096、长度10240：实测与两种策略

测试日期：2026-10-14。代码版本 0.1.2，模型仍为原始 PNSN v3，FP32 / INT8 权重没有重新训练。

v0.1.2 新增 cuDNN 后端：卷积、转置卷积、池化、两个双向 LSTM 使用 cuDNN，保留独立 B 维。FP32 计算，TF32 关闭。Q8 文件在加载时还原权重，因此是 Q8 存储/FP32 计算。

## 接口策略

| 用途 | 输入 | 接口 | 到时与 NMS |
|---|---|---|---|
| 历史单条三分量 | `[3,N]`，N 任意 | `pick_history` / C `pnsn_pick_history` | 内部滑窗；对整条记录全局 NMS；到时相对历史记录起点 |
| 实时独立批量 | `[B,3,L]`，L 默认10240，可6144等128倍数 | `predict_batch`、`pick_batch` / 对应 C API | 各条分别去均值、归一化、LSTM初始状态和 NMS；到时相对各条起点 |

Python 概率输出 `[B,5,L]`，C 原始概率输出 `[B,L,5]`；震相顺序 Noise/Pg/Sg/Pn/Sn。Python `pick_batch` 返回 B 项列表，各项 `[Ki,3]`；C 提供平坦结果和 `offsets[B+1]`。所有 realtime batch 测量实际传入完整 `[4096,3,10240]`，没有把 B 拼成一条历史序列。

CUDA 采用完整 B 的激活张量、逐记录并行 float64 预处理及 B 个独立 LSTM block；im2col 按层分块限制工作区。MLX 也保留完整 batch 图和 B 个独立 recurrent block，卷积按层分块128条。CPU 接口明确为逐条 SIMD 回退，不称为并行 GPU batch。

## 主机原始输入 → 主机概率输出

输入为 seed=20261014 的4096条独立生成 float32波形，L=10240。包括内部去均值、归一化、输出分配，CUDA 包含 H2D/D2H；不含造数、模型加载、NMS。CUDA/PyTorch 为5次暖态中位数；MLX为3次；CPU受耗时限制为1次暖态，不能据此估计统计分布。

| 主机 / 后端 | 精度 | 秒/整批 | 条/秒 | 备注 |
|---|---|---:|---:|---|
| RTX5090 / PyTorch CUDA（重测） | FP32 | 0.291919 | 14031.3 | TF32关闭，原始未融合模型 |
| RTX5090 / C接口 cuDNN | FP32 | 0.263527 | 15543.0 | 14个BN已融合，float64 GPU预处理 |
| RTX5090 / C接口 cuDNN | Q8存储/FP32计算 | 0.263339 | 15554.1 | 加载时还原权重，不是INT8算术 |
| RTX5090 / 自写CUDA | fp32 | 1.555080 | 2633.9 | 真实 batch，GPU float64预处理 |
| RTX5090 / 自写CUDA | int8 | 1.425481 | 2873.4 | 真实 batch，GPU float64预处理 |
| M4 Max / MLX | fp32 | 13.453426 | 304.5 | 真实 batch图，CPU float64预处理 |
| M4 Max / MLX | int8 | 13.842837 | 295.9 | 真实 batch图，CPU float64预处理 |
| M4 Max / C CPU | fp32 | 28.089196 | 145.8 | 逐条SIMD回退，1线程 |
| M4 Max / C CPU | int8 | 21.206716 | 193.1 | 逐条SIMD回退，1线程 |

同一主机边界下，新增 cuDNN FP32 比自写 CUDA FP32 快 **5.90倍**，耗时比同机重测的原始 PyTorch 模型低 **9.73%**。cuDNN 与 PyTorch 均使用成熟 GPU 算子；新增后端已融合14个BN、复用工作区及自行实现 float64 预处理，这个差异不能全部归因于 C 语言。自写 CUDA 仍作为 W8A8 对照保留。

cuDNN 的 Q8存储与 FP32模型使用相同大小的 GPU 缓冲区，速度基本相同。Q8暖态5次中出现0.382/0.293秒波动，p95为0.364秒，原始时间全部保留。

上一版单条 latency 的优势不能外推到 Batch4096；本机 MLX 的单窗优势同样没有在这次大 batch 中等比例延续。MLX INT8并未更快，且峰值活动内存更大，不能仅凭模型文件缩小推断整个 batch 内存降低。

## GPU 内部阶段（不含主机传输与 NMS）

| 后端 / 精度 | 去均值+归一化（秒） | 网络（秒） | 两阶段（秒） |
|---|---:|---:|---:|
| PyTorch CUDA FP32（重测） | 未单独计时 | 0.133633 | 0.142805 |
| C接口 cuDNN FP32 | 0.002875 | 0.104651 | 0.107526 |
| C接口 cuDNN Q8存储/FP32计算 | 0.002876 | 0.104229 | 0.107105 |
| 自写CUDA fp32 | 0.002911 | 1.393185 | 1.396096 |
| 自写CUDA int8 | 0.002910 | 1.265904 | 1.268815 |

PyTorch 是同步 wall clock 的 GPU驻留输入/输出测量；原生 CUDA/cuDNN 为同一GPU stream的 event计时，不含主机传输。不可把这里的驻留时间与上一张表的主机端到端时间混为一个指标。

## 完整实时拾取：含逐条 NMS

另外实际执行了 `pick_batch`，包括原始主机输入到 B 条独立最终结果。下面各项仅1次测量，合成输入中震相候选很少，不能代表真实事件密集数据的 NMS 压力：

| 后端 | 精度 | 秒/整批 | 条/秒 | 总拾取数 |
|---|---|---:|---:|---:|
| C接口 cuDNN | FP32 | 0.330712 | 12385.4 | 48 |
| C接口 cuDNN | Q8存储/FP32计算 | 0.335597 | 12205.1 | 63 |
| 自写 CUDA | fp32 | 1.639595 | 2498.2 | 48 |
| 自写 CUDA | int8 | 1.500206 | 2730.3 | 68 |
| MLX | fp32 | 13.923660 | 294.2 | 48 |
| MLX | int8 | 16.673782 | 245.7 | 63 |

FP32 cuDNN、自写 CUDA 与 MLX均为48个拾取；cuDNN Q8存储/FP32计算为63个。INT8 CUDA为W8A8、MLX为W8A32，因此预测数与概率存在差异；本表为速度测量，合成数据不用于准确率/F1评估。

## 内存

- PyTorch CUDA 主机端到端阶段：峰值 allocated **7.05 GiB**，峰值 reserved **13.56 GiB**。
- cuDNN FP32 / Q8存储：保留的权重、张量及算法工作区 **6.02 GiB**；不含driver、cuDNN内部handle开销及主机内存，是buffer accounting而非进程峰值显存。
- 自写 CUDA INT8：实际保留的权重/张量工作区 **5.89 GiB**；不含 CUDA driver、主机输入/输出和 CPU NMS。这是 buffer accounting，不是进程峰值显存。
- MLX fp32：峰值 active **20.74 GiB**（统一内存；不含缓存统计）。
- MLX int8：峰值 active **30.57 GiB**（统一内存；不含缓存统计）。

输入本身为480 MiB（4096×3×10240×4），概率输出800 MiB（4096×5×10240×4）。INT8权重文件477012B，并不会消除FP32激活、输出和大 batch 的工作区。不同统计口径的内存数字不可直接当作等价的总进程内存。

## 校验与复现

- `B=1/3/5`、`L=128/6144/10240` 与独立单窗逐条对照。
- `[3,N]` 与兼容的 `[N,3]` 历史接口逐条一致；涵盖0/1/2点及尾窗。
- 批内记录置换、真实波形、恒定记录、float64大DC、int32极值；确认不共享均值、LSTM状态或NMS。
- raw+diff 逐记录 NMS、NaN和错误shape拒绝、容量不足时输出/offsets原子性。
- 完整4096批的每个概率均有限且类别和为1，抽取首/中/末记录与独立单窗对比。
- C共享/静态库、ASan/UBSan单元与批API测试通过；老接口CPU/CUDA数值回归通过。

cuDNN原始记录：[速度](batch4096-native-cudnn.json)、[同机PyTorch重测](batch4096-torch-cudnn-comparison.json)、[数值校验](cudnn-validation.json)、[批策略校验](batch-cudnn-validation.json)。标签准确率单列在 [cuDNN报告](CUDNN.md)。

原v0.1.1记录：[PyTorch](batch4096-torch.json)、[CUDA](batch4096-native-cuda.json)、[MLX](batch4096-mlx.json)、[CPU](batch4096-cpu.json)、[本机批正确性](batch-local-validation.json)、[CUDA批正确性](batch-cuda-validation.json)。

```sh
python native/tools/benchmark_batch_torch.py --batch 4096 --length 10240 --repeats 5 --output native/reports/batch4096-torch.json
python native/tools/benchmark_batch_native.py --backend cudnn --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cudnn.json
python native/tools/benchmark_batch_native.py --backend cuda --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cuda.json
python native/tools/benchmark_batch_native.py --backend mlx --batch 4096 --length 10240 --repeats 3 --post --output native/reports/batch4096-mlx.json
python native/tests/validate_batch.py --library native/build/libpnsn_cuda.so
```

实时示例（先将 `native/python` 加入sys.path）：

```python
from pnsn import Picker
with Picker('native/models/pnsn-fp32.bin', library='native/build/libpnsn_cudnn.so', window=10240) as picker:
    history_picks = picker.pick_history(history_3n)     # [3,N]
    probabilities = picker.predict_batch(realtime_b3l) # [4096,3,10240] -> [4096,5,10240]
    realtime_picks = picker.pick_batch(realtime_b3l)    # B个独立结果
```
