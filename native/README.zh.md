# seismicx-picker-c-backend（v0.2.1）

SeismicX 的 C11 震相拾取后端，支持 PNSN v3 和现有 `china.unet.pt` 中国数据 PhaseNet 风格 UNet。两套原始权重来自 [cangyeone/pnsn](https://github.com/cangyeone/pnsn)，不重新训练。CPU 推理自己实现，只依赖 C 标准库/libm；NVIDIA 使用可选 cuDNN；Apple GPU 使用可选 MLX/Metal Python 适配层。纯 C 应用推理不需要 Python、PyTorch 或其他推理框架。

## 两种明确的处理策略

| 策略 | 输入 | 行为 |
|---|---|---|
| 历史数据 | `[3,N]`，N 任意，包括0/1/2 | 内部滑窗、补尾窗、跨窗全局最小堆 NMS；返回原始长度内的拾取 |
| 实时批处理 | `[B,3,L]` | 每条独立预处理、状态和 NMS；GPU 真正按层处理整个 batch |

E/N/Z 顺序，输入可为 float32/float64/int32。内部逐分量使用 float64 移位 Welford 去均值，样本标准差（ddof=1）加1e-6归一化；零方差输出零，拒绝 NaN/Inf。固定窗口：PNSN 为128的倍数，China UNet 为256的倍数，默认10240，6144两者均支持。历史 N 无需为该倍数。

| 模型 | 概率类数 C | 拾取 phase 编号 | 融合 BN |
|---|---:|---|---:|
| PNSN v3 | 5：Noise/Pg/Sg/Pn/Sn | 0=Pg、1=Sg、2=Pn、3=Sn | 14 |
| China UNet | 3：Noise/P/S | 0=P、1=S | 10 |

Python 概率为 `[B,C,L]`，C 概率为 `[B,L,C]`。每个 context 持有可复用工作区，同一 context 一次由一个线程调用。实时记录不能拼成历史长序列。默认 threshold=0.1、同相 min_gap=300 个采样点、overlap=1024、threads=1、difference=0。示例/评估使用100Hz；文件解码、通道对齐和重采样由调用方负责。

## 构建与纯 C 使用

命令在仓库根目录运行；发布仓库包含4个已导出 `.bin`，首次运行不需要导出工具。

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DPNSN_OPENMP=OFF
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure

# history-3n-f64.bin 是按 E整条/N整条/Z整条存放的原始float64字节。
native/build/seismicx_cli --model native/models/china-unet-int8.bin \
  --history-planar --input history-3n-f64.bin --dtype f64 --output picks.csv

# 实时原始字节按 [B,3,L] 存放；逐条输出带record编号的拾取。
native/build/seismicx_cli --model native/models/china-unet-int8.bin \
  --batch 4096 --window 10240 --input realtime-b3l-f32.bin --dtype f32 --output batch-picks.csv
```

加 `--predict` 输出原始float32概率字节；`--batch` 时布局 `[B,L,C]`，单条固定输入时布局 `[L,C]`。不加 `--history-planar` 的历史兼容接口接受交错 `[N,3]`。`--difference` 会合并原始与一阶差分两次推理的候选，再做 NMS。

C 程序包含 [seismicx.h](include/seismicx.h)，链接 `pnsn` / `SeismicX::CPU`。`sx_*` 是兼容 `pnsn_*` ABI 的品牌化 C 门面：

```c
#include <seismicx.h>
/* 调用方已准备float64历史输入 history[3*N] 和输出 picks[capacity]。 */
sx_options options = sx_default_options();
options.window = 6144;
options.overlap = 1024;
sx_context *ctx = sx_open("native/models/china-unet-int8.bin", &options);
if (!ctx) { /* sx_last_error() */ }
else {
    size_t count = 0;
    int status = sx_pick_history(ctx, history, SX_F64, N, picks, capacity, &count);
    /* 返回-2表示输出不足，count给出所需大小，输出保持原样。 */
    /* 概率分配应使用 sx_classes(ctx)，不能硬编码为5。 */
    (void)status;
    sx_close(ctx);
}
```

最小堆以 `(-confidence,sample,phase)` 为键，配合时间索引/区间删除完成精确置信度优先贪心 NMS。候选预算超限明确报错，无静默top-K截断；输出容量不足时输出和batch offsets保持原样。历史推理特征工作区只与窗口有关，候选存储仍随候选数量增长。

## Python 与 MLX

```python
import sys
sys.path.insert(0, 'native/python')
from seismicx import Picker

with Picker('native/models/china-unet-int8.bin', threads=1) as picker:
    history_picks = picker.pick_history(history_3n) # [K,3]: phase/sample/confidence
    probabilities = picker.predict_batch(batch_b3l) # [B,3,L]
    by_record = picker.pick_batch(batch_b3l)          # B个独立拾取数组
    print(picker.model_name, picker.classes)

# 支持任意B，以及可配置的其他固定长度；不是将batch拼成一个波形。
with Picker('native/models/pnsn-int8.bin', window=6144) as picker:
    probabilities = picker.predict_batch(batch_b3l[..., :6144])

# pip install mlx numpy；MLX适配层还使用上述C库的预处理/最小堆。
from pnsn_mlx import MLXPicker
apple = MLXPicker('native/models/china-unet-fp32.bin')
probabilities = apple.predict_batch(batch_b3l)
history_picks = apple.pick_history(history_3n)
```

到时单位为相对记录起点的整数采样点，秒数为 sample/sampling_rate；输出按时间、相位排序。int32输入在去均值前转换为float64。已经存入float32而丢失的小信号精度无法靠后续float64累加恢复。

## cuDNN FP32 与真实 INT8 算术

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DPNSN_CUDNN=ON -DCUDNN_ROOT=/path/to/cudnn -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure
```

120是本次 RTX5090/CUDA12.8 的架构配置；其他GPU需使用自己的架构。需要安装CUDA/cuDNN，本仓库不分发其二进制。

| 库 | 量化文件的实际算术 |
|---|---|
| `pnsn` CPU | 卷积/LSTM矩阵 W8A8，INT32累加；状态/非线性为float |
| `pnsn_cudnn` | Q8加载时还原为float权重，FP32计算 |
| `pnsn_cudnn_int8` China UNet | 全11层卷积W8A8，逐片段激活量化、128条im2col分块、cuDNN1×1卷积；float激活 |
| `pnsn_cudnn_int8` PNSN | 全17层卷积W8A8，转置卷积拆奇偶相位；cuDNN双向LSTM保持FP32 |
| MLX Q8 | W8A32，float激活；不是全图INT8 |

GPU主要算子由cuDNN提供，预处理、布局转换和量化由薄CUDA内核完成。cuDNN RNN接口不支持本模型的INT8 LSTM。旧 `PNSN_CUDA` 自写 PNSN 后端保留为可选对照，China UNet使用cuDNN。

```python
with Picker('native/models/china-unet-int8.bin',
            library='native/build/libpnsn_cudnn_int8.so') as picker:
    probabilities = picker.predict_batch(batch_b3l)
```

## 测速、准确率和边缘部署

完整 [设备速度矩阵](reports/DEVICE_MATRIX.md) 覆盖两套模型 × M4 Max CPU/MLX、3955WX CPU、RTX5090cuDNN，FP32/量化两模式：单条6144/10240、实际B4096概率、完整实时拾取，以及N26001历史拾取。每项注明线程、重复次数、同步、精度和内存口径。CPU INT8更快；GPU/MLX上的INT8并不保证更快，China大batch本次以FP32更快。

[China模型验证](reports/PHASENET.md) 和 [PNSN INT8验证](reports/INT8_CPU.md) 提供原模型对照、5730条缓存标签评估及量化漂移。中国UNet是现有最近邻上采样变体，结构不同于原论文标准PhaseNet，不能当作标准PhaseNet权重。量化会改变概率和拾取，需按目标数据核验。

边缘构建：`-DBUILD_SHARED_LIBS=OFF -DPNSN_OPENMP=OFF`，较小窗口如6144，合理的 `max_candidates` 和小batch。CPU有NEON/DOTPROD、AVX2及标量回退。L10240时INT8 CPU context（不含调用方输入/输出、临时NMS）PNSN约1.76MiB、China约2.88MiB；L6144约1.31/1.85MiB。China INT8模型175724字节，PNSN477012字节。物理边缘板未实测，不能以M4或RTX5090吞吐替代。

Linux/macOS/Windows静态/共享C构建由CI验证；GPU/MLX以设备实测为准。模型格式为版本化小端容器，记录SHA256，严格校验结构/有限值/尾部。导出或更新权重：

```sh
python native/tools/export.py --precision fp32 --output native/models/pnsn-fp32.bin
python native/tools/export.py --precision int8 --output native/models/pnsn-int8.bin
python native/tools/export_unet.py --precision fp32 --output native/models/china-unet-fp32.bin
python native/tools/export_unet.py --precision int8 --output native/models/china-unet-int8.bin
python native/tests/validate.py
python native/tests/validate_batch.py
python native/tests/validate_unet.py --output native/reports/unet-validation.json
```

发布仓库及模型来源见 [根目录README](../README.md) 和 [NOTICE](../NOTICE.md)。旧PNSN专用说明存于 [v0.1.3说明](reports/PNSN_V013_GUIDE.md)。
