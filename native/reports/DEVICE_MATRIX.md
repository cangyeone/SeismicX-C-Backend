# SeismicX-C-Backend：两个模型的设备实测（v0.2.0）

测试日期：2026-10-14。所有表格均来自实际运行，未用单条时间乘4096估算。PNSN为五类BRNNPNSN v3；China UNet为用户选定的 `china.unet.pt` 三类PhaseNet风格模型。

## 设备、精度与口径

- 本机：Apple M4 Max，128 GiB统一内存，macOS ARM64；CPU启用NEON/DOTPROD，MLX 0.32.3使用Metal GPU。
- 服务器：AMD Ryzen Threadripper PRO 3955WX（16核/32线程），WSL2 Linux；CPU启用AVX2 INT8/SSE FP32。CPU表格全部是 **1线程**，不是整机多核吞吐。
- NVIDIA：RTX 5090，32 GiB显存，GPU0，CUDA 12.8 / cuDNN 9.10.2，FP32参考禁用TF32。
- CPU INT8为INT8×INT8、INT32累加；PNSN的LSTM矩阵乘法也量化，状态/非线性保持float。
- cuDNN INT8：China全部11层卷积为W8A8，逐输入片段动态量化；PNSN全部17层卷积为W8A8，双向LSTM保持FP32。整数算术已与独立参考公式核对；未用profiler确认Tensor Core指令。
- MLX的8位模式是 **W8A32**（量化权重、浮点激活），不能称为全图INT8算术。

## 实时批次 `[4096,3,10240]`

表内单位为秒，FP32 / INT8（MLX为Q8 W8A32）。原始host输入到host概率输出，包含内部预处理、输出分配和GPU传输/同步，不含模型加载、波形生成、验证和NMS。CPU/MLX取3次热运行中位数，cuDNN取5次；冷启动和p95见原始JSON。CPU逐记录SIMD执行，GPU按层处理整个batch，China INT8内部im2col分块128条。

| 模型 | M4 CPU | M4 MLX | 3955WX CPU | RTX5090 cuDNN |
|---|---:|---:|---:|---:|
| pnsn | 28.228936 / 20.587952 | 13.453426 / 13.842837 | 96.173383 / 55.486145 | 0.263144 / 0.258891 |
| china-unet | 22.311756 / 18.646826 | 7.646043 / 21.844968 | 67.108526 / 34.875928 | 0.186889 / 0.356087 |

M4 CPU的INT8提速：PNSN约1.37倍、China约1.20倍；3955WX CPU分别约1.73倍、1.92倍。RTX5090的PNSN混合INT8端到端仅约1.02倍，不能据此宣称大幅加速；China INT8耗时约为FP32的1.91倍。China逐片段量化/im2col及小通道卷积的开销抵消了INT8乘法优势。MLX的大批次Q8也较FP32慢，尤其China模型。当前这两台GPU上，China大batch优先使用FP32；这项选择依据本次测量，不能外推到未测试的边缘设备。

## 单窗口 `[1,3,L]`

单位毫秒，FP32 / INT8（MLX W8A32）。3次预热后50次运行中位数，host原始输入到host概率，包含预处理/分配/传输，无NMS。cuDNN FP32取普通库，INT8取INT8库；两套单独测量的FP32也保留在原始报告中，可观察运行波动。

| 模型 / L | M4 CPU | M4 MLX | 3955WX CPU | RTX5090 cuDNN |
|---|---:|---:|---:|---:|
| pnsn / 6144 | 4.067 / 2.924 | 1.282 / 1.344 | 14.120 / 8.125 | 4.200 / 5.198 |
| pnsn / 10240 | 6.795 / 4.949 | 1.638 / 1.766 | 23.997 / 13.076 | 6.138 / 7.711 |
| china-unet / 6144 | 3.277 / 2.788 | 0.816 / 0.823 | 10.064 / 4.958 | 0.736 / 0.688 |
| china-unet / 10240 | 5.426 / 4.558 | 1.016 / 1.021 | 16.177 / 8.131 | 0.753 / 0.713 |

## 历史策略 `[3,N]` 完整拾取

公共三分量SAC示例，N=26001（100Hz，260.01秒），float64输入；窗口10240、重叠1024。表内单位毫秒，FP32 / INT8（MLX W8A32），包含滑窗、预处理、推理、全局最小堆NMS和GPU同步；2次预热后5次中位数。该波形候选数量较少，不能代替密集震相场景的NMS压力测试。

| 模型 | M4 CPU | M4 MLX | 3955WX CPU | RTX5090 cuDNN |
|---|---:|---:|---:|---:|
| pnsn | 20.019 / 15.259 | 11.589 / 11.068 | 69.981 / 42.149 | 17.910 / 20.613 |
| china-unet | 15.779 / 12.821 | 4.256 / 3.294 | 48.472 / 25.683 | 2.662 / 2.756 |

## 实时完整拾取与内存

完整批次拾取同样实际跑完4096条，逐记录独立NMS。每模式各运行1次，因此只报告该次耗时，不称中位数。输入为相同seed的高斯波形、稀疏候选；不同量化方式改变概率和候选数。

| 模型 | 设备 | 精度 | 完整拾取秒 | 拾取数 | GPU内存 GiB |
|---|---|---|---:|---:|---:|
| pnsn | M4 Max CPU (1 thread) | fp32 | 28.199810 | 48 | — |
| pnsn | M4 Max CPU (1 thread) | int8 | 20.539214 | 68 | — |
| pnsn | M4 Max MLX/Metal | fp32 | 13.923660 | 48 | 20.74 |
| pnsn | M4 Max MLX/Metal | int8 | 16.673782 | 63 | 30.57 |
| pnsn | 3955WX CPU (1 thread) | fp32 | 96.272314 | 48 | — |
| pnsn | 3955WX CPU (1 thread) | int8 | 55.550253 | 67 | — |
| pnsn | RTX 5090 cuDNN | fp32 | 0.328851 | 48 | 6.02 |
| pnsn | RTX 5090 cuDNN | int8 | 0.327836 | 60 | 6.96 |
| china-unet | M4 Max CPU (1 thread) | fp32 | 22.715076 | 28 | — |
| china-unet | M4 Max CPU (1 thread) | int8 | 19.000436 | 29 | — |
| china-unet | M4 Max MLX/Metal | fp32 | 8.221567 | 28 | 26.48 |
| china-unet | M4 Max MLX/Metal | int8 | 25.504794 | 24 | 39.56 |
| china-unet | 3955WX CPU (1 thread) | fp32 | 69.037791 | 28 | — |
| china-unet | 3955WX CPU (1 thread) | int8 | 34.395033 | 29 | — |
| china-unet | RTX 5090 cuDNN | fp32 | 0.240115 | 28 | 13.75 |
| china-unet | RTX 5090 cuDNN | int8 | 0.422733 | 29 | 9.03 |

cuDNN内存统计为库所分配的张量、权重和workspace，不是进程显存峰值，未包含驱动/handle/caller host缓冲。MLX统计为其峰值active bytes，属于统一内存；二者口径不同。INT8权重文件变小不保证整个GPU工作区变小。B4096本身的float32输入为480 MiB，概率输出PNSN为800 MiB、China为480 MiB。

物理ARM边缘板、Windows设备以及其他GPU未做速度测量；静态C库和标量回退支持作为移植起点。不要把M4 Max速度当作低功耗ARM速度。

## 复现与证据

先构建库，安装NumPy；MLX测试额外安装MLX，导出/对照测试需要PyTorch。历史示例转换额外需要ObsPy。GPU环境安装CUDA/cuDNN并编译对应库。

```sh
python native/tools/prepare_example.py
python native/tools/benchmark_history.py --backend cpu --output native/reports/history-rerun.json
python native/tools/benchmark_device.py --model china-unet --backend cpu --output native/reports/single-rerun.json
python native/tools/benchmark_batch_native.py --model china-unet --backend cpu --precision both --batch 4096 --length 10240 --repeats 3 --post --output native/reports/batch-rerun.json
# 换 --model pnsn；设备换 --backend mlx / cudnn / cudnn_int8。
# cuDNN两套精度分别跑 cudnn --precision fp32 与 cudnn_int8 --precision int8。
# 重新汇总本次版本化原始报告：
python native/tools/report_device_matrix.py
```

[汇总JSON](device-matrix.json)保留每行来源、精度、p95、重复次数、内存和拾取数量。原始报告：

- [batch4096-cpu-v013.json](batch4096-cpu-v013.json)
- [batch4096-cudnn-fp32-v013.json](batch4096-cudnn-fp32-v013.json)
- [batch4096-mlx.json](batch4096-mlx.json)
- [batch4096-native-cudnn-int8.json](batch4096-native-cudnn-int8.json)
- [cpu-single-3955wx-v013.json](cpu-single-3955wx-v013.json)
- [cpu-single-m4max-v013.json](cpu-single-m4max-v013.json)
- [history-3955wx-cpu-v020.json](history-3955wx-cpu-v020.json)
- [history-5090-cudnn-int8-v020.json](history-5090-cudnn-int8-v020.json)
- [history-5090-cudnn-v020.json](history-5090-cudnn-v020.json)
- [history-m4max-cpu-v020.json](history-m4max-cpu-v020.json)
- [history-m4max-mlx-v020.json](history-m4max-mlx-v020.json)
- [pnsn-batch4096-cpu-3955wx-v020.json](pnsn-batch4096-cpu-3955wx-v020.json)
- [pnsn-batch4096-cpu-picks-m4max.json](pnsn-batch4096-cpu-picks-m4max.json)
- [pnsn-single-cudnn-int8-v020.json](pnsn-single-cudnn-int8-v020.json)
- [pnsn-single-cudnn-v020.json](pnsn-single-cudnn-v020.json)
- [pnsn-single-mlx-v020.json](pnsn-single-mlx-v020.json)
- [unet-batch4096-cpu-3955wx-v020.json](unet-batch4096-cpu-3955wx-v020.json)
- [unet-batch4096-cpu-m4max.json](unet-batch4096-cpu-m4max.json)
- [unet-batch4096-cpu-picks-m4max.json](unet-batch4096-cpu-picks-m4max.json)
- [unet-batch4096-cudnn-fp32.json](unet-batch4096-cudnn-fp32.json)
- [unet-batch4096-cudnn-int8.json](unet-batch4096-cudnn-int8.json)
- [unet-batch4096-mlx.json](unet-batch4096-mlx.json)
- [unet-cpu-single-m4max.json](unet-cpu-single-m4max.json)
- [unet-single-cpu-3955wx-v020.json](unet-single-cpu-3955wx-v020.json)
- [unet-single-cudnn-int8-v020.json](unet-single-cudnn-int8-v020.json)
- [unet-single-cudnn-v020.json](unet-single-cudnn-v020.json)
- [unet-single-mlx-v020.json](unet-single-mlx-v020.json)

数值和标签验证分别见 [China UNet](PHASENET.md) 与 [PNSN INT8](INT8_CPU.md)。早期0.1.x报告作为历史对照保留；本表明确选取当前算术模式的来源，不混用Q8存储/FP32计算的旧结果。
