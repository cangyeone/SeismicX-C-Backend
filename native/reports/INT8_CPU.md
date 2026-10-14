# cuDNN INT8 算术与 CPU 速度（v0.1.3）

测试日期2026-10-14。原始PNSN v3 checkpoint及所有14个BN融合保持不变，无重新训练。两种策略继续使用相同C API：历史 `[3,N]` 和实时独立批 `[B,3,L]`。

## 实际计算精度

新增 `libpnsn_cudnn_int8`，加载 `pnsn-int8.bin`：编码器7层、两层LSTM后的投影卷积、解码器7层转置卷积及最终分类卷积，全17层均使用 cuDNN INT8输入/权重与INT32累加。7个转置卷积分别拆成两个奇偶相位的普通卷积。首层3通道补零到4通道；每条记录、每层按最大绝对值动态量化激活，权重逐输出通道量化。内部特征为NHWC，普通卷积的整数结果在原输出缓冲区还原，减少临时内存及拷贝。

cuDNN9.10.2的双向LSTM使用FP32计算和还原后的Q8权重；池化、层间输出、非线性与softmax也是FP32。因此此后端是 **卷积W8A8 / LSTM FP32**。CPU使用真正的W8A8卷积及LSTM矩阵乘法、INT32累加，LSTM状态/非线性保持FP32；CPU逐patch量化激活，与GPU整条记录的量化规则不同。

原 `libpnsn_cudnn` 保留Q8存储/FP32计算路径。新库加载Q8文件时无FP32卷积回退；不支持的INT8算法会明确返回错误。cuDNN的输入INT8、累加INT32、输出FLOAT配置有 [官方支持说明](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-cnn-library.html)；RNN数据类型限制见 [官方RNN接口](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-adv-library.html)。此次验证了INT8算术，未以profiler确认Tensor Core指令使用情况。

## 4096×3×10240 整批概率速度

相同seed=20261014独立float32记录。计时包括内部float64去均值/归一化和概率输出分配，GPU包括H2D/D2H，排除造数、模型载入与NMS。GPU为5次暖态中位数；CPU为3次暖态中位数，均单线程、B内逐条SIMD计算。

| 硬件 / 后端 | 实际计算 | 整批秒数 | 条/秒 |
|---|---|---:|---:|
| RTX5090 / cuDNN | FP32 | 0.263144 | 15565.6 |
| RTX5090 / cuDNN INT8 | 卷积W8A8 / LSTM FP32 | 0.258891 | 15821.3 |
| M4 Max / C CPU，1线程 | FP32 | 28.228936 | 145.1 |
| M4 Max / C CPU，1线程 | W8A8 | 20.587952 | 199.0 |
| 3955WX / C CPU，1线程 | W8A8 | 55.654380 | 73.6 |

M4 Max的CPU INT8较FP32快 **1.37倍**，耗时减少约27.1%。3955WX是WSL2环境中的AMD Ryzen Threadripper PRO 3955WX，16核心/32逻辑CPU，本次只使用1线程；这项结果不代表整机多核吞吐。本机为Apple M4 Max；ARM NEON+DOTPROD启用，服务器启用AVX2 INT8点积。GPU测试使用GPU0、CUDA12.8、cuDNN9.10.2，FP32关闭TF32。

GPU网络事件中位数：FP32 **0.104541秒**，混合INT8 **0.100001秒**；网络耗时降低约4.34%，概率端到端降低约1.62%。目前混合INT8速度收益有限，LSTM保持FP32，量化/还原也需要时间。输入480MiB、输出800MiB，主机传输对端到端影响较大。

FP32端到端p95为0.264869秒，INT8为0.260816秒。完整 `pick_batch` 含逐记录最小堆NMS单次实测：FP32 **0.328851秒**（48拾取），INT8 **0.327836秒**（60拾取）。合成波形候选少，不能代表事件密集数据的NMS开销。这些合成拾取数不作为准确率指标。

混合INT8保留显式GPU缓冲区7,468,613,980字节（约6.96GiB），FP32为6,461,632,344字节（6.02GiB）。这是权重、张量、显式工作区之和，不是总进程峰值显存；INT8路径还保留FP32激活与额外量化缓冲区，GPU内存没有随权重文件等比例下降。

## CPU 单条实测

相同原始float32输入，内部float64预处理，输出概率，不含NMS。3次预热后50次测量，下面为中位数；两台主机均1线程。

| CPU | 长度 | FP32毫秒 | INT8毫秒 |
|---|---:|---:|---:|
| M4 Max | 6144 | 4.067 | 2.924 |
| M4 Max | 10240 | 6.795 | 4.949 |
| 3955WX | 6144 | 14.120 | 8.125 |
| 3955WX | 10240 | 23.997 | 13.076 |

CPU整批结果来自实际完整4096批调用，没有用单条时间外推。多核批并行本次未实现/测量；库的现有OpenMP是可选层内并行，此表设置threads=1。

## 算子、边界与标签验证

- 独立C/CUDA测试使用每条最大值恰为127的整数输入，使激活scale=1；17层、B=3的INT32累加结果与原始完整卷积/转置卷积CPU公式完全一致，最大整数误差0，还原后误差0。转置参考没有复用奇偶拆分公式。
- `B=1/3/5`、`L=128/6144/10240`、历史任意N、记录置换、真实波形、raw+diff、空输入、NaN、int32极值与容量不足原子性通过。完整4096批概率均有限且和为1；抽取首/中/末记录与独立单窗的最大概率差0.0008384。
- 与独立PyTorch浮点模拟量化图对比，合成窗口最大概率差约0.00268，真实波形约0.01198；这是完整混合图的数值差异，不按FP32的1e-5门限宣称逐点一致。整数算子校验与实际标签结果分别报告。
- CMake CPU单元/批单元、服务器INT8整数测试通过，独立C CLI可执行；CPU核心未改动。

既有31省、5730条有效测试窗口，2020–2022数据，只读，无训练，排除cache尾部填充。人工标签只评Pg/Sg，阈值0.1、NMS间隔300、匹配容差50点；Pn/Sn仅与原始FP32预测配对。

| 后端 | Pg/Sg F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| 原始PyTorch / cuDNN FP32 | 86.224309% | 9733 | 1547 | 1563 |
| cuDNN卷积INT8 / LSTM FP32 | 85.951809% | 9667 | 1531 | 1629 |

混合INT8 F1下降 **0.2725个百分点**。四类参考预测11838个，混合INT8预测11728个，其中11527个在50点内配对（参考匹配率97.3729%）；配对到时误差中位数1点、p95为5点。相对原始FP32的整个测试集最大概率漂移 **0.8706**、平均漂移0.00017766，表明部分样本的量化变化较大。这项F1属于GPU混合后端，不能套用到使用不同激活量化方法的CPU后端。

## 使用与复现

编译 `-DPNSN_CUDNN=ON -DCUDNN_ROOT=/path/to/cudnn` 会同时生成FP32和混合INT8库/CLI；C程序链接 `pnsn_cudnn_int8`。推理不依赖Python/PyTorch。

```python
with Picker('native/models/pnsn-int8.bin',
            library='native/build/libpnsn_cudnn_int8.so') as gpu:
    history_picks = gpu.pick_history(history_3n)
    probabilities = gpu.predict_batch(realtime_b3l)
    picks = gpu.pick_batch(realtime_b3l)

with Picker('native/models/pnsn-int8.bin', threads=1) as cpu:
    probabilities = cpu.predict_batch(realtime_b3l)
```

```sh
python native/tests/validate_cudnn.py --int8 --output native/reports/cudnn-int8-validation.json
python native/tests/validate_batch.py --library native/build/libpnsn_cudnn_int8.so
ctest --test-dir native/build --output-on-failure
python native/tools/benchmark_batch_native.py --backend cudnn_int8 --precision int8 --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cudnn-int8.json
python native/tools/benchmark_batch_native.py --backend cpu --precision int8 --batch 4096 --length 10240 --repeats 3 --output native/reports/batch4096-cpu-int8.json
python native/tools/benchmark_cpu.py --output native/reports/cpu-single.json
python native/tools/evaluate_labeled.py --backend cudnn_int8 --cache /path/to/existing/cache --output native/reports/cudnn-int8-labeled-evaluation.json
```

证据：[GPU INT8速度](batch4096-native-cudnn-int8.json)、[GPU FP32同期重测](batch4096-cudnn-fp32-v013.json)、[本机CPU整批](batch4096-cpu-v013.json)、[服务器CPU整批](batch4096-cpu-3955wx-int8.json)、[本机CPU单条](cpu-single-m4max-v013.json)、[服务器CPU单条](cpu-single-3955wx-v013.json)、[整数校验](cudnn-int8-ops-validation.json)、[混合图对照](cudnn-int8-validation.json)、[批接口](batch-cudnn-int8-validation.json)、[标签](cudnn-int8-labeled-evaluation.json)。
