v0.1.3新增真实INT8卷积后端，见 [INT8与CPU报告](INT8_CPU.md)。本页保留v0.1.2的Q8存储/FP32计算结果。

# cuDNN 后端实测（v0.1.2）

日期2026-10-14；remote CUDA server，RTX5090 GPU0，CUDA12.8，cuDNN9.10.2。目录 `/path/to/SeismicX-C-Backend`。复用已安装库，未安装或更换服务器软件。

公共 C API 不变：历史数据 `[3,N]` 用 `pick_history`，实时数据 `[B,3,L]` 用 `predict_batch` / `pick_batch`。卷积、转置卷积、池化、两个双向LSTM调用cuDNN；BN已经融合入权重，内部float64预处理及C最小堆NMS保留。Python仅为可选接口，C CLI推理无需Python或torch。

## Batch4096、L10240

4096条独立合成float32波形，5次暖态中位数。概率端到端包括原始主机输入、内部GPU去均值/归一化、H2D/D2H及输出分配，不包括造数、加载模型和NMS。

| 后端 | 网络阶段（秒） | 主机端到端（秒/批） | 主机端到端条/秒 |
|---|---:|---:|---:|
| 自写CUDA FP32 | 1.393185 | 1.555080 | 2633.9 |
| 自写CUDA W8A8 | 1.265904 | 1.425481 | 2873.4 |
| cuDNN FP32 | 0.104651 | 0.263527 | 15543.0 |
| cuDNN Q8存储/FP32计算 | 0.104229 | 0.263339 | 15554.1 |
| 原始PyTorch CUDA FP32，同机重测 | 0.133633 | 0.291919 | 14031.3 |

cuDNN FP32较自写CUDA FP32端到端快5.90倍；比原始PyTorch参考耗时低9.73%。网络用CUDA event计时，PyTorch驻留阶段用同步wall clock；两者均关闭TF32。BN融合和工作区复用也影响对比结果，不能将差异全部归因于C语言。

完整 `pick_batch` 含逐记录NMS：FP32单次0.330712秒，Q8存储单次0.335597秒。合成输入候选少，分别48/63个拾取，不能代表事件密集数据后处理压力。

FP32概率端到端p95为0.265872秒。Q8存储出现主机计时波动，p95为0.364200秒；原始5次时间完整保留，不能据此认定Q8更快。两种模型的GPU缓冲区相同，计6,461,632,344字节（6.02GiB）；这是实际保留的张量、权重和显式算法工作区，不是总进程峰值显存，不含driver及cuDNN内部handle开销。概率输入480MiB、输出800MiB，主机传输占比已经较大。

## 数值与已有标签

与独立PyTorch图对照，覆盖B=1/3、L=128/6144/10240、float64大DC及实际公开波形。FP32与原始未融合checkpoint的合成批最大概率误差8.94e-7；真实波形与融合图误差2.86e-6。Q8权重还原后与相同还原权重PyTorch图的真实波形误差2.95e-6。

标签验证使用只读31省缓存，有效test.json行数共5730；窗口10240，既有2020–2022测试数据，无训练。阈值0.1，NMS间隔300，匹配容差50个采样点（100Hz下0.5秒）。人工准确率只评Pg/Sg；Pn/Sn仅比较FP32预测。

| 模型 | Pg/Sg合并F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| 原始PyTorch FP32 | 86.224309% | 9733 | 1547 | 1563 |
| cuDNN FP32 | 86.224309% | 9733 | 1547 | 1563 |
| cuDNN Q8存储/FP32计算 | 86.178790% | 9727 | 1551 | 1569 |

cuDNN FP32全部11838个预测与参考在50点内匹配，95%匹配到时误差为0点，最大8点，整个测试集最大概率误差1.03e-5。Q8存储有11833个预测、11784个匹配（参考匹配率99.5438%），F1下降0.04552个百分点，95%匹配到时误差2点；最大概率漂移0.1382。这是权重量化漂移，和旧自写CUDA W8A8结果分别报告。

cuDNN后端当前使用FP32算术。加载Q8文件后，仅一次还原为FP32权重；GPU激活、LSTM状态和权重均是FP32。模型文件仍477012字节，但不能声称已使用cuDNN INT8/Tensor Core计算。CPU与自写CUDA W8A8路径继续保留。

独立C CLI在没有启动Python的进程中执行通过；固定概率与本机CPU参考最大误差2.98e-7，26001点历史波形的3个拾取震相与采样点完全一致。另通过B=1/3/5批接口回归、空输入、NaN、int32极值、记录置换、raw+diff与容量不足原子性检查；原历史接口和最小堆随机对照回归也通过。

## 复现及范围

编译见 [README](../README.zh.md)。使用cuDNN传统C卷积API和v8 RNN API；传统卷积API已deprecated，此次在9.10.2实测。仅本服务器NVIDIA硬件已测试，未验证其他GPU/Jetson或未来cuDNN版本。

```sh
python native/tests/validate_cudnn.py --output native/reports/cudnn-validation.json
python native/tests/validate_batch.py --library native/build/libpnsn_cudnn.so --output native/reports/batch-cudnn-validation.json
python native/tests/validate.py --library native/build/libpnsn_cudnn.so --output native/reports/cudnn-legacy-regression.json
python native/tools/benchmark_batch_native.py --backend cudnn --batch 4096 --length 10240 --repeats 5 --post --output native/reports/batch4096-native-cudnn.json
python native/tools/benchmark_batch_torch.py --batch 4096 --length 10240 --repeats 5 --output native/reports/batch4096-torch-cudnn-comparison.json
python native/tools/evaluate_labeled.py --backend cudnn --cache /path/to/existing/cache --output native/reports/cudnn-labeled-evaluation.json
```

原始证据：[批速度](batch4096-native-cudnn.json)、[PyTorch重测](batch4096-torch-cudnn-comparison.json)、[数值](cudnn-validation.json)、[批隔离](batch-cudnn-validation.json)、[标签](cudnn-labeled-evaluation.json)、[老接口回归](cudnn-legacy-regression.json)。
