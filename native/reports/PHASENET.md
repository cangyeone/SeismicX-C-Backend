# China PhaseNet 风格 UNet：C与8位适配（v0.2.0）

使用用户明确选定的 `ckpt/china.unet.pt`，参考结构为同一上游仓库的 `models/UNet.py`，checkpoint SHA256：`ec74a633b2c9a448089b05ab3a4ea54ef7377c2b961d1518897c89bc3e7d9b00`。这是最近邻上采样的中国数据UNet变体，结构不同于标准AI4EPS/SeisBench PhaseNet；本次无重训练、无标准PhaseNet权重替换。

## 结构与执行

11层7点卷积，编码通道8/16/32/64/128，四层stride4；解码使用nearest×4后卷积并拼接对应skip（skip在前），输出Noise/P/S三类softmax。10个BatchNorm在导出时用float64融合到卷积权重/偏置，运行时没有BN算子。固定L必须为256的倍数，默认10240，6144可用；历史 `[3,N]` 中N任意。通用C加载器根据模型magic切换图与类数，并严格核对形状和文件长度。

CPU为C11，自写SIMD/标量点积；INT8逐输出通道权重量化，逐卷积输入片段动态激活量化，INT32累加、float反量化/ReLU/softmax。最近邻卷积直接从原始输入按time/4取样，无需CPU物化扩大后的张量。GPU FP32由cuDNN卷积执行。GPU INT8使用与CPU相同的逐片段量化，按128条record分块im2col再调用cuDNN INT8 1×1卷积，含最后不足128条的分块。MLX Q8为W8A32，精度模式独立标注。

| 模型文件 | 字节 | SHA256 |
|---|---:|---|
| china-unet-fp32.bin | 692828 | af32d225cc4174de8867026755aa2209657eaa866e806bbaf2359f36790e1fcd |
| china-unet-int8.bin | 175724 | 18760efea2475dcea4c5b8082bf2330f37c76820f60c00ee71dd8ddfaf4859cc |

## 数值与边界验证

[本机CPU/MLX回归](unet-local-validation.json)、[cuDNN FP32](unet-cudnn-validation.json)、[cuDNN INT8](unet-cudnn-int8-validation.json) 覆盖固定256/6144/10240、独立批次/排列、129条非整块batch、历史0/1/2/尾窗、int32极值、float64大直流偏置、常量/空输入、NaN拒绝、容量不足原子返回、真实26001点波形与raw+diff。

真实窗口FP32概率相对原PyTorch最大绝对误差约1.4e-6（CPU）/1.0e-6（GPU），量化最大绝对误差约0.1808、平均约0.000324。INT8不是数值无损；合成随机波形平均误差约1e-4，不能据此替代真实标签评估。cuDNN整数算子独立测试核对11层、B3的完整stride/nearest卷积公式，INT32累加最大误差0。CPU静态/共享及AddressSanitizer/UndefinedBehaviorSanitizer回归均通过。

## 标签评估：仅Pg/Sg映射范围

只读现有2020–2022年test缓存，31省，共5730个实际test.json条目（排除缓存补齐slots），variant0；PyTorch参考禁用TF32，不重新训练。现有人工标签是Pg/Sg，因此将China模型P/S输出映射到Pg/Sg进行对照。该结果仅代表此缓存/映射范围，不是标准PhaseNet全数据集准确率，也不能直接与PNSN四相指标比较。阈值0.1、NMS gap300采样点、匹配容差±50采样点（100Hz为±0.5秒）。

| 模式 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 原PyTorch FP32 | 7122 | 7797 | 4174 | 47.7378% | 63.0489% | 54.3353% |
| C/cuDNN FP32 | 7122 | 7797 | 4174 | 47.7378% | 63.0489% | 54.3353% |
| C/cuDNN逐片段INT8 | 7140 | 7896 | 4156 | 47.4860% | 63.2082% | 54.2306% |

INT8的F1降低约0.1047个百分点。CPU及cuDNN的TP/FP/FN一致。相对FP32参考的14919个拾取，CPUINT8有14404个匹配（96.548%），匹配到时p95误差4点，概率最大误差0.7656、平均0.0003012。局部概率误差可很大，不能只依据全局平均误差宣称无损。

证据：[CPU标签评估](unet-cpu-int8-labeled-evaluation.json)、[cuDNN INT8标签评估](unet-cudnn-int8-labeled-evaluation.json)。缓存数据未公开，仅发布聚合指标。复现标签评估需要同一已标注缓存，不应假定公开示例即可复现这套F1。

## 速度与部署选择

[完整设备矩阵](DEVICE_MATRIX.md) 包含两种策略、FP32/量化、6144/10240单条、真实B4096批次和内存。China的B4096×3×10240概率输出中位数：M4 CPU1线程FP32/INT8为22.3118/18.6468秒，3955WX CPU1线程67.1085/34.8759秒，M4 MLX FP32/W8A32为7.6460/21.8450秒，RTX5090cuDNN FP32/INT8为0.1869/0.3561秒。

CPU INT8提速，当前GPU大batch INT8更慢。逐片段量化保留了与CPU一致的量化粒度和标签指标，同时增加im2col/量化开销。此GPU上FP32是测得的最快选项。cuDNN所分配工作区/张量约FP3213.75GiB、INT89.03GiB（非进程峰值）；MLX峰值active约26.48/39.56GiB。物理边缘板尚未测速，推荐从静态CPU INT8、小窗口/小batch及显式候选预算开始实机评估。

```sh
python native/tools/export_unet.py --precision int8 --output native/models/china-unet-int8.bin
python native/tests/validate_unet.py --output native/reports/unet-rerun.json
python native/tools/evaluate_labeled.py --model china-unet --backend cudnn_int8 --cache /path/to/existing/cache --output native/reports/unet-labels-rerun.json
```
