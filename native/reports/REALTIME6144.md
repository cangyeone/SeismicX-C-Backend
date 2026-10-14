# 6144点近实时拾取：4096台实测与10/30秒容量预算

日期2026-10-14；每台每周期重新推理完整 `[3,6144]`，单模型，未启用raw+diff。按参考模型100Hz，窗口覆盖61.44秒。更新周期为10秒或30秒，窗口发生重叠；这些容量按每周期一窗/台计算，不把61.44秒窗口长度当作更新周期。

## 实测与公式

本次真正跑了4096条记录，每条为上游公共真实三分量SAC例子的随机6144点切片，并叠加0.5%逐通道标准差的独立噪声。不是实际4096台站采集，也没有直接验证十万台规模。完整计时包含内部float64预处理、推理、最小堆NMS、CPU任务提交及GPU传输/同步；不含数据生成、模型加载、网络采集、文件解码/重采样、结果落库和跨周期去重。CPU重复3次，GPU5次。模型/线程池常驻。

CPU是多个独立context按台站分片并行，每context内部threads=1。M4 Max有12个性能核心+4个能效核心，测试16个worker；3955WX有16核/32逻辑线程，测试32个worker。不是仅把一个CPU context的threads参数调为16/32。GPU仍使用单个完整B4096调用。

```text
理论吞吐容量 = floor(4096 × 周期秒 / 实测中位耗时)
70%时间预算容量 = floor(4096 × 周期秒 × 0.7 / 本次最慢耗时)
固定完整4096分块预算 = floor(周期秒 × 0.7 / 本次最慢耗时) × 4096
```

“理论吞吐”只用于计算上限；“70%预算”保留30%周期时间应对采集、调度与抖动，是规划估算而不是稳定运行保证。后一个完整分块预算更保守。超过4096台的所有容量均为外推，应在真实采集链路和台站数据上做长期压力测试。

| 设备 / worker | 模型 / 精度 | 4096台中位秒 | 本次最慢秒 | 理论10秒台 | 理论30秒台 | 70%预算10秒台 | 70%预算30秒台 |
|---|---|---:|---:|---:|---:|---:|---:|
| M4 Max CPU / 16 | pnsn / int8 | 1.056103 | 1.130876 | 38,784 | 116,352 | 25,353 | 76,061 |
| M4 Max CPU / 16 | china-unet / int8 | 0.957158 | 0.958267 | 42,793 | 128,380 | 29,920 | 89,762 |
| 3955WX CPU / 32 | pnsn / int8 | 1.741338 | 1.766185 | 23,522 | 70,566 | 16,233 | 48,701 |
| 3955WX CPU / 32 | china-unet / int8 | 1.225498 | 1.621072 | 33,423 | 100,269 | 17,687 | 53,061 |
| RTX5090 cuDNN / 1 | pnsn / fp32 | 0.290938 | 0.291955 | 140,786 | 422,358 | 98,206 | 294,620 |
| RTX5090 cuDNN / 1 | china-unet / fp32 | 0.192013 | 0.195591 | 213,318 | 639,955 | 146,591 | 439,775 |
| RTX5090 cuDNN / 1 | pnsn / int8 | 0.277359 | 0.281514 | 147,678 | 443,036 | 101,849 | 305,547 |
| RTX5090 cuDNN / 1 | china-unet / int8 | 0.291911 | 0.292615 | 140,316 | 420,950 | 97,985 | 293,956 |

3955WX China32-worker首次整批1.6211秒，后两次约1.2031/1.2255秒，因此规划采用首次较慢值，不能只取最快值。GPU中PNSN混合INT8略快，China FP32明显快于INT8；PNSN cuDNN INT8的LSTM仍为FP32。CPU INT8的矩阵乘法为W8A8，状态/非线性保持float。

## 4096台与默认串行CPU的区别

| CPU / 模型 | 默认单context、1线程完整秒 | 10秒周期 | 30秒周期 |
|---|---:|---|---|
| M4 Max CPU / pnsn INT8 | 12.463 | 超时 | 能完成 |
| M4 Max CPU / china-unet INT8 | 11.393 | 超时 | 能完成 |
| 3955WX CPU / pnsn INT8 | 33.401 | 超时 | 超时 |
| 3955WX CPU / china-unet INT8 | 21.107 | 超时 | 能完成 |

本次CPU分片并行后，两台CPU的两个模型均能在10秒内处理4096台。CUDA完整拾取约0.2–0.3秒，同样满足10秒与30秒周期。若两个模型同时跑同一台，容量应按两者耗时之和重新计算，不能分别把各自上限都用满。

## 单次batch、显存与周期总台数

B4096、L6144的float32输入为288MiB；host概率输出PNSN480MiB，China288MiB。cuDNN库统计的GPU张量/权重/workspace如下，未包含驱动/handle/caller host内存，不等于进程峰值：

| 模型 / 精度 | 库分配GiB |
|---|---:|
| pnsn / fp32 | 3.627 |
| china-unet / fp32 | 8.451 |
| pnsn / int8 | 4.190 |
| china-unet / int8 | 5.418 |

当前GPU实现的整数索引检查 `batch <= INT_MAX / (L × factor)`，PNSN factor8、China factor32。L6144对应PNSN43,690、China10,922，属于代码索引上界；显存/算法/工作区仍可能使一次调用的可用B更小，未通过OOM搜索验证该上界。因此不要将43,690称为RTX5090已实测单批容量。

生产调度可保持GPU B=4096（本次已实测），超过4096台就分批。每周期的总台数可以大于单次B。CPU按worker拆分台站、每worker复用独立context。float64输入占用两倍host输入空间，GPU传输速度可能变化；本表测的是float32输入、内部float64预处理。

以100Hz、三分量int32/float32计算，新增波形为每台约1200字节/秒：10万台约120MB/s的净波形，尚未算协议、解码和存储成本。每台6144点float32环形窗占73728字节（72KiB），10万台约6.87GiB；窗口/环形缓存应逐台维护，每次新增10/30秒而复用历史。

10秒窗口步进相对61.44秒窗重叠51.44秒；30秒步进重叠31.44秒。跨周期需要按台站、绝对到时及相位去重。模型调用内的独立NMS只处理当前窗口，不会自动消除前一周期的重复拾取。

## 复现

```sh
python native/tools/prepare_example.py
python native/tools/benchmark_realtime.py --backend cpu --workers 1 8 16 --repeats 3 --output native/reports/realtime-local.json
# 3955WX: --workers 1 16 32
python native/tools/benchmark_realtime.py --backend cudnn --precision fp32 --repeats 5 --output native/reports/realtime-gpu-fp32.json
python native/tools/benchmark_realtime.py --backend cudnn_int8 --precision int8 --repeats 5 --output native/reports/realtime-gpu-int8.json
```

[容量计算JSON](realtime6144-capacity.json)保留公式、重复时间、理论/预算/完整分块三种容量及来源。原始结果：

- [realtime6144-cpu-m4max.json](realtime6144-cpu-m4max.json)
- [realtime6144-cpu-3955wx.json](realtime6144-cpu-3955wx.json)
- [realtime6144-real-cudnn-fp32.json](realtime6144-real-cudnn-fp32.json)
- [realtime6144-real-cudnn-int8.json](realtime6144-real-cudnn-int8.json)

另保留高斯输入的纯概率测量，PNSNFP32/INT8约0.1578/0.1555秒，ChinaFP32/INT8约0.1095/0.2143秒；这些不含NMS的数字不能直接当作完整近实时系统的容量。
