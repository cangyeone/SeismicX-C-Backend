# SeismicX-C-Backend

SeismicX品牌的C11震相拾取后端，已适配PNSN v3及现有 `china.unet.pt` 中国数据PhaseNet风格UNet，提供FP32与8位模型、BN融合、内部float64去均值/归一化、精确最小堆NMS，以及CMake跨平台静态/共享编译。

两种输入策略：历史 `[3,N]` 任意长度，以及实时 `[B,3,L]` 独立批处理（默认L10240，也支持6144）。CPU无第三方推理依赖；cuDNN/MLX为可选后端。

- [中文使用与C/Python接口](native/README.zh.md)
- [两个模型在全部已测试设备上的完整速度矩阵](native/reports/DEVICE_MATRIX.md)：M4 Max CPU/MLX、3955WX CPU、RTX5090cuDNN；单条、实际B4096批次、完整历史/实时拾取和内存。
- [China UNet结构、8位数值与标签评估](native/reports/PHASENET.md)
- [PNSN cuDNN INT8与CPU评估](native/reports/INT8_CPU.md)
- [来源与权重说明](NOTICE.md)，[MIT许可证](LICENSE)

量化模式按实际算术标注：CPU W8A8；cuDNN China卷积W8A8，PNSN卷积W8A8/LSTM FP32；MLX Q8为W8A32。测速显示CPU INT8提速，而GPU/MLX的量化模式不保证更快。物理边缘板尚未测速。
