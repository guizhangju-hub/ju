# TENG 传感器数据 CNN 识别

## 项目概述

本项目构建了一个完整的 PyTorch CNN 框架，用于 TENG（摩擦纳米发电机）触觉传感器的一维信号预处理和模式识别。

## 项目特性

### 📊 数据预处理
- **信号处理**：Butterworth 滤波、Savitzky-Golay 平滑、异常值移除
- **特征提取**：统计特征、FFT 频谱分析
- **归一化**：标准化、MinMax、RobustScaler
- **数据增强**：高斯噪声、时间移位、幅度缩放、窗口扭曲、幅度扭曲

### 🧠 深度学习模型
- **基础 CNN**：多层一维卷积网络
- **ResNet-based CNN**：残差连接的深度网络
- 支持自定义架构参数

### 🔧 训练框架
- 完整的训练、验证、测试流程
- 早期停止（Early Stopping）
- 学习率自适应调整
- 梯度裁剪防止梯度爆炸
- 检查点保存和加载

### 📈 评估指标
- 准确率、精确率、召回率、F1 分数
- 混淆矩阵、ROC-AUC
- 每类指标统计

### 📊 可视化
- 信号时域显示
- FFT 频谱分析
- 处理前后对比
- 训练历史曲线
- 混淆矩阵热力图
- 性能指标柱状图

## 项目结构

```
ju/
├── config.py                 # 配置文件
├── requirements.txt          # 依赖包
├── README.md                # 项目说明
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
│
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── dataset.py          # 数据集类
│
├── models/
│   ├── __init__.py
│   └── cnn_model.py        # CNN 模型定义
│
├── preprocessing/
│   ├── signal_processing.py    # 信号处理
│   ├── normalization.py        # 数据归一化
│   └── augmentation.py         # 数据增强
│
└── utils/
    ├── metrics.py          # 评估指标
    └── visualization.py    # 可视化工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

在 `data/raw/` 目录下放入您的传感器数据。支持的格式：
- NumPy 数组 (.npy)
- Pickle 文件 (.pkl)
- CSV 文件

### 3. 配置参数

编辑 `config.py` 文件根据您的数据集调整参数：

```python
SIGNAL_LENGTH = 1024        # 信号长度
NUM_CLASSES = 10            # 分类数
BATCH_SIZE = 32             # 批大小
LEARNING_RATE = 0.001       # 学习率
NUM_EPOCHS = 100            # 训练轮次
```

### 4. 训练模型

```bash
python train.py
```

### 5. 评估模型

```bash
python evaluate.py
```

## 使用示例

### 信号预处理

```python
import numpy as np
from preprocessing.signal_processing import SignalProcessor
from preprocessing.normalization import DataNormalizer
from preprocessing.augmentation import SignalAugmentation

# 初始化处理器
processor = SignalProcessor(sampling_rate=1000)
normalizer = DataNormalizer(method='standard')

# 加载信号
signal = np.random.randn(1024)

# 去噪
filtered = processor.apply_butterworth_filter(signal, cutoff_freq=100)

# 移除异常值
cleaned = processor.remove_outliers(filtered, method='iqr')

# 归一化
normalized = normalizer.fit_transform(cleaned.reshape(-1, 1))

# 数据增强
augmented = SignalAugmentation.compose_augmentations(normalized)
```

### 模型训练

```python
from models.cnn_model import TENGCNN1D
from data.dataset import TENGDataset, get_data_loaders
from train import Trainer
import torch

# 创建模型
model = TENGCNN1D(
    input_channels=1,
    num_classes=10,
    filters=[32, 64, 128],
    kernel_sizes=[3, 3, 3]
)

# 创建数据加载器
train_loader, val_loader, test_loader = get_data_loaders(
    train_signals, train_labels,
    val_signals, val_labels,
    test_signals, test_labels,
    batch_size=32
)

# 训练
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device='cuda',
    lr=0.001
)

trainer.train(num_epochs=100, patience=15)
```

### 信号可视化

```python
from utils.visualization import SignalVisualizer, MetricsVisualizer

# 可视化原始信号
visualizer = SignalVisualizer()
visualizer.plot_signal(signal, sampling_rate=1000, title="原始信号")

# 对比处理前后
visualizer.plot_signal_before_after(original, processed)

# 绘制 FFT 谱
freq, mag = processor.compute_fft(signal)
visualizer.plot_fft(freq, mag)

# 绘制混淆矩阵
MetricsVisualizer.plot_confusion_matrix(y_true, y_pred)
```

## 配置说明

### 信号处理参数
- `SIGNAL_LENGTH`：每个信号样本的长度
- `SAMPLING_RATE`：采样率（Hz）
- `NOISE_METHOD`：去噪方法（'butterworth' 或 'savitzky_golay'）

### 模型参数
- `FILTERS`：每层卷积的过滤器数量
- `KERNEL_SIZES`：卷积核大小
- `POOL_SIZES`：池化窗口大小
- `DROPOUT_RATE`：Dropout 概率

### 训练参数
- `BATCH_SIZE`：批处理大小
- `LEARNING_RATE`：学习率
- `NUM_EPOCHS`：最大训练轮次
- `EARLY_STOPPING_PATIENCE`：早停耐心值

## 高级功能

### 自定义数据加载

```python
from data.dataset import TENGDataset

# 创建自定义数据集
dataset = TENGDataset(
    signals=your_signals,
    labels=your_labels,
    transform=your_augmentation_function
)
```

### 模型架构定制

```python
from models.cnn_model import ResNetCNN1D

# 使用 ResNet 架构
model = ResNetCNN1D(
    input_channels=1,
    num_classes=10,
    num_blocks=[2, 2, 2, 2],
    filters=[64, 128, 256, 512]
)
```

## 性能优化建议

1. **数据增强**：使用多种增强策略提高模型泛化能力
2. **超参数调整**：使用网格搜索或贝叶斯优化找到最优参数
3. **模型集成**：训练多个模型进行集成预测
4. **学习率调度**：采用余弦退火或其他高级调度策略
5. **正则化**：使用 Dropout、L1/L2 正则化防止过拟合

## 故障排除

### 显存不足
- 减小 `BATCH_SIZE`
- 减少 `FILTERS` 数量
- 使用 CPU 训练

### 模型过拟合
- 增加 `DROPOUT_RATE`
- 增加数据增强
- 使用更多的正则化
- 减小模型大小

### 训练缓慢
- 增加 `BATCH_SIZE`
- 减少模型复杂度
- 使用 GPU 加速
- 检查数据加载器的 `num_workers` 参数

## 常见问题

**Q: 如何使用自己的数据？**

A: 将您的数据放在 `data/raw/` 目录，然后在 `train.py` 中修改数据加载部分。

**Q: 如何调整模型复杂度？**

A: 修改 `config.py` 中的 `FILTERS`、`KERNEL_SIZES` 等参数。

**Q: 支持哪些 GPU？**

A: 支持所有 CUDA 兼容的 NVIDIA GPU。在 `config.py` 中设置 `DEVICE = 'cuda'`。

## 参考文献

- LeCun, Y., et al. (1989). "Backpropagation Applied to Handwritten Zip Code Recognition"
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"

## 许可证

MIT License

## 作者

Guizhang Ju (guizhangju-hub)

## 致谢

感谢所有贡献者和使用者的支持！
