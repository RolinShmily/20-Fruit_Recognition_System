<div align="center">

# 🍎 水果识别系统

**基于 ResNet18 + CBAM 注意力机制的 20 种水果智能识别**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**测试集准确率：91.78%** · GUI 桌面应用 · ONNX 推理 · PyInstaller 打包

[English](README_EN.md) · 中文文档

</div>

---

## 项目简介

基于深度卷积神经网络的水果自动识别系统，支持自然背景下的 **20 种水果**分类。针对真实场景中背景干扰、域差异、类别不均衡等问题，采用以下技术方案：

| 技术点 | 方案 |
|--------|------|
| 骨干网络 | ResNet18（ImageNet 预训练） |
| 注意力机制 | CBAM 通道 + 空间双注意力（注入 Layer2/3/4） |
| 数据增强 | CutMix 混合增强 + 多维度变换管道 |
| 类别不均衡 | 加权随机采样 + 加权交叉熵损失 |
| 训练策略 | 两阶段渐进式训练 + 差异化学习率 |
| 亮度适配 | 预测时亮度归一化 |
| 部署方案 | PySide6 GUI + ONNX Runtime（打包仅 ~150MB） |

---

## 主要特性

- **CBAM 注意力机制** — 通道 + 空间双维度引导模型聚焦水果区域，抑制背景干扰
- **CutMix 数据增强** — 混合样本增强，强迫模型从局部区域识别水果，提升泛化能力
- **两阶段渐进式训练** — Phase 1 冻结骨干训练注意力模块，Phase 2 解冻深层精细微调
- **类别不均衡处理** — 加权采样 + 加权损失双重机制，保证少数类识别效果
- **GUI 桌面应用** — 支持单图预测和批量预测，含 Top-5 置信度展示
- **批量预测分析** — 自动生成 9 种统计图表，含混淆矩阵、错误分析、低置信预警
- **轻量部署** — ONNX Runtime 推理，无需完整 PyTorch 环境

---

## 模型架构

```
输入图像 (224×224)
        │
        ▼
┌──────────────────────────────────────────┐
│        ResNet18 骨干网络 (ImageNet)        │
│    ├── Conv1 + BN + ReLU + MaxPool        │
│    ├── Layer1 (64ch)        [冻结]        │
│    ├── Layer2 (128ch)       [冻结]        │
│    │       └──→ CBAM(128)                 │
│    ├── Layer3 (256ch)     [解冻微调]       │
│    │       └──→ CBAM(256)                 │
│    ├── Layer4 (512ch)     [解冻微调]       │
│    │       └──→ CBAM(512)                 │
│    └── AdaptiveAvgPool2d(1,1)             │
└──────────────┬───────────────────────────┘
               │ 512 维特征向量
               ▼
┌──────────────────────────────────────────┐
│           自定义双层分类头                  │
│    Dropout(0.5)                           │
│    Linear(512 → 256) + ReLU + BN          │
│    Dropout(0.3)                           │
│    Linear(256 → 20)                       │
└──────────────┬───────────────────────────┘
               │
          20 类别 logits
```

**模型参数量：** ~1139 万（Phase 1 仅训练 ~21 万参数）

---

## 支持的水果类别（20 种）

| # | 水果 | # | 水果 |
|:---:|------|:---:|------|
| 1 | 🍎 Apple 苹果 | 11 | Jujube 枣 |
| 2 | 🍌 Banana 香蕉 | 12 | 🍐 Pear 梨 |
| 3 | 🍊 Orange 橙子 | 13 | 🍒 Cherry 樱桃 |
| 4 | 🍇 Grape 葡萄 | 14 | 🥥 Coconut 椰子 |
| 5 | 🍉 Watermelon 西瓜 | 15 | Pomegranate 石榴 |
| 6 | 🍓 Strawberry 草莓 | 16 | Papaya 木瓜 |
| 7 | 🍍 Pineapple 菠萝 | 17 | 🥑 Avocado 牛油果 |
| 8 | 🥭 Mango 芒果 | 18 | Blueberry 蓝莓 |
| 9 | 🍋 Lemon 柠檬 | 19 | Cantaloupe 哈密瓜 |
| 10 | 🥝 Kiwi 猕猴桃 | 20 | Dragonfruit 火龙果 |

---

## 快速开始

### 环境要求

- Python 3.8+
- 推荐使用 NVIDIA GPU（支持 CPU 模式）

### 安装

```bash
# 克隆仓库
git clone https://github.com/RolinShmily/fruit-recognition.git
cd fruit-recognition

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 从 [fruits-262 数据集](https://github.com/fruits-262/fruits-262-original-size) 下载原始数据
2. 将所选的 20 种水果类别放入 `data/` 目录
3. 运行数据集拆分脚本：

```bash
python split_dataset.py
```

拆分后的目录结构：

```
data/
├── Training/       # 训练集 (~16,692 张)
├── Validation/     # 验证集 (~3,129 张)
└── Test/           # 测试集 (~1,042 张)
```

### 模型训练

```bash
python train.py
```

两阶段训练策略：
- **Phase 1**（10 epochs）：冻结骨干网络，训练 CBAM + 分类头（lr=1e-3）
- **Phase 2**（15 epochs）：解冻 Layer3+4，差异化学习率微调（骨干=1e-4, 分类头=5e-4）

最佳模型自动保存至 `models/best_resnet18_cbam.pt`。

### 模型评估

```bash
python evaluate.py
```

输出：分类报告（Precision / Recall / F1）、混淆矩阵热力图、各类别准确率。

### 预测推理

```bash
# 单图预测
python predict.py images/apple.jpg

# 批量预测（含真实标签对比）
python batch_predict.py images/
```

批量预测会自动生成：
- **9 种统计图表** — 成功率饼图、类别分布、置信率分布、混淆矩阵等
- **详细文本报告** — 各类别准确率、错误案例分析、低置信预警
- **终端实时进度** — 每张图片的预测结果和置信度

### GUI 图形界面

```bash
# 先导出 ONNX 模型（GUI 推理必需）
python export_onnx.py

# 启动 GUI
python gui_predictor.py
```

GUI 功能：
- 单图预测：选择/拖拽图片，查看 Top-5 预测结果和置信度
- 批量预测：选择图片目录，浏览式查看每张图片结果
- 保存结果：导出原图、标注图、概率图、结果文本

---

## 项目结构

```
├── config.py              # 全局配置（类别、超参数、路径）
├── model.py               # ResNet18 + CBAM 模型定义
├── dataset.py             # 数据集加载器（含 CutMix 增强）
├── train.py               # 两阶段训练脚本
├── evaluate.py            # 模型评估（指标 + 混淆矩阵）
├── predict.py             # 单图预测
├── batch_predict.py       # 批量预测（含分析报告）
├── gui_predictor.py       # PySide6 GUI 应用
├── export_onnx.py         # 导出 ONNX 模型
├── split_dataset.py       # 数据集拆分脚本
├── requirements.txt       # Python 依赖
├── data/                  # 数据集目录（不包含在仓库中）
├── models/                # 模型保存目录
├── results/               # 结果输出目录
└── images/                # 示例测试图片
```

---

## 配置参数

关键超参数定义在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_CBAM` | `True` | 启用 CBAM 注意力机制 |
| `USE_CUTMIX` | `True` | 启用 CutMix 混合增强 |
| `CUTMIX_PROB` | `0.5` | CutMix 触发概率 |
| `PHASE1_EPOCHS` | `10` | Phase 1 训练轮数 |
| `PHASE2_EPOCHS` | `15` | Phase 2 训练轮数 |
| `BATCH_SIZE` | `32` | 训练批大小 |
| `EARLY_STOP_PATIENCE` | `7` | 早停耐心值 |
| `USE_CLASS_WEIGHTS` | `True` | 类别加权交叉熵损失 |
| `USE_WEIGHTED_SAMPLER` | `True` | 加权随机采样 |

---

## 打包发布

将 GUI 打包为独立 exe，无需安装 Python 即可运行：

```bash
# 安装打包工具
pip install pyinstaller onnxruntime

# 确保 ONNX 模型已导出
python export_onnx.py

# 打包
pyinstaller --noconfirm --onedir --windowed ^
    --name "水果识别系统" ^
    --add-data "models\best_resnet18_cbam.onnx;models" ^
    --hidden-import PySide6.QtWidgets ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtGui ^
    --hidden-import onnxruntime ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    gui_predictor.py
```

> **macOS/Linux 用户：** 将 `^` 换行符改为 `\`，`--add-data` 分隔符 `;` 改为 `:`。

| 方案 | 打包体积 | 说明 |
|------|---------|------|
| PyTorch (GPU) | ~2 GB | 包含完整 CUDA 库 |
| PyTorch (CPU) | ~600 MB | 仅 CPU 推理 |
| **ONNX Runtime** | **~150 MB** | 本项目方案，无需 PyTorch |

---

## 技术亮点总结

| 创新点 | 传统方案 | 本项目方案 |
|--------|----------|------------|
| 数据集 | 白背景数据集，域差异严重 | 自然背景数据集，从根源消除域差异 |
| 注意力 | 纯 CNN，无显式注意力引导 | CBAM 通道+空间双注意力，聚焦水果区域 |
| 数据增强 | 基础几何变换 | CutMix + 多维度增强管道 |
| 类别不均衡 | 无处理 | 加权采样 + 加权损失双重机制 |
| 训练策略 | 单阶段训练 | 两阶段渐进式训练，差异化学习率 |

---

## 参考资料

| 论文 / 资源 | 作者 / 年份 | 核心贡献 |
|-------------|------------|----------|
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He et al., 2016 | 提出残差网络 ResNet |
| [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) | Woo et al., 2018 | 提出通道+空间注意力机制 |
| [CutMix: Regularization Strategy](https://arxiv.org/abs/1905.04899) | Yun et al., 2019 | 提出图像混合增强方法 |
| [fruits-262 Dataset](https://github.com/fruits-262/fruits-262-original-size) | — | 自然背景水果图像数据集 |

---

## 开源协议

本项目基于 [MIT 协议](LICENSE) 开源。

---

<div align="center">

**Built with PyTorch · ResNet18 · CBAM · CutMix**

</div>
