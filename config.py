# -*- coding: utf-8 -*-
"""
全局配置文件 - ResNet18 + CBAM 水果识别系统
"""

import os
import sys
import numpy as np

# ==================== GUI路径配置 ====================
# PyInstaller 打包后需要区分两个目录：
#   BASE_DIR      → exe 所在目录（用户可写）
#   _RESOURCE_DIR → _internal/ 目录（捆绑资源，如模型文件）
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    _RESOURCE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _RESOURCE_DIR = BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
VALIDATION_DIR = os.path.join(DATA_DIR, 'Validation')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = os.path.join(_RESOURCE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==================== 类别定义（20 种水果 来自 fruits-262 数据集）====================
FRUIT_CLASSES = [
    'Apple', 'Banana', 'Orange', 'Grape', 'Watermelon',
    'Strawberry', 'Pineapple', 'Mango', 'Lemon', 'Kiwi',
    'Jujube', 'Pear', 'Cherry', 'Coconut', 'Pomegranate',
    'Papaya', 'Avocado', 'Blueberry', 'Cantaloupe', 'Dragonfruit',
]

# 数据集目录名 → config 类别名（fruits-262 数据集目录名 → 类别名映射）
FRUIT_DIR_NAMES = {
    'apple': 'Apple',
    'banana': 'Banana',
    'orange': 'Orange',
    'grape': 'Grape',
    'watermelon': 'Watermelon',
    'strawberry': 'Strawberry',
    'pineapple': 'Pineapple',
    'mango': 'Mango',
    'lemon': 'Lemon',
    'kiwi': 'Kiwi',
    'jujube': 'Jujube',
    'pear': 'Pear',
    'cherry': 'Cherry',
    'coconut': 'Coconut',
    'pomegranate': 'Pomegranate',
    'papaya': 'Papaya',
    'avocado': 'Avocado',
    'blueberry': 'Blueberry',
    'cantaloupe': 'Cantaloupe',
    'dragonfruit': 'Dragonfruit',
}

FRUIT_NAMES_CN = {
    'Apple': '苹果', 'Banana': '香蕉', 'Orange': '橙子',
    'Grape': '葡萄', 'Watermelon': '西瓜', 'Strawberry': '草莓',
    'Pineapple': '菠萝', 'Mango': '芒果', 'Lemon': '柠檬',
    'Kiwi': '猕猴桃', 'Jujube': '枣', 'Pear': '梨',
    'Cherry': '樱桃', 'Coconut': '椰子', 'Pomegranate': '石榴',
    'Papaya': '木瓜', 'Avocado': '牛油果', 'Blueberry': '蓝莓',
    'Cantaloupe': '哈密瓜', 'Dragonfruit': '火龙果',
}

NUM_CLASSES = len(FRUIT_CLASSES)  

# ==================== 基础参数 ====================
IMAGE_SIZE = (224, 224)       # ResNet18 标准输入尺寸
BATCH_SIZE = 32
USE_GPU = True

# ==================== CBAM 注意力机制 ====================
USE_CBAM = True
CBAM_REDUCTION = 16

# ==================== 数据增强 ====================
GAUSSIAN_BLUR_KERNEL = 3
RANDOM_ERASING_P = 0.3
RANDOM_PERSPECTIVE_DIST = 0.2
COLOR_JITTER_BRIGHTNESS = 0.6     # 亮度增强范围
COLOR_JITTER_CSA = 0.4            # 对比度/饱和度增强范围
COLOR_JITTER_HUE = 0.1            # 色调增强范围

# ==================== CutMix 增强 ====================
USE_CUTMIX = True            # 是否启用 CutMix
CUTMIX_PROB = 0.5            # CutMix 触发概率
CUTMIX_ALPHA = 1.0           # Beta 分布参数（1.0 = 均匀分布）

# ==================== 亮度域适配 ====================
USE_BRIGHTNESS_NORMALIZE = True    # 预测时自动归一化亮度
TARGET_BRIGHTNESS = 0.45           # 归一化目标亮度

# ==================== 类别不均衡处理 ====================
USE_CLASS_WEIGHTS = True            # 启用类别加权损失
USE_WEIGHTED_SAMPLER = True         # 启用加权随机采样（过采样少数类）

# ==================== 两阶段训练 ====================
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_BACKBONE_LR = 1e-4
PHASE2_CLASSIFIER_LR = 5e-4
WEIGHT_DECAY = 1e-4
COSINE_T_MAX = 20
EARLY_STOP_PATIENCE = 7

# ==================== 模型保存 ====================
MODEL_NAME = "best_resnet18_cbam.pt"


def compute_class_weights(class_counts):
    """计算类别权重（反频率加权）"""
    counts = np.array(class_counts, dtype=np.float64)
    total = counts.sum()
    n_classes = len(counts)
    weights = total / (n_classes * counts)
    weights = weights / weights.mean()
    import torch
    return torch.FloatTensor(weights)
