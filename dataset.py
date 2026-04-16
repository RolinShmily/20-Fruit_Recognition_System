# -*- coding: utf-8 -*-
"""
数据集处理模块
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from config import (
    TRAIN_DIR, VALIDATION_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE,
    FRUIT_CLASSES, FRUIT_DIR_NAMES,
    GAUSSIAN_BLUR_KERNEL, RANDOM_ERASING_P, RANDOM_PERSPECTIVE_DIST,
    COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_CSA, COLOR_JITTER_HUE,
    USE_CLASS_WEIGHTS, USE_WEIGHTED_SAMPLER,
    USE_BRIGHTNESS_NORMALIZE, TARGET_BRIGHTNESS,
    compute_class_weights,
)


# ==================== 数据变换 ====================

class BrightnessNormalize:
    """将图片亮度归一化到目标值"""
    def __init__(self, target_brightness=0.45):
        self.target = target_brightness

    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        brightness = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        current = brightness.mean()
        if current < 0.01:
            return img
        factor = max(0.5, min(2.0, self.target / current))
        arr = np.clip(arr * factor * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def get_train_transform():
    """训练集随机变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=RANDOM_PERSPECTIVE_DIST, p=0.5),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CSA,
            saturation=COLOR_JITTER_CSA,
            hue=COLOR_JITTER_HUE,
        ),
        transforms.GaussianBlur(kernel_size=GAUSSIAN_BLUR_KERNEL),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=RANDOM_ERASING_P),
    ])


def get_test_transform():
    """测试/验证变换"""
    transform_list = [
        transforms.Resize(IMAGE_SIZE),
    ]
    if USE_BRIGHTNESS_NORMALIZE:
        transform_list.append(BrightnessNormalize(target_brightness=TARGET_BRIGHTNESS))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


# ==================== 数据集类 ====================

class FruitDataset(Dataset):
    """水果图像数据集"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(FRUIT_CLASSES)}
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """遍历数据集目录，使用 FRUIT_DIR_NAMES 做目录名→类别名映射"""
        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            class_name = FRUIT_DIR_NAMES.get(folder_name)
            if class_name is None or class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder_path, img_name), class_idx))

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self):
        counts = np.zeros(len(FRUIT_CLASSES), dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label


# ==================== 数据加载器 ====================

def get_data_loaders(num_workers=4):
    """获取训练/验证/测试数据加载器（支持类别加权）"""
    print(f"正在准备数据加载器 (图像尺寸: {IMAGE_SIZE})...")

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # 三个独立数据集
    train_set = FruitDataset(TRAIN_DIR, transform=train_transform)
    val_set = FruitDataset(VALIDATION_DIR, transform=test_transform)
    test_set = FruitDataset(TEST_DIR, transform=test_transform)

    # ==================== 类别不均衡处理 ====================
    class_weights = None
    sampler = None

    class_counts = train_set.get_class_counts()
    n_classes = len(class_counts)

    # 打印类别分布摘要
    active_counts = class_counts[class_counts > 0]
    print(f"\n类别分布统计:")
    print(f"  总类别数: {n_classes} | 有样本的类别: {len(active_counts)}")
    print(f"  最多: {active_counts.max()} 张 | 最少: {active_counts.min()} 张 | "
          f"不均衡比: {active_counts.max()/active_counts.min():.1f}:1")

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(class_counts)
        print(f"  类别加权损失: 已启用 (权重范围 [{class_weights.min():.2f}, {class_weights.max():.2f}])")

    if USE_WEIGHTED_SAMPLER:
        sample_weights = []
        for _, label in train_set.samples:
            sample_weights.append(1.0 / class_counts[label])
        sample_weights = torch.FloatTensor(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_set),
            replacement=True,
        )
        print(f"  加权随机采样: 已启用 (每 epoch 等概率采样各类)")

    # ==================== 构建 DataLoader ====================
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"\n训练集: {len(train_set)} | 验证集: {len(val_set)} | 测试集: {len(test_set)}")
    print(f"类别数: {n_classes}")

    return train_loader, val_loader, test_loader, class_weights


def check_dataset():
    """检查三个数据集目录是否存在"""
    train_ok = os.path.exists(TRAIN_DIR)
    val_ok = os.path.exists(VALIDATION_DIR)
    test_ok = os.path.exists(TEST_DIR)

    if train_ok and val_ok and test_ok:
        print("数据集已就绪（Training / Validation / Test）")
        return True

    missing = []
    if not train_ok:
        missing.append(f"Training: {TRAIN_DIR}")
    if not val_ok:
        missing.append(f"Validation: {VALIDATION_DIR}")
    if not test_ok:
        missing.append(f"Test: {TEST_DIR}")

    print("数据集不完整，缺少以下目录：")
    for m in missing:
        print(f"  - {m}")
    return False
