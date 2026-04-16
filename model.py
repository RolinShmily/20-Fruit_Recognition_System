# -*- coding: utf-8 -*-
"""
模型定义 - ResNet18 + CBAM 注意力机制
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import (
    NUM_CLASSES, USE_GPU, USE_CBAM, CBAM_REDUCTION,
    MODEL_NAME, MODEL_DIR,
)


class ChannelAttention(nn.Module):
    """通道注意力模块（双池化分支）"""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(in_planes // ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, in_planes, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ResNet18CBAM(nn.Module):
    """基于 ResNet18 + CBAM 的水果分类模型"""

    def __init__(self, num_classes=NUM_CLASSES, use_cbam=USE_CBAM,
                 cbam_reduction=CBAM_REDUCTION):
        super().__init__()
        self.use_cbam = use_cbam
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 逐层拆解 ResNet18
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 64 channels
        self.layer2 = backbone.layer2   # 128 channels
        self.layer3 = backbone.layer3   # 256 channels
        self.layer4 = backbone.layer4   # 512 channels
        self.avgpool = backbone.avgpool

        # CBAM 注意力（注入到 Layer2/3/4 后）
        if use_cbam:
            self.cbam2 = CBAM(128, cbam_reduction)
            self.cbam3 = CBAM(256, cbam_reduction)
            self.cbam4 = CBAM(512, cbam_reduction)

        # 自定义分类头
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam2(x)
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_device():
    """获取计算设备"""
    if USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model):
    """统计模型参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} | 总参数: {total:,}")
    return trainable, total


def save_model(model, optimizer, path=None, **extra):
    """保存模型检查点"""
    if path is None:
        path = MODEL_DIR + '/' + MODEL_NAME
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': NUM_CLASSES,
        'use_cbam': model.use_cbam,
        'cbam_reduction': CBAM_REDUCTION,
        **extra,
    }, path)
    print(f"模型已保存: {path}")


def load_model(path=None):
    """加载模型检查点"""
    if path is None:
        path = MODEL_DIR + '/' + MODEL_NAME
    device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = ResNet18CBAM(
        num_classes=checkpoint['num_classes'],
        use_cbam=checkpoint.get('use_cbam', True),
        cbam_reduction=checkpoint.get('cbam_reduction', 16),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device
