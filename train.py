# -*- coding: utf-8 -*-
"""
模型训练 ResNet18 + CBAM 两阶段渐进式训练

Phase 1: 冻结全部骨干，只训练 CBAM + 分类头
Phase 2: 解冻 Layer3+4，差异化学习率微调
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from tqdm import tqdm

import config
from dataset import check_dataset, get_data_loaders
from model import ResNet18CBAM, get_device, count_parameters, save_model


def cutmix_data(images, labels, alpha=1.0):
    """CutMix 数据增强"""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0))
    shuffled_labels = labels[rand_index]

    # 生成随机裁剪区域
    W, H = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # 粘贴另一张图的对应区域
    images[:, :, x1:x2, y1:y2] = images[rand_index, :, x1:x2, y1:y2]

    # 按面积比调整 lambda
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return images, shuffled_labels, lam


def set_freeze_phase(model, phase):
    """设置冻结阶段Phase 1和Phase 2"""
    for name, param in model.named_parameters():
        if 'fc' not in name and 'cbam' not in name:
            param.requires_grad = (phase == 2 and
                                   ('layer3' in name or 'layer4' in name))
        else:
            param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc='训练', ncols=80):
        images, labels = images.to(device), labels.to(device)

        if config.USE_CUTMIX and np.random.random() < config.CUTMIX_PROB:
            images, shuffled_labels, lam = cutmix_data(
                images, labels, alpha=config.CUTMIX_ALPHA)
            outputs = model(images)
            loss = lam * criterion(outputs, labels) + \
                   (1 - lam) * criterion(outputs, shuffled_labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """在验证集/测试集上评估"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='评估', ncols=80):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def plot_training_history(history, save_dir):
    """绘制训练曲线（准确率 + 损失）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_acc']) + 1)

    ax1.plot(epochs, history['train_acc'], label='训练准确率', marker='.')
    ax1.plot(epochs, history['val_acc'], label='验证准确率', marker='.')

    phase1_len = config.PHASE1_EPOCHS
    if phase1_len < len(epochs):
        ax1.axvline(x=phase1_len, color='red', linestyle='--', alpha=0.5, label='Phase 切换')

    ax1.set_title('准确率 (ResNet18 + CBAM)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率 (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_loss'], label='训练损失', marker='.')
    ax2.plot(epochs, history['val_loss'], label='验证损失', marker='.')
    if phase1_len < len(epochs):
        ax2.axvline(x=phase1_len, color='red', linestyle='--', alpha=0.5, label='Phase 切换')

    ax2.set_title('损失 (ResNet18 + CBAM)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")
    plt.close()


def train():
    """两阶段渐进式训练"""
    print("=" * 60)
    print(f"水果识别模型训练 (ResNet18 + CBAM, v3)")
    print(f"类别数: {config.NUM_CLASSES} | CutMix: {config.USE_CUTMIX}")
    print(f"图像尺寸: {config.IMAGE_SIZE} | CBAM: {config.USE_CBAM}")
    print("=" * 60)

    if not check_dataset():
        return

    train_loader, val_loader, test_loader, class_weights = get_data_loaders()

    device = get_device()
    model = ResNet18CBAM().to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"使用加权交叉熵损失 (权重已移至 {device})")
    else:
        criterion = nn.CrossEntropyLoss()

    # ==================== Phase 1: 冻结骨干，训练 CBAM + 分类头 ====================
    print("\n" + "=" * 60)
    print("Phase 1: 冻结骨干，训练 CBAM + 分类头")
    print("=" * 60)

    set_freeze_phase(model, phase=1)
    count_parameters(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE1_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE1_EPOCHS)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(config.PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[P1] Epoch {epoch+1}/{config.PHASE1_EPOCHS} - "
              f"训练: {train_loss:.4f}/{train_acc:.2f}% | "
              f"验证: {val_loss:.4f}/{val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, epoch=epoch, phase=1)
            print(f"  -> Phase 1 最佳模型 ({val_acc:.2f}%)")

    # ==================== Phase 2: 解冻 Layer3+4，差异化学习率 ====================
    print("\n" + "=" * 60)
    print("Phase 2: 解冻 Layer3+4，差异化学习率微调")
    print("=" * 60)

    set_freeze_phase(model, phase=2)
    count_parameters(model)

    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': config.PHASE2_BACKBONE_LR},
        {'params': classifier_params, 'lr': config.PHASE2_CLASSIFIER_LR},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE2_EPOCHS)

    print(f"差异化学习率: 骨干={config.PHASE2_BACKBONE_LR}, 分类头={config.PHASE2_CLASSIFIER_LR}")

    patience_counter = 0

    for epoch in range(config.PHASE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[P2] Epoch {epoch+1}/{config.PHASE2_EPOCHS} - "
              f"训练: {train_loss:.4f}/{train_acc:.2f}% | "
              f"验证: {val_loss:.4f}/{val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, optimizer, epoch=epoch, phase=2)
            print(f"  -> 全局最佳模型 ({val_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"早停触发！(连续 {config.EARLY_STOP_PATIENCE} 轮未提升)")
            break

    training_time = time.time() - start_time
    print(f"\n训练完成！耗时: {training_time/60:.1f} 分钟, 最佳验证准确率: {best_val_acc:.2f}%")

    plot_training_history(history, config.RESULT_DIR)

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n测试集准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    train()
