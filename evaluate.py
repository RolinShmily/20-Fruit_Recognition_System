# -*- coding: utf-8 -*-
"""
模型评估
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

import config
from dataset import check_dataset, get_data_loaders
from model import load_model


def evaluate():
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}，请先训练模型！")
        return

    print("=" * 60)
    print("模型评估 (ResNet18 + CBAM, v3)")
    print(f"类别数: {config.NUM_CLASSES}")
    print("=" * 60)

    model, device = load_model(model_path)

    if not check_dataset():
        return

    _, _, test_loader, _ = get_data_loaders()

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='预测', ncols=80):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)

    # ==================== 准确率 ====================
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # ==================== 分类报告 ====================
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    class_names_cn = [config.FRUIT_NAMES_CN.get(config.FRUIT_CLASSES[i], config.FRUIT_CLASSES[i])
                      for i in unique_labels]

    report = classification_report(
        y_true, y_pred, labels=unique_labels,
        target_names=class_names_cn, digits=4, zero_division=0,
    )
    print("\n分类报告:")
    print(report)

    # ==================== 混淆矩阵 ====================
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names_cn, yticklabels=class_names_cn)
    plt.title('混淆矩阵 - 水果分类结果 (ResNet18+CBAM)', fontsize=14, fontweight='bold')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    cm_path = os.path.join(config.RESULT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存: {cm_path}")
    plt.close()

    # ==================== 各类别准确率摘要 ====================
    print("\n各类别准确率:")
    for i, name in enumerate(class_names_cn):
        if i < len(cm) and cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
        else:
            acc = 0
        print(f"  {name:6s}: {acc:.2%}")

    # ==================== 保存完整报告 ====================
    report_path = os.path.join(config.RESULT_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"模型: ResNet18 + CBAM (v3)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"类别数: {config.NUM_CLASSES}\n\n")
        f.write(f"{'='*60}\n")
        f.write("分类报告:\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
    print(f"\n完整分类报告已保存: {report_path}")

    print(f"\n评估完成！准确率: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate()
