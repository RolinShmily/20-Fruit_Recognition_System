# -*- coding: utf-8 -*-
"""
单图预测脚本 - v3

模型直接输出 18 类概率，无需子类映射
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import torch
from PIL import Image

import config
from model import load_model
from dataset import get_test_transform


def predict(img_path, model_path=None):
    """预测单张图片的水果类别"""
    # 加载模型
    model, device = load_model(model_path)

    # 使用统一的测试变换（含亮度归一化）
    transform = get_test_transform()

    img = Image.open(img_path).convert('RGB')

    img_tensor = transform(img).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Top-K 预测
    top_indices = np.argsort(probs)[::-1][:5]
    top_names_en = [config.FRUIT_CLASSES[i] for i in top_indices]
    top_names_cn = [config.FRUIT_NAMES_CN.get(name, name) for name in top_names_en]

    # 打印结果
    print(f"\n预测结果: {img_path}")
    print("-" * 60)
    for i in range(len(top_indices)):
        bar = "█" * int(probs[top_indices[i]] * 30)
        cn_name = top_names_cn[i]
        en_name = top_names_en[i]
        prob_pct = probs[top_indices[i]] * 100
        print(f"  {i+1}. {cn_name:6s} ({en_name}) {prob_pct:6.2f}% {bar}")

    # 保存可视化：3 列布局（原图 → 预测标注 → 概率柱状图）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                         gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # 原图
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('原始图片', fontsize=12, fontweight='bold')

    # 预测结果
    ax2.imshow(img.resize(config.IMAGE_SIZE))
    ax2.axis('off')
    top_cn = top_names_cn[0]
    top_en = top_names_en[0]
    ax2.set_title(
        f'预测: {top_cn} ({top_en})\n'
        f'置信度: {probs[top_indices[0]]*100:.1f}%',
        fontsize=12, fontweight='bold',
    )

    # Top-5 概率柱状图
    labels = [f"{cn}\n({en})" for cn, en in zip(top_names_cn, top_names_en)]
    values = [probs[i] * 100 for i in top_indices]
    colors = ['#2ecc71'] + ['#95a5a6'] * (len(top_indices) - 1)
    ax3.barh(labels, values, color=colors)
    ax3.set_xlabel('概率 (%)')
    ax3.set_title('Top-5 预测概率')
    ax3.set_xlim(0, 100)
    ax3.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, 'prediction_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n预测结果图已保存: {save_path}")
    plt.show()

    return top_cn, probs[top_indices[0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='水果图片预测 (ResNet18+CBAM v3)')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--model-path', default=None,
                        help='指定模型文件路径 (覆盖默认路径)')
    args = parser.parse_args()

    predict(args.image, model_path=args.model_path)
