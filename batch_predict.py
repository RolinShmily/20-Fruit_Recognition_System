# -*- coding: utf-8 -*-
"""
批量预测脚本 - v3（含真实标签对比）

对目标文件夹内所有图像进行预测，自动对比真实标签，生成统计图和详细结果报告。
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 强制设置中文字体 - 使用常见Windows字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

import torch
from PIL import Image
from collections import Counter
import re

import config
from model import load_model
from dataset import get_test_transform


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def load_ground_truth(define_path):
    """从 Define.md 加载真实标签"""
    ground_truth = {}
    if not os.path.exists(define_path):
        print(f"警告: 未找到 {define_path}")
        return ground_truth

    with open(define_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('|-') or line.startswith('| 图片文件名'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            filename = parts[1]
            ground_truth[filename] = {
                'cn_name': parts[2],
                'en_name': parts[3]
            }

    print(f"已加载 {len(ground_truth)} 条真实标签")
    return ground_truth


def predict_single(img_path, model, device, transform):
    """预测单张图片"""
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:5]
    top_names_en = [config.FRUIT_CLASSES[i] for i in top_indices]
    top_names_cn = [config.FRUIT_NAMES_CN.get(name, name) for name in top_names_en]
    top5 = [(top_names_cn[i], top_names_en[i], probs[top_indices[i]]) for i in range(len(top_indices))]

    return top_names_cn[0], top_names_en[0], probs[top_indices[0]], top5


def batch_predict(image_dir, model_path=None, output_dir=None):
    """批量预测"""
    define_path = os.path.join(image_dir, 'Define.md')
    ground_truth = load_ground_truth(define_path)

    print("\n正在加载模型...")
    model, device = load_model(model_path)
    transform = get_test_transform()
    print(f"模型加载完成，设备: {device}\n")

    # 收集图像文件
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)],
        key=lambda x: int(re.search(r'\d+', os.path.splitext(x)[0]).group())
                if re.search(r'\d+', os.path.splitext(x)[0]) else x
    )

    if not image_files:
        print(f"未找到图像文件: {image_dir}")
        return []

    print(f"共发现 {len(image_files)} 张图像，开始批量预测...\n")

    # 逐张预测
    results = []
    for i, filename in enumerate(image_files):
        img_path = os.path.join(image_dir, filename)
        gt = ground_truth.get(filename, {})
        gt_cn = gt.get('cn_name', None)

        try:
            cn_name, en_name, confidence, top5 = predict_single(img_path, model, device, transform)
            is_correct = (gt_cn is not None and cn_name == gt_cn)

            results.append({
                'filename': filename,
                'cn_name': cn_name,
                'en_name': en_name,
                'confidence': confidence,
                'top5': top5,
                'gt_cn': gt_cn,
                'is_correct': is_correct,
            })

            status = "✓" if is_correct else ("✗" if gt_cn else "?")
            bar_len = 40
            progress = (i + 1) / len(image_files)
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {i+1}/{len(image_files)} "
                  f"{filename} → {cn_name} ({confidence*100:.1f}%) {status}", end="")
        except Exception as e:
            print(f"\n  跳过 {filename}: {e}")
            results.append({
                'filename': filename,
                'cn_name': '错误',
                'en_name': 'Error',
                'confidence': 0.0,
                'top5': [],
                'gt_cn': gt_cn,
                'is_correct': False,
            })

    print("\n\n预测完成！正在生成统计图...\n")

    if output_dir is None:
        output_dir = config.RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 计算准确率
    valid_results = [r for r in results if r['gt_cn'] is not None]
    correct_count = sum(1 for r in valid_results if r['is_correct'])
    accuracy = correct_count / len(valid_results) if valid_results else 0

    print(f"\n{'='*60}")
    print(f"预测结果统计")
    print(f"{'='*60}")
    print(f"总图片数: {len(results)}")
    print(f"有真实标签: {len(valid_results)}")
    print(f"预测正确: {correct_count}")
    print(f"预测错误: {len(valid_results) - correct_count}")
    print(f"准确率: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")

    # 生成统计图
    _plot_statistics(results, output_dir, accuracy)

    # 保存文本报告
    _save_report(results, output_dir, accuracy)

    return results


def _plot_statistics(results, output_dir, accuracy):
    """生成统计可视化图（简化版：4个图表）"""
    valid = [r for r in results if r['confidence'] > 0]
    if not valid:
        print("没有有效的预测结果，跳过统计图生成。")
        return

    with_gt = [r for r in valid if r['gt_cn'] is not None]
    correct = [r for r in with_gt if r['is_correct']]
    wrong = [r for r in with_gt if not r['is_correct']]
    confidences = [r['confidence'] for r in valid]
    cn_names = [r['cn_name'] for r in valid]

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'批量预测统计结果 - 准确率: {accuracy*100:.1f}%',
                 fontsize=16, fontweight='bold', y=0.98)

    # ==================== 图1: 预测成功率饼图 ====================
    ax = axes[0, 0]
    if with_gt:
        sizes = [len(correct), len(wrong)]
        labels = [f'正确 {len(correct)}', f'错误 {len(wrong)}']
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90,
                                             textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
    else:
        ax.text(0.5, 0.5, '无真实标签\n无法计算准确率',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.7))
        ax.axis('off')
    ax.set_title('预测成功率', fontsize=13, fontweight='bold')

    # ==================== 图2: 类别分布 ====================
    ax = axes[0, 1]
    class_counter = Counter(cn_names)
    classes = sorted(class_counter.keys(), key=lambda x: class_counter[x], reverse=True)
    counts = [class_counter[c] for c in classes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('数量', fontsize=11)
    ax.set_title(f'预测类别分布 (共 {len(valid)} 张)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', fontsize=9)

    # ==================== 图3: 置信率分布 ====================
    ax = axes[1, 0]
    bins = np.arange(0, 1.05, 0.05)

    if correct:
        conf_correct = [r['confidence'] for r in correct]
        ax.hist(conf_correct, bins=bins, color='#2ecc71', alpha=0.6,
                label=f'正确 ({len(correct)})', edgecolor='white', linewidth=0.5)
    if wrong:
        conf_wrong = [r['confidence'] for r in wrong]
        ax.hist(conf_wrong, bins=bins, color='#e74c3c', alpha=0.6,
                label=f'错误 ({len(wrong)})', edgecolor='white', linewidth=0.5)

    if confidences:
        mean_conf = np.mean(confidences)
        ax.axvline(mean_conf, color='#3498db', linestyle='--', linewidth=2,
                   label=f'均值 {mean_conf*100:.0f}%')

    ax.set_xlabel('置信率', fontsize=11)
    ax.set_ylabel('数量', fontsize=11)
    ax.set_title('置信率分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # ==================== 图4: 混淆对分析 ====================
    ax = axes[1, 1]

    if wrong:
        # 统计混淆对（真实类别 → 预测类别）
        confusions = {}
        for r in wrong:
            key = (r['gt_cn'], r['cn_name'])
            confusions[key] = confusions.get(key, 0) + 1

        # 按错误次数排序，取前10个
        top_conf = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:10]

        y_labels = [f"{gt} → {pred}" for gt, pred in [k for k, v in top_conf]]
        x_vals = [v for k, v in top_conf]

        # 绘制横向柱状图
        colors = ['#e74c3c'] * len(y_labels)
        bars = ax.barh(range(len(y_labels)), x_vals, color=colors,
                       edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('错误次数', fontsize=11)
        ax.set_title(f'混淆对分析 (共 {len(wrong)} 个错误)', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 在柱状图右侧标注数值
        for bar, val in zip(bars, x_vals):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, '✓ 完美预测！\n无混淆错误',
                ha='center', va='center', fontsize=14, color='#2ecc71',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8, pad=1))

    ax.set_title('混淆对分析', fontsize=13, fontweight='bold')

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, 'batch_prediction_stats.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"统计图已保存: {save_path}")
    plt.close()


def _save_report(results, output_dir, accuracy):
    """保存文本报告"""
    report_path = os.path.join(output_dir, 'batch_prediction_report.txt')
    valid = [r for r in results if r['confidence'] > 0]
    with_gt = [r for r in valid if r['gt_cn'] is not None]
    correct = [r for r in with_gt if r['is_correct']]
    wrong = [r for r in with_gt if not r['is_correct']]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  批量预测报告 - ResNet18 + CBAM 水果识别 (v3)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"总图像数: {len(results)}\n")
        f.write(f"有效预测: {len(valid)}\n")
        f.write(f"有真实标签: {len(with_gt)}\n")
        f.write(f"预测正确: {len(correct)}\n")
        f.write(f"预测错误: {len(wrong)}\n")
        f.write(f"准确率: {accuracy*100:.2f}%\n\n")

        if valid:
            confidences = [r['confidence'] for r in valid]
            f.write(f"平均置信率: {np.mean(confidences)*100:.1f}%\n")
            f.write(f"最低置信率: {np.min(confidences)*100:.1f}%\n")
            f.write(f"最高置信率: {np.max(confidences)*100:.1f}%\n\n")

        # 类别分布
        f.write("-" * 70 + "\n")
        f.write("  预测类别分布\n")
        f.write("-" * 70 + "\n")
        class_counter = Counter(r['cn_name'] for r in valid)
        for name, count in class_counter.most_common():
            f.write(f"  {name}: {count} 张 ({count/len(valid)*100:.1f}%)\n")

        # 各类别准确率
        if with_gt:
            f.write("\n" + "-" * 70 + "\n")
            f.write("  各类别预测准确率\n")
            f.write("-" * 70 + "\n")
            class_acc = {}
            for r in with_gt:
                gt = r['gt_cn']
                if gt not in class_acc:
                    class_acc[gt] = {'correct': 0, 'total': 0}
                class_acc[gt]['total'] += 1
                if r['is_correct']:
                    class_acc[gt]['correct'] += 1

            for gt in sorted(class_acc.keys(), key=lambda x: class_acc[x]['total'], reverse=True):
                acc = class_acc[gt]['correct'] / class_acc[gt]['total']
                f.write(f"  {gt}: {class_acc[gt]['correct']}/{class_acc[gt]['total']} "
                       f"({acc*100:.1f}%)\n")

        # 错误案例分析
        if wrong:
            f.write("\n" + "-" * 70 + "\n")
            f.write("  错误案例分析\n")
            f.write("-" * 70 + "\n")
            for r in wrong:
                f.write(f"  [{r['filename']}] 真实:{r['gt_cn']} → 预测:{r['cn_name']} "
                       f"({r['confidence']*100:.1f}%)\n")

        # 详细结果
        f.write("\n" + "=" * 70 + "\n")
        f.write("  详细预测结果\n")
        f.write("=" * 70 + "\n\n")

        for i, r in enumerate(results):
            status = "✓ 正确" if r['is_correct'] else ("✗ 错误" if r['gt_cn'] else "? 未知")
            f.write(f"[{i+1:3d}] {r['filename']} - {status}\n")
            f.write(f"      真实: {r['gt_cn'] or '未知'} ({r['gt_en'] or 'N/A'})\n")
            f.write(f"      预测: {r['cn_name']} ({r['en_name']})  "
                   f"置信率: {r['confidence']*100:.1f}%\n")
            if r['top5']:
                f.write("      Top-5: ")
                top5_str = " | ".join(
                    f"{cn}({en}) {prob*100:.1f}%"
                    for cn, en, prob in r['top5']
                )
                f.write(top5_str + "\n")
            f.write("\n")

    print(f"预测报告已保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量水果图片预测 (ResNet18+CBAM v3)')
    parser.add_argument('image_dir', help='图像文件夹路径')
    parser.add_argument('--model-path', default=None,
                        help='指定模型文件路径 (覆盖默认路径)')
    parser.add_argument('--output-dir', default=None,
                        help='结果输出目录 (默认: results/)')
    args = parser.parse_args()

    batch_predict(args.image_dir, model_path=args.model_path, output_dir=args.output_dir)
