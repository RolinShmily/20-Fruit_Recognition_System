# -*- coding: utf-8 -*-
"""
将训练好的 PyTorch 模型导出为 ONNX 格式

用法: python export_onnx.py
导出后生成: models/best_resnet18_cbam.onnx
"""

import os
import sys
import torch
from model import load_model
from config import MODEL_DIR, MODEL_NAME


def export():
    pt_path = os.path.join(MODEL_DIR, MODEL_NAME)
    onnx_path = os.path.join(MODEL_DIR, MODEL_NAME.replace('.pt', '.onnx'))

    if not os.path.exists(pt_path):
        print(f"错误: 未找到 PyTorch 模型 {pt_path}")
        print("请先训练模型或确认模型文件存在")
        sys.exit(1)

    print("正在加载 PyTorch 模型...")
    model, device = load_model(pt_path)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224).to(device)

    print("正在导出 ONNX 格式...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        opset_version=13,
        dynamo=False,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"导出完成: {onnx_path} ({size_mb:.1f} MB)")
    print()
    print("下一步: 运行 gui_predictor.py 时会自动使用 ONNX 模型")


if __name__ == '__main__':
    export()
