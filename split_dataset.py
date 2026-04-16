# -*- coding: utf-8 -*-
"""
数据集拆分脚本 — 从 Training 中拆出 15% 验证集 + 5% 测试集
"""
import os
import random
import shutil

random.seed(42)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
VAL_DIR = os.path.join(BASE_DIR, 'Validation')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

total_train = 0
total_val = 0
total_test = 0

for fruit in sorted(os.listdir(TRAIN_DIR)):
    src = os.path.join(TRAIN_DIR, fruit)
    if not os.path.isdir(src):
        continue

    os.makedirs(os.path.join(VAL_DIR, fruit), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, fruit), exist_ok=True)

    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)

    n = len(imgs)
    n_test = max(1, round(n * 0.05))
    n_val = max(1, round(n * 0.15))

    test_imgs = imgs[:n_test]
    val_imgs = imgs[n_test:n_test + n_val]

    for f in val_imgs:
        shutil.move(os.path.join(src, f), os.path.join(VAL_DIR, fruit, f))
    for f in test_imgs:
        shutil.move(os.path.join(src, f), os.path.join(TEST_DIR, fruit, f))

    remaining = n - len(val_imgs) - len(test_imgs)
    total_train += remaining
    total_val += len(val_imgs)
    total_test += len(test_imgs)
    print(f'{fruit:15s}  total={n:5d}  train={remaining:5d}  val={len(val_imgs):4d}  test={len(test_imgs):4d}')

print(f'\n{"SUMMARY":15s}  train={total_train}  val={total_val}  test={total_test}  total={total_train+total_val+total_test}')
print('Done!')
