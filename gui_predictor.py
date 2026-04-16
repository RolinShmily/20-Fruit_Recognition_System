# -*- coding: utf-8 -*-
"""
水果识别 GUI 程序 - 基于 PySide6

功能：
- 单图预测：选择图片进行预测，显示 Top-5 预测结果，支持保存结果
- 批量预测：批量预测所有图片，浏览式界面，支持标记正确性，生成统计报告
"""

import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog,
    QFrame, QGroupBox, QGridLayout, QMessageBox,
    QSplitter, QTextEdit, QStatusBar, QDialog, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

import config


# ==================== 通用样式 ====================
GLOBAL_STYLE = """
QMainWindow, QDialog, QMessageBox {
    background-color: #ffffff;
}
QTabWidget::pane { border: 1px solid #bbb; background: white; }
QTabBar::tab {
    background: #dcdcdc; color: #444; padding: 8px 20px;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    font-size: 13px;
}
QTabBar::tab:selected { background: white; color: #111; font-weight: bold; }
QPushButton {
    background-color: #3a7bd5; color: #ffffff; border: none;
    padding: 7px 16px; border-radius: 3px; font-size: 13px;
    font-weight: bold;
}
QPushButton:hover { background-color: #2e6bc4; }
QPushButton:pressed { background-color: #2558a3; }
QPushButton[type="nav"]    { background-color: #5b4a8a; color: #ffffff; }
QPushButton[type="nav"]:hover { background-color: #4e3f7a; }
QPushButton[type="ok"]     { background-color: #1a7a3a; color: #ffffff; }
QPushButton[type="ok"]:hover  { background-color: #15662f; }
QPushButton[type="err"]    { background-color: #c0392b; color: #ffffff; }
QPushButton[type="err"]:hover { background-color: #a33025; }
QPushButton[type="save"]   { background-color: #0d7a5f; color: #ffffff; }
QPushButton[type="save"]:hover { background-color: #0a6650; }
QPushButton[type="warn"]   { background-color: #d4760a; color: #ffffff; }
QPushButton[type="warn"]:hover { background-color: #b56508; }
QPushButton:disabled { background-color: #b0b0b0; color: #666666; }
QGroupBox {
    font-size: 13px; font-weight: bold; color: #1a1a1a;
    border: 1px solid #bbb; border-radius: 4px;
    margin-top: 8px; padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #1a1a1a;
}
QLabel { color: #1a1a1a; }
QStatusBar { color: #333; font-size: 12px; }
QStatusBar QLabel { color: #333; }
QCheckBox { color: #1a1a1a; spacing: 6px; }
QTextBrowser, QTextEdit {
    border: 1px solid #bbb; background: white; color: #1a1a1a;
    font-size: 12px;
}
"""


# ==================== ONNX 推理辅助 ====================
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_BRIGHTNESS = 0.45


def preprocess_image(img):
    """将 PIL Image 预处理为 ONNX 输入 (1, 3, 224, 224) float32

    等价于: Resize(224) → BrightnessNormalize → ToTensor → Normalize
    """
    # Resize
    img = img.resize((224, 224), Image.BILINEAR)

    # 亮度归一化（与训练时一致）
    arr = np.array(img, dtype=np.float32) / 255.0
    brightness = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    current = brightness.mean()
    if current >= 0.01:
        factor = max(0.5, min(2.0, _TARGET_BRIGHTNESS / current))
        arr = np.clip(arr * factor, 0, 1)

    # Normalize (ImageNet 均值/标准差)
    arr = (arr - _MEAN) / _STD

    # HWC → CHW, 添加 batch 维度
    arr = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    return arr


def _softmax(x):
    """numpy softmax"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ==================== 预测线程 ====================
class PredictionThread(QThread):
    progress = Signal(str)
    finished = Signal(dict)
    batch_progress = Signal(int, int)
    error = Signal(str)

    def __init__(self, mode, data):
        super().__init__()
        self.mode = mode
        self.data = data

    def run(self):
        try:
            self.progress.emit("正在加载模型...")
            onnx_path = self.data['onnx_path']
            session = ort.InferenceSession(
                onnx_path, providers=['CPUExecutionProvider'])

            if self.mode == 'single':
                self._predict_one(session)
            else:
                self._predict_batch(session)
        except Exception as e:
            self.error.emit(f"预测失败: {e}")

    def _predict_one(self, session):
        """单图预测，返回完整 top5"""
        img_path = self.data['img_path']
        self.progress.emit("正在预测...")

        img = Image.open(img_path).convert('RGB')
        inp = preprocess_image(img)

        logits = session.run(None, {'input': inp})[0][0]
        probs = _softmax(logits)

        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = []
        for idx in top5_idx:
            en = config.FRUIT_CLASSES[idx]
            cn = config.FRUIT_NAMES_CN.get(en, en)
            top5.append({'cn_name': cn, 'en_name': en, 'confidence': float(probs[idx])})

        self.progress.emit("预测完成")
        self.finished.emit({'img_path': img_path, 'top5': top5})

    def _predict_batch(self, session):
        """批量预测，每张图片保存完整 top5"""
        image_dir = self.data['image_dir']
        self.progress.emit("正在扫描图像...")

        files = sorted(f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')))

        if not files:
            self.error.emit("未找到图像文件")
            return

        total = len(files)
        results = []

        for i, fname in enumerate(files):
            img_path = os.path.join(image_dir, fname)
            self.progress.emit(f"正在预测: {fname} ({i+1}/{total})")
            self.batch_progress.emit(i + 1, total)

            try:
                img = Image.open(img_path).convert('RGB')
                inp = preprocess_image(img)
                logits = session.run(None, {'input': inp})[0][0]
                probs = _softmax(logits)

                top5_idx = np.argsort(probs)[::-1][:5]
                top5 = []
                for idx in top5_idx:
                    en = config.FRUIT_CLASSES[idx]
                    cn = config.FRUIT_NAMES_CN.get(en, en)
                    top5.append({'cn_name': cn, 'en_name': en, 'confidence': float(probs[idx])})

                results.append({
                    'filename': fname, 'img_path': img_path,
                    'top5': top5,
                    'cn_name': top5[0]['cn_name'],
                    'en_name': top5[0]['en_name'],
                    'confidence': top5[0]['confidence'],
                    'marked': False, 'marked_correct': False,
                })
            except Exception:
                results.append({
                    'filename': fname, 'img_path': img_path,
                    'top5': [],
                    'cn_name': '错误', 'en_name': 'Error',
                    'confidence': 0.0,
                    'marked': False, 'marked_correct': False,
                })

        self.progress.emit("预测完成")
        self.finished.emit({'results': results})


# ==================== 单图保存对话框 ====================
class SaveResultDialog(QDialog):
    def __init__(self, img_path, prediction, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.prediction = prediction
        self.setWindowTitle("保存预测结果")
        self.setFixedSize(420, 240)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("选择要保存的内容："))

        self.chk_orig = QCheckBox("原始图片")
        self.chk_anno = QCheckBox("预测标注图"); self.chk_anno.setChecked(True)
        self.chk_chart = QCheckBox("Top-5 概率图"); self.chk_chart.setChecked(True)
        self.chk_text = QCheckBox("预测结果文本"); self.chk_text.setChecked(True)

        for w in (self.chk_orig, self.chk_anno, self.chk_chart, self.chk_text):
            layout.addWidget(w)

        layout.addSpacing(12)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("选择保存位置")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _save(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not d:
            return
        try:
            stem = Path(self.img_path).stem
            if self.chk_orig.isChecked():
                Image.open(self.img_path).save(os.path.join(d, f"{stem}_original.png"))
            if self.chk_anno.isChecked():
                self._save_annotated(os.path.join(d, f"{stem}_predicted.png"))
            if self.chk_chart.isChecked():
                self._save_chart(os.path.join(d, f"{stem}_top5.png"))
            if self.chk_text.isChecked():
                self._save_text(os.path.join(d, f"{stem}_result.txt"))
            QMessageBox.information(self, "成功", f"已保存到:\n{d}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _save_annotated(self, path):
        img = Image.open(self.img_path).convert('RGB').resize(config.IMAGE_SIZE)
        draw = ImageDraw.Draw(img)
        try:
            fl = ImageFont.truetype("msyh.ttc", 36)
            fs = ImageFont.truetype("msyh.ttc", 20)
        except Exception:
            fl = fs = ImageFont.load_default()
        t = self.prediction['top5'][0]
        draw.rectangle([0, 0, 224, 75], fill=(0, 0, 0, 180))
        draw.text((8, 5), f"预测: {t['cn_name']}", font=fl, fill=(255, 255, 255))
        draw.text((8, 42), f"({t['en_name']}) {t['confidence']*100:.1f}%", font=fs, fill=(180, 255, 180))
        img.save(path)

    def _save_chart(self, path):
        top5 = self.prediction['top5']
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{t['cn_name']} ({t['en_name']})" for t in top5]
        vals = [t['confidence'] * 100 for t in top5]
        colors = ['#4a90e2'] + ['#aaa'] * 4
        ax.barh(labels[::-1], vals[::-1], color=colors)
        ax.set_xlabel('置信度 (%)')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor='white')
        plt.close(fig)

    def _save_text(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"图片: {self.img_path}\n\n")
            for i, t in enumerate(self.prediction['top5']):
                f.write(f"#{i+1}. {t['cn_name']} ({t['en_name']})  {t['confidence']*100:.2f}%\n")


# ==================== 统计报告对话框 ====================
class StatisticsReportDialog(QDialog):
    def __init__(self, batch_results, parent=None):
        super().__init__(parent)
        self.results = batch_results
        self.figs = {}  # 保存 fig 引用，供导出用
        self.setWindowTitle("批量预测统计报告")
        self.setMinimumSize(950, 680)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # ---- 统计摘要 ----
        summary = QGroupBox("统计摘要")
        sg = QGridLayout()
        total = len(self.results)
        marked = sum(1 for r in self.results if r['marked'])
        correct = sum(1 for r in self.results if r.get('marked_correct'))
        wrong = marked - correct
        valid = [r for r in self.results if r['confidence'] > 0]
        avg_conf = np.mean([r['confidence'] for r in valid]) * 100 if valid else 0

        row = 0
        sg.addWidget(QLabel(f"总图片数: {total}"), row, 0)
        sg.addWidget(QLabel(f"已标记: {marked}"), row, 1)
        if marked > 0:
            row += 1
            sg.addWidget(QLabel(f"正确: {correct}"), row, 0)
            sg.addWidget(QLabel(f"错误: {wrong}"), row, 1)
            row += 1
            acc = correct / marked * 100
            acc_label = QLabel(f"准确率: {acc:.1f}%")
            acc_label.setStyleSheet("font-weight: bold;")
            sg.addWidget(acc_label, row, 0)
        else:
            row += 1
            note = QLabel("(尚未标记任何图片，准确率暂不可用)")
            note.setStyleSheet("color: #666; font-style: italic;")
            sg.addWidget(note, row, 0, 1, 2)

        row += 1
        sg.addWidget(QLabel(f"平均置信度: {avg_conf:.1f}%"), row, 0)
        summary.setLayout(sg)
        layout.addWidget(summary)

        # ---- 图表 Tab ----
        self.tabs = QTabWidget()

        # 置信度分布 Tab
        self.tabs.addTab(self._build_confidence_tab(), "置信度分布")

        # 混淆对 Tab
        self.tabs.addTab(self._build_confusion_tab(), "混淆对")

        # 详细结果 Tab
        self.tabs.addTab(self._build_detail_tab(), "详细结果")

        layout.addWidget(self.tabs, 1)

        # ---- 底部按钮 ----
        btn_row = QHBoxLayout()
        export_btn = QPushButton("保存图表")
        export_btn.setProperty("type", "save")
        export_btn.clicked.connect(self._export_charts)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(export_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ---- 置信度分布 ----
    def _build_confidence_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        valid = [r for r in self.results if r['confidence'] > 0]
        if not valid:
            lay.addWidget(QLabel("暂无有效预测结果。"))
            return w

        all_c = [r['confidence'] for r in valid]
        correct_c = [r['confidence'] for r in valid if r.get('marked_correct')]
        wrong_c = [r['confidence'] for r in valid if r.get('marked') and not r.get('marked_correct')]
        has_marked = bool(correct_c or wrong_c)

        fig = Figure(figsize=(9, 4))
        ax = fig.add_subplot(111)
        bins = np.arange(0, 1.05, 0.05)

        if has_marked:
            if correct_c:
                ax.hist(correct_c, bins=bins, color='#27ae60', alpha=0.6,
                        label=f'正确 (n={len(correct_c)})', edgecolor='white')
            if wrong_c:
                ax.hist(wrong_c, bins=bins, color='#e74c3c', alpha=0.6,
                        label=f'错误 (n={len(wrong_c)})', edgecolor='white')
            ax.set_title('置信度分布 (正确 vs 错误)')
        else:
            ax.hist(all_c, bins=bins, color='#4a90e2', alpha=0.7,
                    label=f'全部 (n={len(all_c)})', edgecolor='white')
            ax.set_title('置信度分布 (全部预测)')

        ax.set_xlabel('置信度')
        ax.set_ylabel('数量')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()

        self.figs['confidence'] = fig
        lay.addWidget(FigureCanvas(fig))
        return w

    # ---- 混淆对 ----
    def _build_confusion_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        wrong_results = [r for r in self.results if r['marked'] and not r['marked_correct']]

        if not wrong_results:
            msg = "暂无混淆数据。"
            if sum(1 for r in self.results if r['marked']) == 0:
                msg = "尚未标记任何图片，混淆对暂不可用。"
            elif not wrong_results:
                msg = "所有标记图片均预测正确，无混淆对。"
            lay.addWidget(QLabel(msg))
            return w

        # 用 QTextBrowser 展示混淆对列表
        text = QTextEdit()
        text.setReadOnly(True)
        content = f"共 {len(wrong_results)} 张图片被标记为预测错误：\n\n"

        # 按预测类别分组
        by_pred = {}
        for r in wrong_results:
            key = r['cn_name']
            if key not in by_pred:
                by_pred[key] = []
            by_pred[key].append(r)

        for pred_class, items in sorted(by_pred.items(), key=lambda x: -len(x[1])):
            content += f"【预测为 {pred_class}】({len(items)} 张)\n"
            for item in items:
                conf_str = f"{item['confidence']*100:.1f}%"
                content += f"  - {item['filename']}  (置信度: {conf_str})\n"
                if item['top5'] and len(item['top5']) >= 2:
                    alts = "、".join(
                        f"{t['cn_name']}({t['confidence']*100:.1f}%)"
                        for t in item['top5'][1:4]
                    )
                    content += f"    其他候选: {alts}\n"
            content += "\n"

        text.setText(content)
        lay.addWidget(text)
        return w

    # ---- 详细结果 ----
    def _build_detail_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        text = QTextEdit()
        text.setReadOnly(True)

        lines = ["=" * 80, "详细预测结果", "=" * 80, ""]

        for i, r in enumerate(self.results):
            if r['marked']:
                status = "正确" if r['marked_correct'] else "错误"
            else:
                status = "未标记"
            lines.append(f"[{i+1:3d}] {r['filename']} - {status}")
            lines.append(f"      预测: {r['cn_name']} ({r['en_name']})  置信度: {r['confidence']*100:.1f}%")
            if r.get('top5') and len(r['top5']) > 1:
                top5_str = "  |  ".join(
                    f"{t['cn_name']} {t['confidence']*100:.1f}%"
                    for t in r['top5']
                )
                lines.append(f"      Top-5: {top5_str}")
            lines.append("")

        text.setText("\n".join(lines))
        lay.addWidget(text)
        return w

    # ---- 导出图表 ----
    def _export_charts(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not d:
            return
        try:
            saved = []
            for name, fig in self.figs.items():
                path = os.path.join(d, f"{name}.png")
                fig.savefig(path, dpi=150, facecolor='white')
                saved.append(path)
            if saved:
                QMessageBox.information(self, "成功", f"图表已保存到:\n" + "\n".join(saved))
            else:
                QMessageBox.information(self, "提示", "当前无可保存的图表。")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))


# ==================== 主窗口 ====================
class FruitPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(config.MODEL_DIR, 'best_resnet18_cbam.onnx')
        self._thread = None
        self.current_image_path = None
        self.single_prediction = None
        self.batch_results = []
        self.batch_index = 0
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("水果识别系统 - ResNet18 + CBAM")
        self.setGeometry(80, 80, 1200, 720)
        self.setStyleSheet(GLOBAL_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 标题
        title = QLabel("水果识别系统 v3")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Microsoft YaHei', 18, QFont.Bold))
        root.addWidget(title)

        # 模型选择
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型:"))
        self.model_label = QLabel("使用默认模型")
        self.model_label.setWordWrap(True)
        model_row.addWidget(self.model_label, 1)
        btn_model = QPushButton("选择模型")
        btn_model.setMaximumWidth(90)
        btn_model.clicked.connect(self._on_select_model)
        model_row.addWidget(btn_model)
        root.addLayout(model_row)

        # Tabs
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_single_tab(), "单图预测")
        self.tabs.addTab(self._build_batch_tab(), "批量预测")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

    # ==================== 单图预测 Tab ====================
    def _build_single_tab(self):
        w = QWidget()
        h = QHBoxLayout(w)
        splitter = QSplitter(Qt.Horizontal)

        # 左：预览
        left = QWidget()
        lv = QVBoxLayout(left)

        g_preview = QGroupBox("图片预览")
        gpv = QVBoxLayout()
        self.s_img = QLabel("请选择或拖拽图片到此处")
        self.s_img.setAlignment(Qt.AlignCenter)
        self.s_img.setMinimumSize(420, 420)
        self.s_img.setStyleSheet(
            "QLabel{border:2px dashed #aaa; border-radius:4px;"
            "background:#fafafa; color:#666; font-size:13px;}"
        )
        self.s_img.setAcceptDrops(True)
        gpv.addWidget(self.s_img)
        g_preview.setLayout(gpv)
        lv.addWidget(g_preview)

        btns = QHBoxLayout()
        b_sel = QPushButton("选择图片"); b_sel.clicked.connect(self._on_select_image)
        self.s_btn_pred = QPushButton("开始预测")
        self.s_btn_pred.clicked.connect(self._on_predict_single)
        self.s_btn_save = QPushButton("保存结果")
        self.s_btn_save.setProperty("type", "save")
        self.s_btn_save.clicked.connect(self._on_save_single)
        self.s_btn_save.setEnabled(False)
        btns.addWidget(b_sel); btns.addWidget(self.s_btn_pred); btns.addWidget(self.s_btn_save)
        lv.addLayout(btns)

        # 右：结果
        right = QWidget()
        rv = QVBoxLayout(right)
        g_result = QGroupBox("Top-5 预测结果")
        grv = QVBoxLayout()
        self.s_labels = []
        for i in range(5):
            lb = QLabel(f"{i+1}. -")
            lb.setFont(QFont('Microsoft YaHei', 12))
            grv.addWidget(lb)
            self.s_labels.append(lb)
        g_result.setLayout(grv)
        rv.addWidget(g_result)
        rv.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        h.addWidget(splitter)
        return w

    # ==================== 批量预测 Tab ====================
    def _build_batch_tab(self):
        w = QWidget()
        h = QHBoxLayout(w)
        splitter = QSplitter(Qt.Horizontal)

        # 左：预览 + 控制
        left = QWidget()
        lv = QVBoxLayout(left)

        # 目录行
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("图片目录:"))
        self.b_dir_label = QLabel("未选择")
        dir_row.addWidget(self.b_dir_label, 1)
        b_dir = QPushButton("选择目录")
        b_dir.clicked.connect(self._on_select_batch_dir)
        dir_row.addWidget(b_dir)
        lv.addLayout(dir_row)

        # 批量预测按钮
        self.b_btn_pred = QPushButton("批量预测所有图片")
        self.b_btn_pred.setEnabled(False)
        self.b_btn_pred.clicked.connect(self._on_predict_batch)
        lv.addWidget(self.b_btn_pred)

        # 预览
        g_preview = QGroupBox("图片预览")
        gpv = QVBoxLayout()
        self.b_img = QLabel("请先选择目录并开始预测")
        self.b_img.setAlignment(Qt.AlignCenter)
        self.b_img.setMinimumSize(420, 380)
        self.b_img.setStyleSheet(
            "QLabel{border:1px solid #aaa; border-radius:4px;"
            "background:#fafafa; color:#666;}"
        )
        gpv.addWidget(self.b_img)
        g_preview.setLayout(gpv)
        lv.addWidget(g_preview)

        # 导航
        nav = QHBoxLayout()
        self.b_btn_prev = QPushButton("上一张")
        self.b_btn_prev.setProperty("type", "nav")
        self.b_btn_prev.clicked.connect(lambda: self._show_batch(self.batch_index - 1))
        self.b_btn_prev.setEnabled(False)

        self.b_idx = QLabel("0 / 0")
        self.b_idx.setAlignment(Qt.AlignCenter)
        self.b_idx.setStyleSheet("font-size:14px; font-weight:bold; color: #1a1a1a;")

        self.b_btn_next = QPushButton("下一张")
        self.b_btn_next.setProperty("type", "nav")
        self.b_btn_next.clicked.connect(lambda: self._show_batch(self.batch_index + 1))
        self.b_btn_next.setEnabled(False)

        nav.addWidget(self.b_btn_prev)
        nav.addWidget(self.b_idx, 1)
        nav.addWidget(self.b_btn_next)
        lv.addLayout(nav)

        # 标记
        mark_row = QHBoxLayout()
        self.b_btn_ok = QPushButton("预测正确")
        self.b_btn_ok.setProperty("type", "ok")
        self.b_btn_ok.clicked.connect(lambda: self._mark(True))
        self.b_btn_ok.setEnabled(False)

        self.b_btn_err = QPushButton("预测错误")
        self.b_btn_err.setProperty("type", "err")
        self.b_btn_err.clicked.connect(lambda: self._mark(False))
        self.b_btn_err.setEnabled(False)

        mark_row.addWidget(self.b_btn_ok)
        mark_row.addWidget(self.b_btn_err)
        lv.addLayout(mark_row)

        # 报告
        self.b_btn_report = QPushButton("查看统计报告")
        self.b_btn_report.setEnabled(False)
        self.b_btn_report.clicked.connect(self._on_report)
        lv.addWidget(self.b_btn_report)

        # 右：结果
        right = QWidget()
        rv = QVBoxLayout(right)

        g_res = QGroupBox("预测结果")
        grv = QVBoxLayout()
        self.b_labels = []
        for i in range(5):
            lb = QLabel(f"{i+1}. -")
            lb.setFont(QFont('Microsoft YaHei', 12))
            grv.addWidget(lb)
            self.b_labels.append(lb)
        g_res.setLayout(grv)
        rv.addWidget(g_res)

        self.b_status = QLabel("状态: 未标记")
        self.b_status.setStyleSheet(
            "padding:8px; background:#ecf0f1; border-radius:4px; color:#444; font-weight:bold;")
        rv.addWidget(self.b_status)

        rv.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        h.addWidget(splitter)
        return w

    # ==================== 事件 ====================
    def _on_select_model(self):
        default_dir = config.MODEL_DIR if os.path.isdir(config.MODEL_DIR) else os.path.join(os.getcwd(), "models")
        p, _ = QFileDialog.getOpenFileName(
            self, "选择模型", default_dir,
            "ONNX 模型 (*.onnx);;所有文件 (*)")
        if p:
            self.model_path = p
            self.model_label.setText(p)
            self.status.showMessage(f"已选择模型: {p}")

    # ---- 单图 ----
    def _on_select_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
        if p:
            self._load_single(p)

    def _load_single(self, path):
        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.warning(self, "错误", "无法加载图片"); return
        self.current_image_path = path
        self.s_img.setPixmap(pm.scaled(self.s_img.size(),
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status.showMessage(f"已加载: {Path(path).name}")
        # 清除旧结果
        for lb in self.s_labels:
            lb.setText(f"{self.s_labels.index(lb)+1}. -")
            lb.setStyleSheet("color: #999;")
        self.single_prediction = None
        self.s_btn_pred.setEnabled(True)
        self.s_btn_save.setEnabled(False)

    def _on_predict_single(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择图片"); return
        self.s_btn_pred.setEnabled(False)
        self.s_btn_pred.setText("预测中...")
        self.s_btn_pred.setProperty("type", "warn")
        self.s_btn_pred.style().unpolish(self.s_btn_pred)
        self.s_btn_pred.style().polish(self.s_btn_pred)
        self.status.showMessage("正在预测...")

        self._thread = PredictionThread('single', {
            'img_path': self.current_image_path,
            'onnx_path': self.model_path})
        self._thread.progress.connect(self.status.showMessage)
        self._thread.finished.connect(self._on_single_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_single_done(self, result):
        self.single_prediction = result
        top5 = result['top5']
        for i, lb in enumerate(self.s_labels):
            if i < len(top5):
                t = top5[i]
                lb.setText(f"{i+1}. {t['cn_name']} ({t['en_name']}) - {t['confidence']*100:.1f}%")
                if i == 0:
                    lb.setStyleSheet(
                        "font-size:14px; font-weight:bold; color:#1a6b37;"
                        "padding:4px; background:#e8f8ef; border-radius:3px;")
                else:
                    lb.setStyleSheet("color: #333;")
            else:
                lb.setText(f"{i+1}. -")
                lb.setStyleSheet("color: #999;")

        self.s_btn_pred.setEnabled(True)
        self.s_btn_pred.setText("开始预测")
        self.s_btn_pred.setProperty("type", "")
        self.s_btn_pred.style().unpolish(self.s_btn_pred)
        self.s_btn_pred.style().polish(self.s_btn_pred)
        self.s_btn_save.setEnabled(True)
        self.status.showMessage(
            f"预测完成: {top5[0]['cn_name']} - {top5[0]['confidence']*100:.1f}%")

    def _on_save_single(self):
        if not self.single_prediction:
            return
        SaveResultDialog(self.current_image_path, self.single_prediction, self).exec()

    # ---- 批量 ----
    def _on_select_batch_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if d:
            self.batch_dir = d
            self.b_dir_label.setText(d)
            self.b_btn_pred.setEnabled(True)
            self.status.showMessage(f"已选择目录: {d}")

    def _on_predict_batch(self):
        if not hasattr(self, 'batch_dir'):
            return
        self.b_btn_pred.setEnabled(False)
        self.b_btn_pred.setText("正在预测...")
        self.b_btn_pred.setProperty("type", "warn")
        self.b_btn_pred.style().unpolish(self.b_btn_pred)
        self.b_btn_pred.style().polish(self.b_btn_pred)

        self._thread = PredictionThread('batch', {
            'image_dir': self.batch_dir,
            'onnx_path': self.model_path})
        self._thread.progress.connect(self.status.showMessage)
        self._thread.batch_progress.connect(
            lambda c, t: self.b_btn_pred.setText(f"预测中: {c}/{t}"))
        self._thread.finished.connect(self._on_batch_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_batch_done(self, result):
        self.batch_results = result['results']
        self.batch_index = 0

        self.b_btn_pred.setText("预测完成")
        self.b_btn_pred.setProperty("type", "save")
        self.b_btn_pred.style().unpolish(self.b_btn_pred)
        self.b_btn_pred.style().polish(self.b_btn_pred)

        if self.batch_results:
            self._show_batch(0)
            # 只要有预测结果就可以查看报告
            self.b_btn_report.setEnabled(True)
            self.status.showMessage(f"批量预测完成: {len(self.batch_results)} 张图片")

    def _show_batch(self, idx):
        if not (0 <= idx < len(self.batch_results)):
            return
        self.batch_index = idx
        r = self.batch_results[idx]

        pm = QPixmap(r['img_path'])
        if not pm.isNull():
            self.b_img.setPixmap(pm.scaled(self.b_img.size(),
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))

        marked = sum(1 for x in self.batch_results if x['marked'])
        self.b_idx.setText(f"{idx+1} / {len(self.batch_results)}   已标记: {marked}")

        self.b_btn_prev.setEnabled(idx > 0)
        self.b_btn_next.setEnabled(idx < len(self.batch_results) - 1)

        # 显示 top5
        top5 = r.get('top5', [])
        for i, lb in enumerate(self.b_labels):
            if i < len(top5):
                t = top5[i]
                lb.setText(f"{i+1}. {t['cn_name']} ({t['en_name']}) - {t['confidence']*100:.1f}%")
                if i == 0:
                    lb.setStyleSheet(
                        "font-size:14px; font-weight:bold; color:#1a6b37;"
                        "padding:4px; background:#e8f8ef; border-radius:3px;")
                else:
                    lb.setStyleSheet("color: #333;")
            else:
                lb.setText(f"{i+1}. -")
                lb.setStyleSheet("color: #999;")

        # 标记状态
        self._refresh_mark_ui(r)

    def _refresh_mark_ui(self, r):
        if not r['marked']:
            self.b_status.setText("状态: 未标记")
            self.b_status.setStyleSheet(
                "padding:8px; background:#ecf0f1; border-radius:4px;"
                "color:#444; font-weight:bold;")
            self.b_btn_ok.setEnabled(True)
            self.b_btn_err.setEnabled(True)
        elif r['marked_correct']:
            self.b_status.setText("状态: 已标记为正确")
            self.b_status.setStyleSheet(
                "padding:8px; background:#d5f4e6; border-radius:4px;"
                "color:#1a6b37; font-weight:bold;")
            self.b_btn_ok.setEnabled(False)
            self.b_btn_err.setEnabled(True)
        else:
            self.b_status.setText("状态: 已标记为错误")
            self.b_status.setStyleSheet(
                "padding:8px; background:#fadbd8; border-radius:4px;"
                "color:#a93226; font-weight:bold;")
            self.b_btn_ok.setEnabled(True)
            self.b_btn_err.setEnabled(False)

    def _mark(self, correct):
        r = self.batch_results[self.batch_index]
        r['marked'] = True
        r['marked_correct'] = correct
        self._refresh_mark_ui(r)

        marked = sum(1 for x in self.batch_results if x['marked'])
        self.b_idx.setText(f"{self.batch_index+1} / {len(self.batch_results)}   已标记: {marked}")

        if marked == len(self.batch_results):
            self.status.showMessage("所有图片已标记")

    def _on_report(self):
        StatisticsReportDialog(self.batch_results, self).exec()

    def _on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.status.showMessage("预测失败")
        # 恢复按钮状态
        for btn in (self.s_btn_pred, self.b_btn_pred):
            btn.setEnabled(True)
            btn.setText("开始预测" if btn is self.s_btn_pred else "批量预测所有图片")
            btn.setProperty("type", "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)


# ==================== 入口 ====================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = FruitPredictorGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
