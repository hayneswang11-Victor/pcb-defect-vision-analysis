# PCB缺陷机器视觉检测与质量分析

基于 **DeepPCB + OpenCV + YOLOv8n** 的电子制造缺陷检测与质量分析项目。项目重点不是堆叠复杂模型，而是建立一条完整、可复现的工程流程：

**数据审计 → 传统视觉基线 → YOLO缺陷检测 → 独立测试集评估 → 阈值选择 → FP/FN错误分析 → 自动归因与人工复核 → 结果冻结**

> 本项目使用公开研究数据，不代表真实工厂 AOI 部署、真实产线良率提升或相机/光源/运动平台集成经验。

---

## 1. 项目概览

本项目使用 DeepPCB 六类 PCB 缺陷数据，对传统图像处理方法和深度学习目标检测方法进行对比，并进一步将模型输出转化为制造质量分析视角下的误报、漏检、类别错误与阈值权衡。

六类缺陷：

- open
- short
- mousebite
- spur
- copper
- pin-hole

数据审计后得到：

| 数据项 | 数量 |
|---|---:|
| 有效图像 | 1,500 |
| 缺陷框总数 | 10,013 |
| Train | 850 images / 5,834 boxes |
| Validation | 150 images / 1,039 boxes |
| Test | 500 images / 3,140 boxes |

原始 DeepPCB 数据不包含在本仓库中。数据来源：  
https://github.com/tangsanli5201/DeepPCB

---

## 2. 技术路线

### 2.1 OpenCV传统视觉基线

使用测试图与无缺陷模板图完成：

1. 灰度化与轻度平滑
2. 模板绝对差分
3. 固定阈值分割
4. 形态学开/闭运算
5. 连通域提取与候选框生成
6. IoU 匹配与 Precision / Recall / F1 计算

仅在 Validation 集上进行 30 组轻量参数扫描，参数冻结后再评估独立 Test 集。

冻结 Test 结果：

| Precision | Recall | F1 |
|---:|---:|---:|
| 0.0726 | 0.0736 | 0.0731 |

该结果作为传统规则方法的对照基线，不再针对 Test 集继续调参。

### 2.2 YOLOv8n 六类缺陷检测

训练配置：

| 参数 | 设置 |
|---|---|
| 模型 | YOLOv8n |
| 输入尺寸 | 640 × 640 |
| Epochs | 40 |
| Batch | 8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 训练集 | 850 images |
| 验证集 | 150 images |
| 独立测试集 | 500 images |

Validation：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.981 | 0.954 | 0.987 | 0.770 |

独立 Test：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| **0.951** | **0.928** | **0.969** | **0.738** |

### 2.3 两个冻结工作点

Best-F1 工作点：

| Confidence | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.55 | 0.9437 | 0.9398 | **0.9418** |

质量分析工作点：

| Confidence | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.65 | **0.9635** | 0.9153 | 0.9388 |

质量工作点用于后续类别指标、FP/FN统计和错误分析。

---

## 3. 质量分析结果

在质量工作点下：

| 指标 | 结果 |
|---|---:|
| Macro Precision | 0.9619 |
| Macro Recall | 0.9157 |
| Macro F1 | 0.9375 |
| Total FP | 109 |
| Total FN | 266 |

类别层面，`short` 是相对主要的错误来源之一。项目进一步对错误样本进行了分层分析，而不是仅报告单一 mAP。

---

## 4. Top错误样本自动归因

对 Top 20 高错误样本进行自动化错误审计，共得到 **87 个错误事件**。

### FN归因

| 原因 | 数量 |
|---|---:|
| low_confidence_miss | 49 |
| pure_miss | 14 |
| class_confusion | 9 |
| localization_iou | 1 |

### FP归因

| 原因 | 数量 |
|---|---:|
| background_false_positive | 9 |
| class_confusion | 4 |
| localization_iou | 1 |

自动归因同时计算目标尺寸、局部对比度和局部边缘密度等视觉因素。脚本不会自动宣称数据标注错误，而是对无法由规则充分解释的样本设置 `manual_check_required=True`。

自动审计标记的 **13 张疑难图已完成人工检查**，人工检查仅用于错误解释，没有重新训练模型，也没有继续使用 Test 集调参。

---

## 5. 项目结构

```text
pcb-defect-vision-analysis/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ project.yaml
├─ src/
│  ├─ common.py
│  ├─ 00_audit_raw_dataset.py
│  ├─ 01_prepare_yolo_dataset.py
│  ├─ 02_opencv_baseline.py
│  ├─ 02a_tune_opencv_val.py
│  ├─ 02b_opencv_test_frozen.py
│  ├─ 03_train_yolo.py
│  ├─ 04_evaluate_yolo.py
│  ├─ 05_quality_analysis.py
│  └─ 06_automated_error_review.py
├─ docs/
│  ├─ 数据说明.md
│  ├─ 运行步骤.md
│  └─ PCB缺陷机器视觉检测与质量分析_项目报告_v1.0.pdf
├─ results/
│  ├─ metrics/
│  ├─ figures/
│  └─ error_review/
└─ models/
   ├─ best.pt
   ├─ training_args.yaml
   └─ training_results.csv
```

原始数据、YOLO中间训练缓存、Python虚拟环境及大体积临时输出不上传 GitHub。

---

## 6. 运行环境

主要依赖：

- Python 3.12
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML
- PyTorch
- Ultralytics

安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若使用 NVIDIA GPU，需要安装与本机驱动兼容的 CUDA 版 PyTorch。

---

## 7. 数据准备

将 DeepPCB 放置到：

```text
data/raw/DeepPCB/
```

并确认：

```text
data/raw/DeepPCB/PCBData/
```

存在。

数据集本身不随仓库发布。

---

## 8. 复现流程

### Step 1：数据审计

```powershell
python src\00_audit_raw_dataset.py
```

### Step 2：转换为 YOLO 数据结构

```powershell
python src\01_prepare_yolo_dataset.py
```

### Step 3：OpenCV传统视觉基线

```powershell
python src\02_opencv_baseline.py
python src\02a_tune_opencv_val.py
python src\02b_opencv_test_frozen.py
```

### Step 4：YOLO训练

```powershell
python src\03_train_yolo.py
```

### Step 5：独立测试集评估与阈值分析

```powershell
python src\04_evaluate_yolo.py
```

### Step 6：质量分析

```powershell
python src\05_quality_analysis.py
```

### Step 7：Top错误样本自动归因

```powershell
python src\06_automated_error_review.py
```

---

## 9. 结果冻结

项目正式结果已建立冻结版本，包含：

- OpenCV Validation 参数扫描结果
- OpenCV Frozen Test
- YOLO 最佳权重与训练参数
- YOLO 独立 Test 指标
- 阈值扫描与两个冻结工作点
- 类别级质量指标
- Top 20 错误样本自动归因
- 13 张疑难案例人工检查记录
- SHA256 完整性校验表

冻结后不再根据 Test 集修改模型或参数。

---

## 10. 项目报告

完整项目报告见：

```text
docs/PCB缺陷机器视觉检测与质量分析_项目报告_v1.0.pdf
```

报告包含数据审计、传统视觉基线、YOLO训练与测试、阈值选择、类别分析、错误归因、人工复核及 Claim Boundary。

---

## 11. Claim Boundary

本项目能够支持以下表述：

- 使用 OpenCV 完成 PCB 图像差分、阈值分割、形态学处理与候选缺陷定位；
- 使用 YOLOv8n 完成六类 PCB 缺陷目标检测；
- 使用独立 Test 集评估 Precision、Recall、mAP 与 F1；
- 对 Confidence Threshold 进行冻结工作点选择；
- 对 FP/FN、类别错误和高错误样本进行自动归因与人工确认；
- 将模型结果转化为制造质量分析视角下的错误统计和可视化。

本项目**不代表**：

- 真实工厂 AOI 系统部署；
- 真实相机、镜头、光源和运动控制硬件集成；
- 企业生产数据管理经验；
- 真实产线良率提升；
- 企业级缺陷数据库或 MES/SPC 系统接入。


