# Degradation-Aware Training for Robust Semantic Segmentation（项目复现代码）

核心思想是：
- 使用 **PASCAL VOC 2012** 做语义分割；
- 使用 **DeepLabv3 / U-Net** 做 baseline；
- 在训练阶段对输入图像随机施加退化（高斯噪声 / 运动模糊 / JPEG压缩），形成 **degradation-aware training**；
- 用 mIoU、定性可视化、ablation（退化概率 / 单退化类型贡献）验证鲁棒性提升。


---

## 一、环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行环境
- Python 3.10+
- CUDA GPU（8GB 可跑 U-Net；12GB+ 更适合 DeepLabv3）

---

## 二、数据集准备（VOC2012）

下载后目录应类似：
```text
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/
```

## 三、先跑通 baseline

### 1) DeepLabv3 baseline（clean-only）
```bash
python -m src.train --config configs/unet_baseline.yaml --run_name unet_baseline_cpu
```

### 2) 评估 baseline（干净验证集）
```bash
python -m src.evaluate \
  --config configs/unet_baseline.yaml \
  --ckpt runs/unet_baseline_cpu/best.pt \
  --split val
```

---

## 四、退化感知训练（核心方法）

```bash
python -m src.train --config configs/deeplabv3_degaware.yaml
```

关键配置（`configs/deeplabv3_degaware.yaml`）：
- `degradation.enable: true`
- `degradation.prob: 0.7`
- `degradation.types: [gaussian_noise, motion_blur, jpeg]`
- `degradation.min_severity/max_severity: 1~5`
- `degradation.one_of: true`（每次随机选一种退化）

---

## 五、鲁棒性评估（不同退化与强度）

### 1) 指定退化类型 + 强度评估（例：JPEG severity=3）
```bash
python -m src.evaluate   --config configs/deeplabv3_degaware.yaml   --ckpt runs/deeplabv3_degaware/best.pt   --eval_degradation jpeg   --eval_severity 3
```

### 2) 自动生成 robustness 曲线（3种退化 × 5档强度）
```bash
python scripts/run_ablation_grid.py   --mode robustness   --config configs/deeplabv3_degaware.yaml   --ckpt runs/deeplabv3_degaware/best.pt   --out_dir ./analysis_outputs
```

输出：
- `analysis_outputs/robustness_results.csv`
- `analysis_outputs/robustness_curve.png`

---

## 六、Ablation

### A. 退化概率 p 的影响
复制 degaware config，修改：
- `degradation.prob: 0.3 / 0.5 / 0.7 / 0.9`

然后分别训练，最后用 compare_runs 汇总：
```bash
python scripts/run_ablation_grid.py   --mode compare_runs   --config configs/deeplabv3_degaware.yaml   --runs     baseline=runs/deeplabv3_baseline/best.pt     p03=runs/deeplabv3_degaware_p03/best.pt     p05=runs/deeplabv3_degaware_p05/best.pt     p07=runs/deeplabv3_degaware/best.pt   --out_dir ./analysis_outputs_compare
```

### B. 单退化类型贡献
修改 `degradation.types`：
- `[gaussian_noise]`
- `[motion_blur]`
- `[jpeg]`
- `[gaussian_noise, motion_blur, jpeg]`

做法同上。

---

## 七、定性可视化

```bash
python scripts/visualize_predictions.py   --config configs/deeplabv3_degaware.yaml   --ckpt runs/deeplabv3_degaware/best.pt   --indices 0 5 10   --eval_degradation motion_blur   --eval_severity 4   --out_dir ./qual_vis
```

每个样本会输出：
- 原图 `orig.png`
- 退化图 `degraded.png`
- resize/crop 后输入图 `degraded_resized.png`
- GT 彩色 mask
- Pred 彩色 mask
- GT / Pred overlay（可直接放 slide）

---

## 八、可选优化：boundary correlation loss（proposal 的 Optimization 项）

在 config 中打开：
```yaml
loss:
  boundary_weight: 0.15
```

实现方式是轻量版：
- 用 Sobel 提取预测概率图与 GT one-hot 的边界响应；
- 用 L1 损失约束边界一致性；
- 与 CE loss 组合训练。

> 建议先完成 baseline + deg-aware + robustness 曲线，再尝试边界 loss。

---

## 九、实验顺序

### Phase 1（最小闭环）
1. 下载 VOC2012
2. 跑通 DeepLabv3 baseline
3. baseline 在 3 种退化（severity=3）上评估（证明问题存在）
4. 跑通 DeepLabv3 degradation-aware
5. 对比 robustness 曲线（初步结果）

### Phase 2（完整结果）
6. p 的 ablation
7. 单退化类型 ablation
8. 跑 U-Net baseline / deg-aware（验证方法泛化）
9. 整理定性图（overlay）

### Phase 3（加分）
10. boundary loss
11. 边界案例分析与失败案例分析

---

## 十、常见问题排查

### 1) 显存不够
- `train.batch_size` 改小（例如 4→2）
- `data.transform.crop_size` 改小（480→384 或 320）
- U-Net 先跑通再上 DeepLabv3

### 2) mIoU 异常低
- 确保 `loss.ignore_index = 255`
- mask 插值必须是最近邻（代码中已处理）
- 确认训练/验证 split 使用的是 VOC segmentation 的 `train` / `val`

### 3) 退化太强训练不稳定
- 降低 `degradation.prob`
- 降低 `max_severity`（先跑 1~3）
- 保持 `one_of: true`（避免叠加多种退化）

---

## 十一、与 presentation 四个板块的对应关系

1. **Problem trying to solve**  
   用 baseline 在退化下 mIoU 明显下降来展示问题（+ 退化样例图）

2. **Data that using**  
   VOC2012 + synthetic degradations（noise / blur / JPEG，多强度）

3. **Proposed solutions**  
   Degradation-aware stochastic training pipeline（训练时随机退化）

4. **Results we have so far**  
   Robustness 曲线、Ablation 表、Qualitative overlay

---

## 十二、重要文件说明（你最常改的地方）

- `configs/*.yaml`：实验配置（模型、训练、退化、loss）
- `src/degradations.py`：退化实现（类型/强度映射）
- `src/train.py`：训练主程序
- `src/evaluate.py`：鲁棒性评估
- `scripts/run_ablation_grid.py`：批量评估与绘图
- `scripts/visualize_predictions.py`：定性可视化导出
