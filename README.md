# DiffScanAuth

Language: **English (default)** | [中文](#中文说明)

---

## English

### Overview
This repository is the executable paper codebase for:

**DiffScanAuth: Human-Gaze-Supervised Sequential Evidence Accumulation for AI-Generated Image Detection**

It upgrades the earlier scaffold into a paper-aligned implementation with:

1. final baselines
2. final ablations
3. shuffled-gaze control
4. dummy-data mode before real data upload
5. end-to-end training, evaluation, prediction, visualization, and analysis scripts
6. leakage-safe image-level splitting
7. stage-wise support for teacher pretraining, student training, and optional efficiency refinement

### What Is Implemented

#### Main paper method
- `DiffScanAuth`
  - dual-stream encoder
  - diffusion-based scanpath teacher
  - causal gaze student
  - foveated evidence reader
  - evidence accumulator
  - learned stop or fixed-K inference

#### Main baselines
- `exp_vit_b16`: ViT-B/16 static classifier
- `exp_aide_style`: AIDE-style RGB + artifact/frequency hybrid detector
- `exp_vit_heatmap`: ViT + human gaze heatmap supervision
- `exp_seqdet_no_gaze`: sequential detector without human gaze supervision
- `exp_diffscanauth`: full DiffScanAuth

#### Ablations
- `ablation_no_gaze_supervision`
- `ablation_heatmap_instead_of_scanpath`
- `ablation_no_teacher_distillation`
- `ablation_no_local_stream`
- `ablation_fixed_k`

#### Control
- `control_shuffled_gaze`

### Current Validation Status
Validated locally on **March 20, 2026**:

1. `pytest -q` -> `9 passed`
2. `python scripts/smoke_test.py` -> passed
3. short dummy training passes completed for:
   - `exp_vit_b16_dummy_short`
   - `exp_seqdet_no_gaze_dummy_short`
   - `exp_diffscanauth_dummy_short`
4. example metrics, predictions, checkpoints, tables, and figures were generated under `outputs/`

### Environment

#### Official target
- Python `3.11`
- PyTorch `2.x`
- PyTorch Lightning
- Hydra / OmegaConf
- timm
- torchvision
- torchmetrics
- scikit-learn
- pandas
- numpy
- pillow / opencv-python
- matplotlib / seaborn
- einops
- tqdm

#### Optional
- transformers
- mamba-ssm
- wandb

#### Local note
The repository target remains **Python 3.11**, but the recent local validation on this machine also passed under the existing Miniforge Python `3.12` environment. For formal experiments, keep Python `3.11` as the paper environment.

### Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Quick Start

#### Full preprocessing pipeline
```bash
python scripts/inspect_dataset.py
python scripts/build_metadata.py
python scripts/preprocess_eye_tracking.py --force-rebuild-eye-tracking
python scripts/make_splits.py
python scripts/smoke_test.py
```

#### Train main experiments
```bash
python scripts/train.py experiment=exp_vit_b16
python scripts/train.py experiment=exp_aide_style
python scripts/train.py experiment=exp_vit_heatmap
python scripts/train.py experiment=exp_seqdet_no_gaze
python scripts/train.py experiment=exp_diffscanauth
```

#### Train ablations and control
```bash
python scripts/train.py experiment=ablation_no_gaze_supervision
python scripts/train.py experiment=ablation_heatmap_instead_of_scanpath
python scripts/train.py experiment=ablation_no_teacher_distillation
python scripts/train.py experiment=ablation_no_local_stream
python scripts/train.py experiment=ablation_fixed_k
python scripts/train.py experiment=control_shuffled_gaze
```

#### Evaluate / predict / analyze
```bash
python scripts/evaluate.py --experiment exp_diffscanauth --ckpt outputs/checkpoints/exp_diffscanauth/last.ckpt
python scripts/predict.py --experiment exp_diffscanauth --ckpt outputs/checkpoints/exp_diffscanauth/last.ckpt --split test
python scripts/analyze_results.py
python scripts/make_tables.py
python scripts/make_confidence_curve.py --predictions outputs/predictions/exp_diffscanauth_test_predictions.csv
python scripts/make_qualitative_figures.py --ours outputs/predictions/exp_diffscanauth_test_predictions.csv --no-gaze outputs/predictions/exp_seqdet_no_gaze_test_predictions.csv
```

#### Run the whole suite sequentially
```bash
python scripts/run_all_experiments.py --epochs 1
```

### Stage-Wise DiffScanAuth Training
The paper training protocol is supported explicitly.

#### Stage 1: teacher pretraining
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_teacher_stage1 \
  model.training_stage=teacher \
  model.use_teacher=true \
  model.use_teacher_distill=false
```

#### Stage 2: student + reader + accumulator + classifier
Load encoder + teacher weights from stage 1:
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_stage2 \
  model.training_stage=student \
  model.teacher_ckpt=outputs/checkpoints/exp_diffscanauth_teacher_stage1/last.ckpt \
  model.freeze_teacher=true
```

#### Stage 3: optional efficiency refinement
Load full model weights from stage 2:
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_stage3_refine \
  model.training_stage=refine \
  model.init_from_ckpt=outputs/checkpoints/exp_diffscanauth_stage2/last.ckpt \
  model.use_rl=true \
  model.loss_rl=0.05
```

### Dummy-Data Mode
You do **not** need to upload the real dataset before using the repo.

If `data/raw/` does not contain a valid dataset:

1. synthetic images are generated
2. synthetic metadata is built
3. synthetic eye-tracking trajectories are generated
4. splits and heatmaps are built
5. the entire training/evaluation stack remains runnable

Important commands:
```bash
python scripts/build_metadata.py --allow-synthetic --synthetic-num-images 120
python scripts/preprocess_eye_tracking.py --allow-synthetic --force-rebuild-eye-tracking
```

### Real Data Schema

#### `metadata.csv`
Required columns:
- `image_id`
- `image_path`
- `label`
- `scene`
- `source_type`
- `generator`
- `width`
- `height`
- `split`

#### `eye_tracking.csv`
Required columns:
- `subject_id`
- `image_id`
- `t`
- `x`
- `y`
- `duration`
- `event_type`
- `validity`
- `pupil`

#### `processed_fixations.csv`
Required columns:
- `subject_id`
- `image_id`
- `fixation_idx`
- `x_norm`
- `y_norm`
- `duration_norm`
- `duration_ms`
- `delta_x`
- `delta_y`
- `patch_index`
- `split`

### Data Split Safety
Splitting is done by **image_id**, not by subject-row.

This guarantees:

1. all subjects for the same image stay in the same split
2. train/val/test leakage across the same image is prevented
3. stratification is attempted in this order:
   - `label + scene + generator`
   - `label + scene`
   - `label`
   - random fallback with downgrade logging

Artifacts:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/split_report.json`

### Fallback Policy
The code is paper-oriented, but robust to limited environments.

1. Global semantic stream
   - preferred: SigLIP2-like adapter
   - fallback: timm transformer / torchvision-compatible backbone
2. Local artifact stream
   - preferred: DINOv2-like dense adapter
   - fallback: timm dense backbone / ConvNeXt / ResNet feature map
3. Accumulator
   - preferred: Mamba-style backend when available
   - fallback: GRU
4. Diffusion teacher
   - implemented in-repo as a lightweight, real working denoising teacher
5. Logging
   - wandb if available and configured
   - otherwise TensorBoard
   - otherwise CSV logger

### Output Layout
- `outputs/logs/`: Lightning logs
- `outputs/checkpoints/`: model checkpoints
- `outputs/metrics/`: JSON/CSV summaries
- `outputs/predictions/`: per-sample predictions
- `outputs/figures/`: ROC, PR, confusion matrices, confidence curves, qualitative scanpaths

### How To Replace Dummy Data With Real Data Later
When your real dataset is ready:

1. place images and raw tables anywhere under `data/raw/`
2. rerun:
   ```bash
   python scripts/inspect_dataset.py
   python scripts/build_metadata.py
   python scripts/preprocess_eye_tracking.py --force-rebuild-eye-tracking
   python scripts/make_splits.py
   ```
3. inspect:
   - `data/processed/metadata.csv`
   - `data/processed/eye_tracking.csv`
   - `data/processed/processed_fixations.csv`
   - `data/splits/split_report.json`
4. if your raw format differs, adapt only `src/datasets/dataset_adapter.py` and rerun the same pipeline

### File-By-File Guide

#### Root files
| File | Function |
|---|---|
| `README.md` | bilingual project manual and usage guide |
| `requirements.txt` | dependency list |
| `pyproject.toml` | project metadata and pytest configuration |
| `.gitignore` | ignores outputs, caches, and local artifacts |

#### `configs/`
| File | Function |
|---|---|
| `configs/config.yaml` | Hydra root config; default entry uses `diffscanauth` |
| `configs/data/default.yaml` | dataset paths, preprocessing settings, split ratios, loader settings, synthetic-data knobs |
| `configs/trainer/default.yaml` | shared Lightning trainer defaults |
| `configs/model/vit_b16.yaml` | final ViT-B/16 baseline config |
| `configs/model/aide_style.yaml` | final AIDE-style baseline config |
| `configs/model/vit_gaze_heatmap.yaml` | final ViT + heatmap supervision config |
| `configs/model/seqdet_no_gaze.yaml` | final sequential no-human-gaze baseline config |
| `configs/model/diffscanauth.yaml` | final full DiffScanAuth config |
| `configs/model/baseline_static.yaml` | legacy scaffold config kept for backward compatibility |
| `configs/model/baseline_heatmap.yaml` | legacy scaffold config kept for backward compatibility |
| `configs/model/seq_gaze_detector.yaml` | legacy scaffold config kept for backward compatibility |
| `configs/experiment/exp_vit_b16.yaml` | experiment preset for ViT-B/16 |
| `configs/experiment/exp_aide_style.yaml` | experiment preset for AIDE-style detector |
| `configs/experiment/exp_vit_heatmap.yaml` | experiment preset for ViT + gaze heatmap |
| `configs/experiment/exp_seqdet_no_gaze.yaml` | experiment preset for sequential no-gaze baseline |
| `configs/experiment/exp_diffscanauth.yaml` | experiment preset for full DiffScanAuth |
| `configs/experiment/ablation_no_gaze_supervision.yaml` | ablation: remove gaze supervision |
| `configs/experiment/ablation_heatmap_instead_of_scanpath.yaml` | ablation: heatmap supervision instead of scanpath supervision |
| `configs/experiment/ablation_no_teacher_distillation.yaml` | ablation: remove diffusion-teacher distillation |
| `configs/experiment/ablation_no_local_stream.yaml` | ablation: remove local artifact stream |
| `configs/experiment/ablation_fixed_k.yaml` | ablation: fixed-K instead of learned stop |
| `configs/experiment/control_shuffled_gaze.yaml` | control: shuffled gaze trajectories |
| `configs/experiment/baseline_static.yaml` | legacy experiment preset kept for compatibility |
| `configs/experiment/baseline_heatmap.yaml` | legacy experiment preset kept for compatibility |
| `configs/experiment/seq_gaze_detector.yaml` | legacy fixed-K sequential preset |
| `configs/experiment/seq_gaze_stop.yaml` | legacy learned-stop sequential preset |

#### `scripts/`
| File | Function |
|---|---|
| `scripts/inspect_dataset.py` | scans `data/raw/` and saves a structural inspection report |
| `scripts/build_metadata.py` | builds standardized `metadata.csv` from raw images or synthetic fallback |
| `scripts/preprocess_eye_tracking.py` | normalizes eye-tracking and produces `processed_fixations.csv` |
| `scripts/make_splits.py` | creates leakage-safe train/val/test splits and heatmaps |
| `scripts/train.py` | Hydra training entrypoint; also runs test and exports metrics/predictions/figures |
| `scripts/evaluate.py` | loads a checkpoint and evaluates it on `train` / `val` / `test` |
| `scripts/predict.py` | exports prediction CSVs from a checkpoint |
| `scripts/analyze_results.py` | aggregates prediction CSVs into metrics and plots |
| `scripts/make_tables.py` | builds summary tables from metrics JSON files |
| `scripts/make_qualitative_figures.py` | produces human vs ours vs no-gaze scanpath figure panels |
| `scripts/make_confidence_curve.py` | plots confidence-over-time from sequential predictions |
| `scripts/run_all_experiments.py` | sequentially runs the full main-paper experiment suite |
| `scripts/smoke_test.py` | fast executable regression test over all final model families |

#### `src/`
| File | Function |
|---|---|
| `src/__init__.py` | package marker |

#### `src/datasets/`
| File | Function |
|---|---|
| `src/datasets/dataset_schema.py` | simple schema definitions and typed records |
| `src/datasets/dataset_adapter.py` | raw-data scanner, metadata builder, synthetic-data generator, eye-tracking adapter |
| `src/datasets/image_dataset.py` | image-level dataset for static classifiers |
| `src/datasets/gaze_dataset.py` | subject-image sequential gaze dataset with optional shuffled-gaze control |
| `src/datasets/collate.py` | batch collation for static and sequential models |
| `src/datasets/transforms.py` | gaze-safe image transforms |
| `src/datasets/split_utils.py` | image-level stratified split logic and leakage checks |

#### `src/features/`
| File | Function |
|---|---|
| `src/features/gaze_processing.py` | normalization, fixation conversion, duration normalization, delta computation |
| `src/features/heatmap.py` | heatmap generation and serialization |
| `src/features/fixation_tokenizer.py` | patch tokenization and patch histogram helpers |

#### `src/models/`
| File | Function |
|---|---|
| `src/models/backbones.py` | timm-first backbone wrappers and fallback feature extraction |
| `src/models/vit_b16_classifier.py` | final ViT-B/16 static baseline |
| `src/models/aide_style_detector.py` | final AIDE-style static forensic detector |
| `src/models/vit_heatmap_model.py` | final ViT + heatmap auxiliary model |
| `src/models/seqdet_no_gaze.py` | final sequential detector without human gaze supervision |
| `src/models/diffscanauth.py` | final DiffScanAuth model |
| `src/models/baseline_static.py` | legacy scaffold static baseline |
| `src/models/baseline_heatmap.py` | legacy scaffold heatmap baseline |
| `src/models/seq_gaze_detector.py` | legacy scaffold sequential detector |
| `src/models/__init__.py` | package marker |

#### `src/models/modules/`
| File | Function |
|---|---|
| `src/models/modules/dual_stream_encoder.py` | global semantic stream + local artifact stream wrapper |
| `src/models/modules/diffusion_teacher.py` | working lightweight diffusion-style scanpath teacher |
| `src/models/modules/gaze_student.py` | causal fixation policy student |
| `src/models/modules/foveated_reader.py` | final foveated evidence reader |
| `src/models/modules/accumulator.py` | GRU / Mamba-style evidence accumulator |
| `src/models/modules/stop_head.py` | learned stop head |
| `src/models/modules/losses.py` | final paper-aligned loss bundle |
| `src/models/modules/gaze_policy.py` | legacy policy module kept for compatibility |
| `src/models/modules/glimpse_reader.py` | legacy glimpse reader kept for compatibility |
| `src/models/modules/heads.py` | classification and helper heads |
| `src/models/modules/__init__.py` | package marker |

#### `src/lightning/`
| File | Function |
|---|---|
| `src/lightning/lit_vit.py` | Lightning wrapper for ViT-B/16 |
| `src/lightning/lit_aide.py` | Lightning wrapper for AIDE-style detector |
| `src/lightning/lit_heatmap.py` | Lightning wrapper for heatmap-supervised static model |
| `src/lightning/lit_seqdet.py` | Lightning wrapper for sequential no-gaze baseline |
| `src/lightning/lit_diffscanauth.py` | Lightning wrapper for full DiffScanAuth |
| `src/lightning/lit_seq_base.py` | shared sequential training logic, checkpoint-loading utilities, logging/export |
| `src/lightning/lit_static.py` | legacy static baseline Lightning wrapper |
| `src/lightning/lit_seq.py` | legacy sequential Lightning wrapper |
| `src/lightning/__init__.py` | package marker |

#### `src/evaluation/`
| File | Function |
|---|---|
| `src/evaluation/metrics_classification.py` | classification metrics, calibration metrics, confusion matrix bundle |
| `src/evaluation/metrics_gaze.py` | sequential decision-step, scanpath, and distance metrics |
| `src/evaluation/calibration.py` | ECE and Brier helpers |
| `src/evaluation/bootstrapping.py` | bootstrap confidence intervals |
| `src/evaluation/__init__.py` | package marker |

#### `src/utils/`
| File | Function |
|---|---|
| `src/utils/pipeline.py` | reusable end-to-end builders for metadata, preprocessing, dataloaders, and modules |
| `src/utils/plotting.py` | confusion, ROC, PR, confidence, and scanpath visualization helpers |
| `src/utils/seed.py` | deterministic seeding |
| `src/utils/io.py` | path, CSV, JSON, and directory helpers |
| `src/utils/logging.py` | shared logger helper |
| `src/utils/registry.py` | lightweight registry placeholder |
| `src/utils/__init__.py` | package marker |

#### `tests/`
| File | Function |
|---|---|
| `tests/test_dataset.py` | end-to-end synthetic data pipeline test |
| `tests/test_splits.py` | split leakage test |
| `tests/test_forward.py` | forward tests for static, sequential, and checkpoint-loading paths |
| `tests/test_smoke.py` | smoke-level executable regression test |

### Known Limitations
1. Exact SigLIP2-NaFlex and DINOv2 pretrained weights are exposed through adapters/fallback names, but offline environments may fall back to lighter timm/torchvision backbones.
2. The diffusion teacher is intentionally lightweight and practical rather than a very heavy diffusion transformer.
3. The Mamba path remains optional; GRU is the default stable fallback.
4. Dummy-data metrics are only pipeline validation numbers, not paper numbers.

---

## 中文说明

### 项目概览
本仓库是论文 **DiffScanAuth: Human-Gaze-Supervised Sequential Evidence Accumulation for AI-Generated Image Detection** 的可执行实验代码库。

当前版本不是最初的脚手架，而是已经升级到与论文结构对齐的实现，包含：

1. 论文主方法
2. 论文主 baseline
3. 论文 ablation
4. shuffled-gaze 对照实验
5. 在真实数据尚未上传前可直接运行的 dummy-data 模式
6. 训练、验证、测试、预测、可视化、结果分析全流程
7. 支持 teacher 预训练、student 主训练、可选效率微调的三阶段训练

### 已实现内容

#### 主方法
- `DiffScanAuth`
  - 双流视觉编码器
  - diffusion scanpath teacher
  - causal gaze student
  - foveated evidence reader
  - evidence accumulator
  - learned stop / fixed-K 两种模式

#### 主 baseline
- `exp_vit_b16`
- `exp_aide_style`
- `exp_vit_heatmap`
- `exp_seqdet_no_gaze`
- `exp_diffscanauth`

#### Ablation
- `ablation_no_gaze_supervision`
- `ablation_heatmap_instead_of_scanpath`
- `ablation_no_teacher_distillation`
- `ablation_no_local_stream`
- `ablation_fixed_k`

#### 对照实验
- `control_shuffled_gaze`

### 当前验证状态
截至 **2026 年 3 月 20 日**，本地已经完成：

1. `pytest -q` -> `9 passed`
2. `python scripts/smoke_test.py` -> 通过
3. 以下三个 dummy 短训练已实际跑通：
   - `exp_vit_b16_dummy_short`
   - `exp_seqdet_no_gaze_dummy_short`
   - `exp_diffscanauth_dummy_short`
4. `outputs/` 下已经生成示例 checkpoint、prediction、metrics、tables、figures

### 环境要求

#### 官方目标环境
- Python `3.11`
- PyTorch `2.x`
- PyTorch Lightning
- Hydra / OmegaConf
- timm
- torchvision
- torchmetrics
- scikit-learn
- pandas
- numpy
- pillow / opencv-python
- matplotlib / seaborn
- einops
- tqdm

#### 可选依赖
- transformers
- mamba-ssm
- wandb

#### 本地说明
论文目标环境仍然建议使用 **Python 3.11**。当前这台机器上最近一次回归验证是在已有的 Miniforge Python `3.12` 环境中完成的，也能通过，但正式实验仍建议按论文环境固定到 `3.11`。

### 安装
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 快速开始

#### 完整数据准备流程
```bash
python scripts/inspect_dataset.py
python scripts/build_metadata.py
python scripts/preprocess_eye_tracking.py --force-rebuild-eye-tracking
python scripts/make_splits.py
python scripts/smoke_test.py
```

#### 训练主要实验
```bash
python scripts/train.py experiment=exp_vit_b16
python scripts/train.py experiment=exp_aide_style
python scripts/train.py experiment=exp_vit_heatmap
python scripts/train.py experiment=exp_seqdet_no_gaze
python scripts/train.py experiment=exp_diffscanauth
```

#### 训练 ablation 和 control
```bash
python scripts/train.py experiment=ablation_no_gaze_supervision
python scripts/train.py experiment=ablation_heatmap_instead_of_scanpath
python scripts/train.py experiment=ablation_no_teacher_distillation
python scripts/train.py experiment=ablation_no_local_stream
python scripts/train.py experiment=ablation_fixed_k
python scripts/train.py experiment=control_shuffled_gaze
```

#### 评估 / 预测 / 分析
```bash
python scripts/evaluate.py --experiment exp_diffscanauth --ckpt outputs/checkpoints/exp_diffscanauth/last.ckpt
python scripts/predict.py --experiment exp_diffscanauth --ckpt outputs/checkpoints/exp_diffscanauth/last.ckpt --split test
python scripts/analyze_results.py
python scripts/make_tables.py
python scripts/make_confidence_curve.py --predictions outputs/predictions/exp_diffscanauth_test_predictions.csv
python scripts/make_qualitative_figures.py --ours outputs/predictions/exp_diffscanauth_test_predictions.csv --no-gaze outputs/predictions/exp_seqdet_no_gaze_test_predictions.csv
```

#### 顺序跑完整实验集
```bash
python scripts/run_all_experiments.py --epochs 1
```

### DiffScanAuth 三阶段训练
当前代码已经支持论文里的三阶段训练。

#### 阶段 1：teacher 预训练
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_teacher_stage1 \
  model.training_stage=teacher \
  model.use_teacher=true \
  model.use_teacher_distill=false
```

#### 阶段 2：student + reader + accumulator + classifier
从阶段 1 checkpoint 中加载 encoder + teacher：
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_stage2 \
  model.training_stage=student \
  model.teacher_ckpt=outputs/checkpoints/exp_diffscanauth_teacher_stage1/last.ckpt \
  model.freeze_teacher=true
```

#### 阶段 3：可选效率微调
从阶段 2 加载完整模型：
```bash
python scripts/train.py \
  experiment=exp_diffscanauth \
  experiment.name=exp_diffscanauth_stage3_refine \
  model.training_stage=refine \
  model.init_from_ckpt=outputs/checkpoints/exp_diffscanauth_stage2/last.ckpt \
  model.use_rl=true \
  model.loss_rl=0.05
```

### Dummy 数据模式
即使你现在还没有上传真实数据，这个仓库也能直接运行。

当 `data/raw/` 下没有有效数据时，系统会自动：

1. 生成 synthetic 图像
2. 生成 synthetic metadata
3. 生成 synthetic 眼动轨迹
4. 生成 split 和 heatmap
5. 保证训练、评估、分析脚本都能执行

常用命令：
```bash
python scripts/build_metadata.py --allow-synthetic --synthetic-num-images 120
python scripts/preprocess_eye_tracking.py --allow-synthetic --force-rebuild-eye-tracking
```

### 统一数据 Schema

#### `metadata.csv`
至少需要这些列：
- `image_id`
- `image_path`
- `label`
- `scene`
- `source_type`
- `generator`
- `width`
- `height`
- `split`

#### `eye_tracking.csv`
至少需要这些列：
- `subject_id`
- `image_id`
- `t`
- `x`
- `y`
- `duration`
- `event_type`
- `validity`
- `pupil`

#### `processed_fixations.csv`
至少需要这些列：
- `subject_id`
- `image_id`
- `fixation_idx`
- `x_norm`
- `y_norm`
- `duration_norm`
- `duration_ms`
- `delta_x`
- `delta_y`
- `patch_index`
- `split`

### 防泄漏划分规则
划分单位是 **image_id**，不是被试行级别。

这意味着：

1. 同一张图的所有被试都在同一个 split
2. 不会出现同图跨 train/val/test 泄漏
3. 分层顺序为：
   - `label + scene + generator`
   - `label + scene`
   - `label`
   - 随机回退，并记录日志

输出文件：
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/split_report.json`

### 回退策略
代码优先对齐论文，但也保证在受限环境下可运行。

1. 全局语义流
   - 首选 SigLIP2 风格适配器
   - 回退到 timm transformer / torchvision backbone
2. 局部伪影流
   - 首选 DINOv2 风格 dense adapter
   - 回退到 timm dense backbone / ConvNeXt / ResNet
3. 累积器
   - 若可用，支持 Mamba 风格
   - 默认稳定回退为 GRU
4. Diffusion teacher
   - 仓库内已经实现轻量但真实可训练的 teacher，不是占位符
5. 日志
   - 优先 wandb
   - 否则 TensorBoard
   - 再否则 CSV logger

### 输出目录
- `outputs/logs/`
- `outputs/checkpoints/`
- `outputs/metrics/`
- `outputs/predictions/`
- `outputs/figures/`

### 之后如何切换到真实数据
当你上传真实数据后：

1. 把原始图片和原始表格放到 `data/raw/` 下任意合适位置
2. 重新运行：
   ```bash
   python scripts/inspect_dataset.py
   python scripts/build_metadata.py
   python scripts/preprocess_eye_tracking.py --force-rebuild-eye-tracking
   python scripts/make_splits.py
   ```
3. 检查：
   - `data/processed/metadata.csv`
   - `data/processed/eye_tracking.csv`
   - `data/processed/processed_fixations.csv`
   - `data/splits/split_report.json`
4. 如果原始格式和当前假设不一致，优先修改 `src/datasets/dataset_adapter.py`，然后重跑同一套流程

### 文件逐项说明

#### 根目录文件
| 文件 | 作用 |
|---|---|
| `README.md` | 中英文项目说明与使用手册 |
| `requirements.txt` | 依赖列表 |
| `pyproject.toml` | 项目元信息与 pytest 配置 |
| `.gitignore` | 忽略输出、缓存和本地临时文件 |

#### `configs/`
| 文件 | 作用 |
|---|---|
| `configs/config.yaml` | Hydra 根配置，默认入口指向 `diffscanauth` |
| `configs/data/default.yaml` | 数据路径、预处理、split、loader、synthetic 参数 |
| `configs/trainer/default.yaml` | Lightning 训练器默认参数 |
| `configs/model/vit_b16.yaml` | 最终版 ViT-B/16 baseline 配置 |
| `configs/model/aide_style.yaml` | 最终版 AIDE-style baseline 配置 |
| `configs/model/vit_gaze_heatmap.yaml` | 最终版热图监督静态模型配置 |
| `configs/model/seqdet_no_gaze.yaml` | 最终版无人类 gaze 顺序模型配置 |
| `configs/model/diffscanauth.yaml` | 最终版 DiffScanAuth 配置 |
| `configs/model/baseline_static.yaml` | 为兼容旧脚手架保留的老配置 |
| `configs/model/baseline_heatmap.yaml` | 为兼容旧脚手架保留的老配置 |
| `configs/model/seq_gaze_detector.yaml` | 为兼容旧脚手架保留的老配置 |
| `configs/experiment/exp_vit_b16.yaml` | ViT-B/16 实验预设 |
| `configs/experiment/exp_aide_style.yaml` | AIDE-style 实验预设 |
| `configs/experiment/exp_vit_heatmap.yaml` | ViT + gaze heatmap 实验预设 |
| `configs/experiment/exp_seqdet_no_gaze.yaml` | 无 gaze 顺序基线实验预设 |
| `configs/experiment/exp_diffscanauth.yaml` | DiffScanAuth 主实验预设 |
| `configs/experiment/ablation_no_gaze_supervision.yaml` | 去掉 gaze supervision 的 ablation |
| `configs/experiment/ablation_heatmap_instead_of_scanpath.yaml` | 用 heatmap 替代 scanpath 的 ablation |
| `configs/experiment/ablation_no_teacher_distillation.yaml` | 去掉 teacher distillation 的 ablation |
| `configs/experiment/ablation_no_local_stream.yaml` | 去掉 local artifact stream 的 ablation |
| `configs/experiment/ablation_fixed_k.yaml` | fixed-K 替代 learned stop 的 ablation |
| `configs/experiment/control_shuffled_gaze.yaml` | shuffled gaze 对照实验 |
| `configs/experiment/baseline_static.yaml` | 兼容旧版本的老实验预设 |
| `configs/experiment/baseline_heatmap.yaml` | 兼容旧版本的老实验预设 |
| `configs/experiment/seq_gaze_detector.yaml` | 兼容旧版本的 fixed-K 预设 |
| `configs/experiment/seq_gaze_stop.yaml` | 兼容旧版本的 learned-stop 预设 |

#### `scripts/`
| 文件 | 作用 |
|---|---|
| `scripts/inspect_dataset.py` | 扫描 `data/raw/` 并输出结构检查报告 |
| `scripts/build_metadata.py` | 从原始图像或 synthetic fallback 构建 `metadata.csv` |
| `scripts/preprocess_eye_tracking.py` | 标准化眼动数据并生成 `processed_fixations.csv` |
| `scripts/make_splits.py` | 生成防泄漏 train/val/test 划分并写出 heatmap |
| `scripts/train.py` | Hydra 训练入口；训练结束后自动测试并导出结果 |
| `scripts/evaluate.py` | 加载 checkpoint 在指定 split 上评估 |
| `scripts/predict.py` | 导出指定 checkpoint 的预测 CSV |
| `scripts/analyze_results.py` | 聚合 prediction CSV，计算指标并出图 |
| `scripts/make_tables.py` | 从 metrics JSON 生成表格 |
| `scripts/make_qualitative_figures.py` | 生成 Human vs Ours vs No-Gaze 的定性图 |
| `scripts/make_confidence_curve.py` | 从顺序模型预测结果生成 confidence-over-time 曲线 |
| `scripts/run_all_experiments.py` | 顺序执行主实验、ablation、control |
| `scripts/smoke_test.py` | 对最终模型家族做快速可执行回归测试 |

#### `src/`
| 文件 | 作用 |
|---|---|
| `src/__init__.py` | Python 包标记 |

#### `src/datasets/`
| 文件 | 作用 |
|---|---|
| `src/datasets/dataset_schema.py` | 数据 schema 与简单类型定义 |
| `src/datasets/dataset_adapter.py` | 原始数据扫描、metadata 构建、synthetic 数据生成、眼动适配 |
| `src/datasets/image_dataset.py` | 静态图像模型的数据集 |
| `src/datasets/gaze_dataset.py` | 顺序 gaze 数据集，支持 shuffled gaze control |
| `src/datasets/collate.py` | static / sequential 两类 batch 的拼接逻辑 |
| `src/datasets/transforms.py` | 不破坏 gaze 对齐关系的图像变换 |
| `src/datasets/split_utils.py` | image-level 分层划分与泄漏检查 |

#### `src/features/`
| 文件 | 作用 |
|---|---|
| `src/features/gaze_processing.py` | 归一化、fixation 化、duration 标准化、delta 计算 |
| `src/features/heatmap.py` | fixation heatmap 生成与存储 |
| `src/features/fixation_tokenizer.py` | fixation 到 patch token 的离散化与统计 |

#### `src/models/`
| 文件 | 作用 |
|---|---|
| `src/models/backbones.py` | timm 优先、torchvision 回退的 backbone 包装器 |
| `src/models/vit_b16_classifier.py` | 最终版 ViT-B/16 baseline |
| `src/models/aide_style_detector.py` | 最终版 AIDE-style 静态鉴伪器 |
| `src/models/vit_heatmap_model.py` | 最终版热图监督静态模型 |
| `src/models/seqdet_no_gaze.py` | 最终版无人类 gaze 顺序模型 |
| `src/models/diffscanauth.py` | 最终版 DiffScanAuth 主模型 |
| `src/models/baseline_static.py` | 旧脚手架静态 baseline |
| `src/models/baseline_heatmap.py` | 旧脚手架热图 baseline |
| `src/models/seq_gaze_detector.py` | 旧脚手架顺序模型 |
| `src/models/__init__.py` | 包标记 |

#### `src/models/modules/`
| 文件 | 作用 |
|---|---|
| `src/models/modules/dual_stream_encoder.py` | 全局语义流 + 局部伪影流封装 |
| `src/models/modules/diffusion_teacher.py` | 轻量可训练 diffusion 风格 scanpath teacher |
| `src/models/modules/gaze_student.py` | 因果 gaze policy student |
| `src/models/modules/foveated_reader.py` | 最终版 foveated evidence reader |
| `src/models/modules/accumulator.py` | GRU / Mamba 风格证据累积器 |
| `src/models/modules/stop_head.py` | learned stop head |
| `src/models/modules/losses.py` | 论文最终损失组合 |
| `src/models/modules/gaze_policy.py` | 旧版本 policy 模块，保留兼容性 |
| `src/models/modules/glimpse_reader.py` | 旧版本 glimpse reader，保留兼容性 |
| `src/models/modules/heads.py` | 分类头和辅助头 |
| `src/models/modules/__init__.py` | 包标记 |

#### `src/lightning/`
| 文件 | 作用 |
|---|---|
| `src/lightning/lit_vit.py` | ViT-B/16 的 Lightning 包装 |
| `src/lightning/lit_aide.py` | AIDE-style 的 Lightning 包装 |
| `src/lightning/lit_heatmap.py` | 热图监督静态模型的 Lightning 包装 |
| `src/lightning/lit_seqdet.py` | 无 gaze 顺序模型的 Lightning 包装 |
| `src/lightning/lit_diffscanauth.py` | DiffScanAuth 的 Lightning 包装 |
| `src/lightning/lit_seq_base.py` | 顺序模型共享训练逻辑、checkpoint 注入、日志导出 |
| `src/lightning/lit_static.py` | 旧脚手架静态 Lightning 包装 |
| `src/lightning/lit_seq.py` | 旧脚手架顺序 Lightning 包装 |
| `src/lightning/__init__.py` | 包标记 |

#### `src/evaluation/`
| 文件 | 作用 |
|---|---|
| `src/evaluation/metrics_classification.py` | 分类指标、校准指标、混淆矩阵打包 |
| `src/evaluation/metrics_gaze.py` | 决策步数、scanpath、距离类指标 |
| `src/evaluation/calibration.py` | ECE 和 Brier 的辅助函数 |
| `src/evaluation/bootstrapping.py` | bootstrap 置信区间 |
| `src/evaluation/__init__.py` | 包标记 |

#### `src/utils/`
| 文件 | 作用 |
|---|---|
| `src/utils/pipeline.py` | 数据准备、dataloader、module 构建等全流程辅助 |
| `src/utils/plotting.py` | confusion/ROC/PR/confidence/scanpath 的绘图函数 |
| `src/utils/seed.py` | 随机种子固定 |
| `src/utils/io.py` | 路径、CSV、JSON、目录辅助 |
| `src/utils/logging.py` | 统一 logger 辅助 |
| `src/utils/registry.py` | 轻量 registry 占位 |
| `src/utils/__init__.py` | 包标记 |

#### `tests/`
| 文件 | 作用 |
|---|---|
| `tests/test_dataset.py` | synthetic 数据流程测试 |
| `tests/test_splits.py` | split 泄漏检查测试 |
| `tests/test_forward.py` | static / sequential / teacher checkpoint 路径测试 |
| `tests/test_smoke.py` | 可执行 smoke 回归测试 |

### 已知限制
1. SigLIP2-NaFlex 和 DINOv2 的“精确官方权重”在离线环境下不一定可直接下载，因此会回退到轻量 timm/torchvision backbone。
2. 当前 diffusion teacher 是为可运行性和工程稳定性设计的轻量实现，不是超大规模 diffusion transformer。
3. Mamba 路径是可选增强项，默认稳定后端仍是 GRU。
4. dummy 数据上的结果只是验证 pipeline，可运行不代表论文最终指标。
