# 可复现性与可行性说明

## 研究主线

本项目题目保留“多模态网络钓鱼攻击智能识别系统”的表述，但技术路线明确为：

1. 系统处理文本、URL、网络行为等多源信息；
2. 核心分类器采用增强版 BERT-TextCNN；
3. URL 与网络行为信息用于输入解析、记录、预警辅助和消融分析；
4. 多模态神经融合模型作为实验对照和局限分析，不作为当前部署模型。

## 生产系统入口

推荐使用以下入口运行系统：

```bash
cd src
python -m uvicorn service_layer.app:app --host 0.0.0.0 --port 8000
```

前端入口：

```bash
cd frontend
npm install
npm run dev
```

生产推理主链路：

```text
Vue前端输入
  -> FastAPI detection API
  -> InferenceService
  -> BERT-TextCNN核心分类
  -> URL/network多源信息解析
  -> 检测记录/预警/前端展示
```

`src/core/inference.py` 与 `src/core/inference/inference_engine.py` 保留为历史兼容或脚本用途，不作为推荐生产入口。

## 实验复现

### 基线对比

基线模型结果来自：

```text
output/experiments/formal_20260425_merged_feasible/summary.json
```

包含 TF-IDF+LR、TextCNN、BERT、LSTM，以及早期 BERT-TextCNN 和多模态消融结果。

### 增强版 BERT-TextCNN

最终主模型结果来自：

```text
output/experiments/formal_20260501_rescued_hybrid/summary.json
```

对应设置：

- 固定数据划分；
- 训练配置：`balanced`；
- 随机种子：`42,123,456`；
- 模型：增强版 BERT-TextCNN（`[CLS]` 全局语义 + CNN 局部 n-gram）。

复现实验命令：

```bash
python scripts/experiment/run_experiment.py ^
  --session-dir output/experiments/formal_20260501_rescued_hybrid_recheck ^
  --models BERT-TextCNN ^
  --seeds 42,123,456 ^
  --training-profile balanced
```

## 可引用结果

增强版 BERT-TextCNN 正式结果：

- Accuracy: `0.9938 ± 0.0003`
- Precision: `0.9944 ± 0.0006`
- Recall: `0.9932 ± 0.0008`
- F1: `0.9938 ± 0.0003`
- FNR: `0.0068 ± 0.0008`
- AUC: `0.9989 ± 0.0002`

论文中建议写法：

> 在固定数据划分和三随机种子实验设置下，增强版 BERT-TextCNN 取得最高平均 F1，并相较 BERT 基线进一步降低漏报率。由于强 BERT 基线已经接近性能天花板，提升幅度较小，但其对漏报率的改善具有网络钓鱼检测场景下的实际价值。

## 多模态消融边界

多模态融合模型在当前数据质量和特征覆盖条件下未取得稳定增益。论文中应将其定位为：

- 消融实验；
- 局限分析；
- 后续研究方向。

不建议将其写成“本文核心模型”或“性能提升主要来源”。

## 落地可行性

### 技术可行

- BERT-TextCNN 已完成三 seed 正式实验；
- 前后端链路可运行；
- 检测记录、预警、模型管理和批量检测功能已实现；
- URL 与网络行为信息可被解析和记录。

### 工程可行

- 后端采用 FastAPI；
- 前端采用 Vue 3 + Element Plus；
- 数据层使用 SQLAlchemy；
- 推理服务有缓存和规则兜底；
- 默认关闭在线网络探测，降低 SSRF 风险。

### 可复现

- 实验脚本支持 checkpoint 续跑；
- 正式结果保存在 `output/experiments/`；
- 前端构建命令固定为 `npm run build`；
- 后端可通过 `py_compile` 和接口 smoke test 做基础验证。

## 验证清单

每次提交前建议执行：

```bash
python -m py_compile src/inference_layer/inference_service.py src/service_layer/schemas.py
```

```bash
cd frontend
npm run build
```

建议人工 smoke test：

1. 单样本文本检测；
2. 单样本 URL-only 检测；
3. 批量 CSV：仅 `url` 列；
4. 批量 CSV：`text,url,scenario` 三列；
5. 模型管理页能显示 BERT-TextCNN；
6. 性能页能显示模型版本指标和趋势图。

## 风险与边界

- 当前系统无完整鉴权模块，备份和模型管理接口应仅在可信环境中开放；
- 规则校准层会在高风险关键词场景下修正模型输出，论文中应称为“工程化后处理/风险校准”，不要写成纯神经网络输出；
- 多模态神经融合仍需更高质量 URL/网络行为数据和更稳定训练策略后再作为主模型。

