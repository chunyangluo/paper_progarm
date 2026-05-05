# 基于 BERT-TextCNN 混合模型的多模态网络钓鱼攻击智能识别系统

> 本 README 面向论文写作、项目答辩与代码评审，重点说明研究主线、系统实现、实验结果、可复现方式与工程边界。

## 1. 项目定位

本项目围绕中文网络钓鱼攻击识别任务，构建一个支持文本、URL 与网络行为等多源信息处理的智能识别系统。系统题目保留“多模态”表述，但当前可复现、可部署的技术主线为：

1. 系统层面支持多源信息输入、解析、记录、预警与可视化展示；
2. 核心分类器采用增强版 BERT-TextCNN 混合模型；
3. URL 与网络行为特征用于辅助分析、记录、预警和消融实验；
4. 多模态神经融合模型作为对照实验和局限分析，不作为当前生产推理模型。

该表述与实验结果、代码实现和论文提纲保持一致，可避免将“多模态信息处理系统”误写为“多模态融合模型一定优于文本模型”。

## 2. 研究贡献

- **多源信息处理流程**：面向短信、邮件、URL 链接等钓鱼场景，整理文本语义、URL 结构和网络行为相关信息，用于检测、解释、记录和预警。
- **增强版 BERT-TextCNN 模型**：将 BERT 的 `[CLS]` 全局语义表示与 TextCNN 的局部 n-gram 特征进行融合，兼顾上下文语义理解和局部钓鱼话术捕获。
- **完整智能识别系统**：实现前端检测交互、批量检测、预警中心、模型管理、性能展示、功能演示和后端 API 服务。
- **可复现实验验证**：提供固定划分、三随机种子、checkpoint 续跑和结果汇总文件，便于论文复核与评审阅读。
- **工程化风险控制**：默认关闭在线网络探测，避免对用户输入 URL 发起不受控访问；规则兜底和缓存机制提升系统稳定性。

## 3. 系统总体架构

```text
用户输入（文本 / URL / CSV）
  -> Vue 3 + Element Plus 前端
  -> FastAPI 服务层
  -> InferenceService 推理服务
  -> BERT-TextCNN 核心识别模型
  -> URL / network 多源信息解析
  -> SQLite 检测记录、预警记录、模型版本记录
  -> 前端结果展示、预警中心、性能数据与功能演示
```

### 3.1 主要模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 前端界面 | `frontend/src/` | Vue 3 + Element Plus，包含首页、单样本检测、批量检测、预警中心、模型管理、性能数据、技术亮点和功能演示 |
| API 服务 | `src/service_layer/` | FastAPI 应用、路由、依赖注入和异常处理 |
| 推理服务 | `src/inference_layer/inference_service.py` | 生产推理主入口，统一使用 BERT-TextCNN，并附带多源信息分析结果 |
| 模型定义 | `src/core/models/model_definitions.py` | 增强版 BERT-TextCNN、旧权重兼容结构和辅助模块 |
| 数据存储 | `src/data_layer/` | SQLAlchemy 模型、数据库连接、检测记录与预警仓储 |
| 实验脚本 | `scripts/experiment/` | 对比实验、训练编排、checkpoint 和结果汇总 |
| 实验结果 | `output/experiments/` | 正式实验、消融实验、summary、checkpoint 和 worker 结果 |
| 论文提纲 | `docs/thesis/论文主要内容.ini` | 论文各章节写作主线、创新点、结果引用方式 |
| 可复现说明 | `docs/reproducibility_and_feasibility.md` | 复现实验、系统运行、可行性和风险边界 |

## 4. 核心模型设计

### 4.1 BERT-TextCNN 混合结构

增强版 BERT-TextCNN 的设计目标是充分利用两类特征：

- BERT 分支通过 `[CLS]` 表示建模句级全局语义；
- TextCNN 分支在 BERT token 序列上使用 2/3/4-gram 卷积核捕获局部短语模式；
- 两路特征分别投影后拼接，进入 dropout 和全连接分类层。

该结构可在强 BERT 基线已经接近性能天花板时，进一步关注召回率和漏报率，符合网络钓鱼检测任务中“降低漏报风险”的实际需求。

### 4.2 部署权重兼容性

生产推理默认加载：

```text
models/bert_textcnn_best.pth
```

当前代码同时支持两类权重：

- 增强版结构：`[CLS] + CNN` 融合，分类层输入维度为 256；
- 旧版结构：仅 CNN 池化，分类层输入维度为 192，通过 `BERTTextCNNLegacy` 兼容加载。

论文中建议以增强版结构和 `formal_20260501_rescued_hybrid` 实验结果为主；若部署目录仍使用旧结构权重，应在答辩或部署说明中标明“工程兼容加载”，并尽量用增强版权重替换生产文件。

## 5. 实验设计与结果分析

### 5.1 实验设置

- 数据划分：固定训练 / 验证 / 测试划分；
- 随机种子：`42`、`123`、`456`；
- 训练配置：`balanced`；
- 评价指标：Accuracy、Precision、Recall、F1、FPR、FNR、AUC；
- 重点指标：F1 和 FNR。其中 FNR 表示漏报率，在钓鱼检测中对应未被拦截的攻击样本。

### 5.2 正式结果汇总

主模型结果来自：

```text
output/experiments/formal_20260501_rescued_hybrid/summary.json
```

基线结果来自：

```text
output/experiments/formal_20260425_merged_feasible/summary.json
```

| 模型 | Accuracy | Precision | Recall | F1 | FPR | FNR | AUC |
|------|---------:|----------:|-------:|---:|----:|----:|----:|
| TF-IDF + LR | 0.9181 | 0.9709 | 0.8620 | 0.9132 | 0.0258 | 0.1380 | 0.9592 |
| TextCNN | 0.9929 | 0.9935 | 0.9923 | 0.9929 | 0.0065 | 0.0077 | 0.9992 |
| LSTM | 0.9898 | 0.9906 | 0.9890 | 0.9898 | 0.0093 | 0.0110 | 0.9980 |
| BERT | 0.9937 | 0.9951 | 0.9924 | 0.9937 | 0.0049 | 0.0076 | 0.9989 |
| 增强版 BERT-TextCNN | **0.9938** | 0.9944 | **0.9932** | **0.9938** | 0.0056 | **0.0068** | 0.9989 |

增强版 BERT-TextCNN 三随机种子正式结果：

| 指标 | 均值 | 标准差 |
|------|-----:|-------:|
| Accuracy | 0.9938 | 0.0003 |
| Precision | 0.9944 | 0.0006 |
| Recall | 0.9932 | 0.0008 |
| F1 | 0.9938 | 0.0003 |
| FPR | 0.0056 | 0.0006 |
| FNR | 0.0068 | 0.0008 |
| AUC | 0.9989 | 0.0002 |

### 5.3 结果解读

实验表明，增强版 BERT-TextCNN 在固定划分和三随机种子设置下取得最高平均 F1，并在主要文本模型中获得最低 FNR。与 BERT 基线相比，绝对提升幅度较小：

- F1：`0.993720 -> 0.993768`；
- Recall：`0.992364 -> 0.993175`；
- FNR：`0.007636 -> 0.006825`。

该提升幅度不宜夸大，因为 BERT 在该数据集上已经接近性能天花板。论文中更稳妥、也更符合实验事实的表述是：增强版 BERT-TextCNN 在强 BERT 基线基础上实现了小幅但方向明确的召回率提升和漏报率降低，而漏报率降低对于网络钓鱼检测具有实际安全价值。

### 5.4 多模态消融结论

早期多模态实验结果显示，`Multimodal-text_only` 和 `Multimodal-text_url` 未能稳定优于文本主模型，部分设置出现单类预测或性能塌缩。该现象说明：

- 手工 URL 特征和网络行为特征质量、覆盖率仍不足；
- 多模态融合对特征分布、训练稳定性和优化策略更敏感；
- 当前阶段更适合将多源信息用于系统解析、记录、预警辅助和消融分析。

因此，论文中不建议将多模态融合模型写为核心贡献。更建议表述为：系统具备多模态 / 多源信息处理能力，核心判别模型采用表现稳定的 BERT-TextCNN；多模态深度融合作为局限与后续工作。

## 6. 系统功能与演示

### 6.1 已实现功能

- 单样本检测：支持文本输入、URL 输入、文本 + 关联 URL；
- 批量检测：支持含 `text` 或 `url` 列的 CSV 文件；
- 多源信息分析：返回 URL 特征、网络行为特征和 feature summary；
- 预警中心：高风险结果可生成预警记录并支持查看和处理；
- 模型管理：展示 BERT-TextCNN 版本、已加载模型和运行设备；
- 性能数据：展示检测统计、趋势和模型版本指标；
- 功能演示：提供全流程 Tour、分模块引导、单样本示例和批量示例。

### 6.2 推荐演示路径

1. 启动后端和前端；
2. 打开首页进入“功能演示”；
3. 运行“开始全流程引导”；
4. 运行“单样本示例 + 引导”，观察检测结果和多源信息分析；
5. 运行“示例批量检测”，观察 CSV-like 批量结果；
6. 打开预警中心、模型管理和性能数据页面，展示系统闭环。

## 7. 运行与复现

### 7.1 后端启动

```bash
cd src
python -m uvicorn service_layer.app:app --host 127.0.0.1 --port 8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

### 7.2 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端默认访问地址通常为：

```text
http://127.0.0.1:5173
```

### 7.3 实验复现

增强版 BERT-TextCNN 复现命令：

```bash
python scripts/experiment/run_experiment.py ^
  --session-dir output/experiments/formal_20260501_rescued_hybrid_recheck ^
  --models BERT-TextCNN ^
  --seeds 42,123,456 ^
  --training-profile balanced
```

实验完成后查看：

```text
output/experiments/formal_20260501_rescued_hybrid_recheck/summary.json
```

## 8. 代码规范与最终检查

### 8.1 当前规范化处理

- 生产入口统一为 `src/service_layer/app.py`；
- 推理主链路统一为 `src/inference_layer/inference_service.py`；
- 生产推理不再使用多模态神经融合模型；
- `ModelTypeEnum` 保留 BERT-TextCNN，避免前后端模型类型漂移；
- 前端移除“多模态融合模型”误导性选择项；
- README、论文提纲与实验结论统一到“多源信息处理 + BERT-TextCNN 核心识别”的表述；
- `.gitignore` 已排除 `*.pth`、`*.pt`、CSV 数据集、日志、数据库、`node_modules` 和本地缓存。

### 8.2 建议提交前检查

后端语法检查：

```bash
python -m py_compile ^
  src/inference_layer/inference_service.py ^
  src/service_layer/schemas.py ^
  src/service_layer/api/detection.py ^
  src/data_layer/repository.py
```

前端构建检查：

```bash
cd frontend
npm run build
```

Git 状态检查：

```bash
git status
```

## 9. 论文写作索引

| 论文部分 | 推荐引用材料 |
|----------|--------------|
| 写作过程与结构 | `docs/thesis/论文写作指导建议.md` |
| 摘要与创新点 | `docs/thesis/论文主要内容.ini`、本 README 第 1-2 节 |
| 相关理论 | `docs/thesis/论文主要内容.ini` 第二章、`src/core/models/model_definitions.py` |
| 数据集构建 | `docs/reports/dataset_building_guide.md`、`data/数据集收集报告.md`、`docs/reproducibility_and_feasibility.md` |
| 模型设计 | `src/core/models/model_definitions.py`、`scripts/experiment/models.py` |
| 系统实现 | `src/service_layer/`、`src/inference_layer/`、`frontend/src/views/` |
| 实验设置 | `scripts/experiment/run_experiment.py`、`scripts/experiment/data_split.py` |
| 实验结果 | `output/experiments/formal_20260501_rescued_hybrid/summary.json`、`output/experiments/formal_20260425_merged_feasible/summary.json` |
| 结果分析 | `output/experiments/formal_20260501_rescued_hybrid/rescued_hybrid_conclusion.md` |
| 可复现性 | `docs/reproducibility_and_feasibility.md` |
| 系统演示 | `frontend/src/views/DemoCenter.vue`、`frontend/src/demo/` |

## 10. 风险边界与后续优化

### 10.1 已知边界

- 当前系统未实现完整用户认证和权限控制，备份、模型管理等接口应仅在可信环境开放；
- 生产系统默认关闭在线网络探测，避免 SSRF 风险；
- 多模态神经融合模型尚未稳定优于文本主模型，不应作为论文主结论；
- `*.pth` 权重和大量 CSV 数据未纳入 Git，迁移环境时需单独备份；
- 旧权重兼容逻辑用于工程可用性，论文主模型应以增强版结构为准。

### 10.2 后续优化方向

- 用增强版 BERT-TextCNN 重新导出生产权重，替换旧结构 `bert_textcnn_best.pth`；
- 引入更高质量、更稳定覆盖的 URL 与网络行为特征；
- 探索门控融合、置信度校准或跨模态注意力等更稳健的融合策略；
- 增加用户认证、角色权限、审计日志和部署监控；
- 建立 CI 流程，自动运行后端语法检查、前端构建和关键 API smoke test；
- 为论文结果图表生成统一脚本，减少手工整理误差。

## 11. Git 与版本管理

远程仓库：

```text
https://github.com/chunyangluo/paper_progarm
```

常用命令：

```bash
git status
git add -A
git commit -m "描述本次修改"
git push
```

注意：模型权重和大规模 CSV 数据被 `.gitignore` 忽略，克隆仓库后需要从本地备份或网盘恢复 `models/` 中的权重文件。

## 12. 推荐论文结论表述

本文构建了面向中文网络钓鱼场景的多模态智能识别系统，系统能够处理文本、URL 与网络行为等多源信息，并将其用于输入解析、特征记录、预警辅助和可视化展示。在核心识别模型方面，本文采用增强版 BERT-TextCNN，将 BERT 的全局语义表示与 TextCNN 的局部 n-gram 特征进行融合。实验结果表明，在固定数据划分和三随机种子设置下，该模型取得较高的检测性能，并相较强 BERT 基线进一步降低漏报率。多模态融合实验未取得稳定增益，说明多源特征质量和融合稳定性仍是后续研究的重要方向。整体而言，本系统在模型效果、工程实现、可复现性和实际应用价值方面具备较好的完整性与可落地性。
