# 多模态网络钓鱼智能识别系统（工程仓库）

> **说明**：若你本地的 `README.md` 曾变为空白，多为误操作或未保存覆盖；本仓库当前无 Git 历史时可从此文件重新复制备份。

面向中文钓鱼场景的**多源信息处理 + BERT-TextCNN 核心判别**系统：FastAPI 后端、Vue 3 前端、PyTorch 模型与对比实验脚本。论文题目可保留「多模态」表述，正文技术路线以「多源信息 + 混合核心模型」为准（详见 `docs/reproducibility_and_feasibility.md`）。

---

## 快速启动

**后端（推荐入口）**

```bash
cd src
python -m uvicorn service_layer.app:app --host 127.0.0.0 --port 8000
```

**前端**

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 Vite 提示的本地地址（一般为 `http://127.0.0.1:5173`）。侧栏 **「演示」→「功能演示」** 可开启界面引导与示例检测流程。

API 交互文档：`http://127.0.0.1:8000/docs`

---

## 目录结构（摘要）

| 路径 | 作用 |
|------|------|
| `src/service_layer/` | FastAPI 应用、路由与依赖注入 |
| `src/inference_layer/` | 推理服务、模型管理、资源监控 |
| `src/core/models/model_definitions.py` | BERT-TextCNN（增强版）与旧权重兼容结构 `BERTTextCNNLegacy` |
| `src/data_layer/` | SQLite 与仓储 |
| `frontend/src/` | Vue 3 + Element Plus 前端 |
| `scripts/experiment/` | 对比实验编排（`run_experiment.py`、`models.py`） |
| `models/` | 部署用权重（如 `bert_textcnn_best.pth`）与 `model_versions.json` |
| `output/experiments/` | 实验输出、`summary.json`、会话状态等 |
| `docs/reproducibility_and_feasibility.md` | 可复现命令、研究边界与可行性 |
| `# 论文主要内容.ini` | 论文章节要点与写作提示 |

---

## 模型与权重

- 生产推理默认加载 `models/bert_textcnn_best.pth`。
- 若权重为**早期结构**（仅 CNN 池化 + `fc` 192 维），加载时会自动使用 `BERTTextCNNLegacy`；与当前增强版（`[CLS]` + CNN 融合、`fc` 256 维）一致的新权重需自行训练后覆盖，以免论文中的「增强结构」与部署文件不一致。

---

## 实验复现（摘要）

增强版 BERT-TextCNN 主结果示例路径：

`output/experiments/formal_20260501_rescued_hybrid/summary.json`

基线汇总示例路径：

`output/experiments/formal_20260425_merged_feasible/summary.json`

复现命令模板（Windows 可改用 `^` 换行）：

```bash
python scripts/experiment/run_experiment.py ^
  --session-dir output/experiments/<你的会话目录> ^
  --models BERT-TextCNN ^
  --seeds 42,123,456 ^
  --training-profile balanced
```

具体参数与结果引用以 `docs/reproducibility_and_feasibility.md` 为准。

---

## 论文写作参考索引

| 写作板块 | 建议对照文件 |
|----------|----------------|
| 摘要、研究背景与意义 | `# 论文主要内容.ini`、`docs/reproducibility_and_feasibility.md` |
| 系统总体设计 / 多源信息 | `src/inference_layer/inference_service.py`、`README` 本节前表 |
| 核心模型 BERT-TextCNN | `src/core/models/model_definitions.py`、`scripts/experiment/models.py` |
| 实验设计与结果 | `output/experiments/**/summary.json`、`output/experiments/formal_20260501_rescued_hybrid/rescued_hybrid_conclusion.md` |
| 局限与消融（多模态） | 同上 `summary.json` 中多模态相关条目、`docs/reproducibility_and_feasibility.md` |
| 前端与演示 | `frontend/src/views/DemoCenter.vue`、`frontend/src/demo/` |

---

## 可行性与边界（一句话）

在线探测用户 URL 默认关闭；管理类接口请在可信环境使用；完整结论依赖固定数据划分与随机种子，见 `docs/reproducibility_and_feasibility.md`。

---

## Git 版本管理

仓库已初始化（默认分支 `main`），根目录 `.gitignore` 会排除：`node_modules`、`venv`、日志、`*.pth` / `*.pt`、`data/**/*.csv`、本地数据库与 `src/data/raw/chifraud/`（嵌套仓库）等大文件；**实验 JSON、源码与 `models/model_versions.json` 仍被跟踪**。

常用命令：

```bash
git status
git add -A
git commit -m "描述本次修改"
```

首次在本机推送到远程前，请将提交者改为你自己的信息（当前可能为占位配置）：

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
```

然后关联远程并推送：

```bash
git remote add origin <你的仓库 HTTPS 或 SSH 地址>
git push -u origin main
```
