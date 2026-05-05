# 夜间网络行为特征采集功能说明

本文档说明如何使用“分片采集 + 自动续跑 + 合并”功能，为 10 万级数据集持续补齐 `network_features`，用于后续真正多模态实验。

---

## 1. 功能目标

该功能用于解决大数据集一次性在线采集耗时长、易中断的问题，提供：

- 分片采集（按 chunk 处理）
- 自动记录进度并续跑
- 每晚仅跑固定分片数（适合定时任务）
- 全部分片完成后合并为完整数据集

核心脚本：

- `scripts/nightly_network_feature_pipeline.py`

配套 bat：

- `scripts/run_nightly_network_collection.bat`（夜间采集）
- `scripts/merge_nightly_network_collection.bat`（合并结果）
- `scripts/run_nightly_collection_with_dashboard.bat`（启动监控页并触发采集）

---

## 2. 目录与产物

默认运行目录（按 `run-name`）：

- `data/nightly_network_runs/<run-name>/`

关键文件：

- `state.json`：续跑状态（已完成分片、下次起点）
- `chunks/chunk_*.csv`：每个分片的采集结果
- `latest_report.json`：最新进度/覆盖率报告
- `<dataset>_with_network_features_merged.csv`：最终合并产物

---

## 3. 快速开始

### 3.1 每晚采集（推荐）

直接运行：

```bat
scripts\run_nightly_network_collection.bat
```

默认配置（可在 bat 内改）：

- 输入集：`data/versions/dataset_20260421_100k.csv`
- 分片大小：`20000`
- 每次执行分片数：`2`
- 并发批次：`100`

日志输出：

- `logs/nightly_network_collection_yyyyMMdd_HHmmss.log`

### 3.2 合并结果

当分片全部完成后执行：

```bat
scripts\merge_nightly_network_collection.bat
```

日志输出：

- `logs/nightly_network_merge_yyyyMMdd_HHmmss.log`

---

## 4. 命令行手动运行（可选）

### 4.1 采集模式

```bash
python scripts/nightly_network_feature_pipeline.py \
  --input "data/versions/dataset_20260421_100k.csv" \
  --run-name "dataset_20260421_100k_nightly" \
  --chunk-size 20000 \
  --chunks-per-run 2 \
  --batch-size 100 \
  --no-auto-merge
```

### 4.2 仅合并

```bash
python scripts/nightly_network_feature_pipeline.py \
  --input "data/versions/dataset_20260421_100k.csv" \
  --run-name "dataset_20260421_100k_nightly" \
  --merge-only
```

---

## 5. 续跑机制说明

每次采集后会更新 `state.json`：

- `completed_chunks`：已完成分片列表
- `next_chunk_idx`：下次起始分片
- `total_chunks`：总分片数

再次执行同一 `run-name` 时，脚本会自动跳过已完成分片并继续。

---

## 6. 与多模态实验接入

1. 等合并产物生成：
   - `data/nightly_network_runs/<run-name>/<dataset>_with_network_features_merged.csv`
2. 用该文件重建实验 split（`train/val/test`）。
3. 运行多模态消融实验。
4. 训练前覆盖率检查会自动验证 `network_features` 质量（低于阈值会报错）。

---

## 7. 常见问题

### Q1: Redis 未连接怎么办？

可忽略。脚本会回退到本地缓存，不影响采集结果。

### Q2: 采集很慢或超时较多怎么办？

- 降低 `batch-size`（例如 100 -> 50）
- 减少 `chunks-per-run`，改为多晚执行
- 保持稳定网络环境，优先夜间运行

### Q3: 中途关闭窗口会丢进度吗？

不会。已完成分片会记录在 `state.json`，下次可续跑。

### Q4: 如何判断是否已经采集完成？

查看 `latest_report.json`：

- `remaining_chunks` 为 `0` 即完成

---

## 8. 推荐执行策略（10 万级）

- 每晚跑 1~2 个分片
- 每周合并一次并统计覆盖率
- 覆盖率达到实验阈值后重建 split 并跑正式实验

这样能在可控时间内持续提升 `network_features` 覆盖率，逐步达到大规模多模态实验要求。

---

## 9. Windows 计划任务自动运行

### 9.1 图形界面方式（推荐）

1. 打开“任务计划程序” -> “创建基本任务”  
2. 名称填写：`NightlyNetworkCollection`  
3. 触发器选择“每天”，时间例如 `23:30`  
4. 操作选择“启动程序”  
5. 程序或脚本填写：

```text
cmd.exe
```

6. 添加参数填写（推荐用“监控+触发采集”入口）：

```text
/c "c:\Users\chuny\Desktop\paper_progarm\scripts\run_nightly_collection_with_dashboard.bat"
```

7. “起始于”填写（可选但建议）：

```text
c:\Users\chuny\Desktop\paper_progarm
```

8. 完成后可右键任务 -> “运行”做一次手动验证。

### 9.2 命令行方式（schtasks）

以管理员 PowerShell / CMD 执行：

```bat
schtasks /Create /TN "NightlyNetworkCollection" /SC DAILY /ST 23:30 ^
  /TR "cmd /c \"c:\Users\chuny\Desktop\paper_progarm\scripts\run_nightly_network_collection.bat\"" /F
```

验证任务：

```bat
schtasks /Query /TN "NightlyNetworkCollection" /V /FO LIST
```

手动触发：

```bat
schtasks /Run /TN "NightlyNetworkCollection"
```

删除任务：

```bat
schtasks /Delete /TN "NightlyNetworkCollection" /F
```

---

## 10. 采集运行监控页面（简易前端）

已提供页面：

- `docs/nightly_network_feature_dashboard.html`

增强能力：

- 页面可直接“开始采集/停止采集”
- 停止后下次从 `state.json` 断点继续
- 显示分片内实时进度（`current_chunk_processed_urls / current_chunk_total_urls`）

该页面可显示：

- 当前运行目录
- 分片完成数 / 剩余数
- 进度条
- 覆盖率（合并后）
- 最近更新时间
- 自动刷新（默认每 30 秒）

### 10.1 启动方式

建议在项目根目录启动本地静态服务（不要直接双击 html）：

```bash
python -m http.server 8765
```

然后浏览器打开：

```text
http://localhost:8765/docs/nightly_network_feature_dashboard.html
```

页面默认读取：

- `/data/nightly_network_runs/dataset_20260421_100k_nightly/latest_report.json`

你也可以在页面中修改 `run-name` 和刷新间隔。
