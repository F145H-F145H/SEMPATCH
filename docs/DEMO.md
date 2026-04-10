# CVE 导向二进制匹配 Demo（M1）

> **See also**: [QUICKSTART.md](QUICKSTART.md) for copy-paste commands | [WORKFLOWS.md](WORKFLOWS.md) for full workflow documentation.

本页描述**推荐默认路径**：单 ELF → Ghidra/缓存提取多模态特征 → **`TwoStagePipeline`**（SAFE 粗筛 + 多模态精排）→ 带 **CVE 列表** 的 Top-K 报告。

**研究原型**（合成特征 → 短训、无真实二进制）与 **CI 最短路径**见 [开发指南：合成数据最短路径](DEVELOPMENT.md#synthetic-short-path)；与本文 **CVE Demo** 分章，避免与「真实二进制 + CVE 报告」混淆。

<a id="主流程与相关脚本"></a>

## 主流程与相关脚本

M1 **产品入口**与 **同流水线评估** 均基于 **`matcher.two_stage.TwoStagePipeline`**（[`src/matcher/two_stage.py`](../src/matcher/two_stage.py)），职责如下。

| 用途 | 入口 | 输出 / 指标 |
|------|------|----------------|
| **CVE 匹配报告（单一命名 Demo）** | [`scripts/demo_cve_match.py`](../scripts/demo_cve_match.py) | `matches.json`、`report.md`（及可选 `query_features.json`）；每条候选含 `cve` 数组等，见下文契约。 |
| **两阶段检索指标** | [`scripts/eval_two_stage.py`](../scripts/eval_two_stage.py) | 在 `ground_truth.json`、`query_features.json`、库 SAFE 嵌入、`library_features.json` 上计算 Recall@K、Precision@K、MRR；**不**生成 CVE 报告，但与 Demo **共用** `TwoStagePipeline` 的粗筛 + 精排逻辑。 |

**自动化冒烟**：[`tests/test_eval_two_stage_cli.py`](../tests/test_eval_two_stage_cli.py) 子进程调用 `eval_two_stage.py`，数据目录为 `benchmarks/smoke/two_stage`。

**人造 CVE 库（无 Ghidra）**：`benchmarks/smoke/fake_cve/`（`FAKE-CVE-*`）+ [`tests/test_fake_cve_match.py`](../tests/test_fake_cve_match.py)；验收：`pytest -m fake_cve` 或 `make eval-smoke`（`run_cve_match_pipeline` 与 `sempatch.py match`，不调用 `scripts/demo_cve_match.py`）。

**根目录 `./sempatch` 双参数**：`./sempatch <查询ELF> <两阶段库目录>` 等价于 `python sempatch.py match --query-binary <ELF> --two-stage-dir <目录>`（库目录须含 `library_features.json` 或 `library_safe_embeddings.json`）；单参数非子命令时仍为 legacy `extract`。详见 [`sempatch_argv.py`](../sempatch_argv.py)。

**手动最小命令**（小体积 fixture；精排权重默认 `output/best_model.pth`，须存在且与特征维度匹配）：

```bash
PYTHONPATH=src python scripts/eval_two_stage.py \
  --data-dir benchmarks/smoke/two_stage \
  --max-queries 1 \
  -k 1
```

使用完整 **`data/two_stage/`** 前，需先按 [two_stage_split.md](two_stage_split.md)、[filter_features_pipeline.md](filter_features_pipeline.md) 等准备 `ground_truth.json`、`query_features.json`、`library_safe_embeddings.json`、`library_features.json`。

---

## 输入 / 输出契约

### CLI 参数

| 参数 | 必选 | 说明 |
|------|------|------|
| `--query-binary` | 与 `--query-features` 二选一 | 查询侧 ELF 绝对或相对路径（相对仓库根解析） |
| `--query-features` | 与 `--query-binary` 二选一 | 已生成的查询特征 JSON，格式 `{function_id: multimodal}`；**不调用 Ghidra**（便于 CI / 无 GUI 环境） |
| `--library-emb` | 是 | 库侧 **SAFE** 嵌入 JSON（与 `LibraryFaissIndex` 一致），每项含 `vector`；`function_id` 优先，否则 `name` 作为 ID |
| `--library-features` | 是 | 库侧 `library_features.json`：`{function_id: multimodal}` |
| `--output-dir` | 是 | 输出目录（将写入 `matches.json`、`report.md`；若从二进制提取，会写入 `query_features.json`） |
| `--model-path` | 否 | 精排权重，默认 `output/best_model.pth` |
| `--safe-model-path` | 否 | 粗筛 SAFE 权重；**须与构建 `--library-emb` 时所用权重一致** |
| `--coarse-k` | 否 | 粗筛候选数，默认 `100` |
| `--top-k` | 否 | 写入报告的精排后截断 Top-K，默认 `10` |
| `--max-queries` | 否 | 仅处理前 N 个查询函数（顺序为 `function_id` 排序），默认全部 |
| `--cpu` | 否 | 强制 CPU 推理（不优先 CUDA） |
| `--use-dfg-model` / `--no-use-dfg-model` | 否 | 精排是否构造 **含 DFG 分支** 的 `MultiModalFusionModel`：默认 **按检查点** `meta.use_dfg` 或权重键推断；若用 `--no-use-dfg-model` 则强制旧版（无 DFG 子模块），即使检查点含 DFG 权重也会 `strict=False` 丢弃多余键。训练 DFG 模型见 `train_multimodal.py --use-dfg` 与 [`docs/dfg_fusion_design.md`](dfg_fusion_design.md)。 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `matches.json` | 机器可读：配置摘要 + 每个查询的候选列表 |
| `report.md` | 人类可读摘要 |
| `query_features.json` | 仅当使用 `--query-binary` 时生成，供复现 |

### `matches.json` 字段约定

顶层：

- `config`：复现用摘要（Git 短哈希、路径、`coarse_k`、`top_k`、模型路径、`query_binary` 或 `query_features` 来源等）。
- `queries`：列表；每项包含：
  - `query_function_id`
  - `query_binary`（由 `function_id` 中 `binary_rel|entry` 解析出的二进制相对路径）
  - `match_status`：`ok` 或 `no_credible_match`（与 `report.md` 一致）
  - `filter_meta`：策略与诊断（`mode`、`reranked_count`、`reject_reason`、`min_similarity` / `tie_margin` 等）；与 `candidates` 平级
  - `top_k`：`match_filter=top_k` 时为截断 K；阈值模式下为 `null`
  - `candidates`：按策略过滤后的列表；精排得分在列表内**降序**；**不做**按函数名或 CVE 的去重/合并。每项含：
    - `rank`（1-based）
    - `candidate_function_id`
    - `candidate_name`（来自库嵌入条目的 `name`，缺失则用 ID）
    - `similarity`（精排得分，浮点）
    - `cve`：**始终为 JSON 数组**；库条目无 CVE 或空字段时为 `[]`

### 常见错误与退出码

| 情况 | 行为 |
|------|------|
| 路径不存在、库嵌入为空 | `stderr` 说明原因，`exit 1` |
| 未安装 / 未配置 Ghidra（仅 `--query-binary`） | `stderr` 提示检查 `GHIDRA_HOME` 与 `sempatch.cfg`，`exit 1` |
| 查询特征提取全部失败 | `exit 1` |
| 维度或模型加载失败 | 底层异常信息写入 `stderr`，非 0 退出 |

---

## 查询侧与库侧一致性（B.4）

1. **SAFE 粗筛**：`--library-emb` 必须由与 **`--safe-model-path`（若指定）** 一致的 SAFE 模型生成；查询向量由同一套 SAFE 对查询 multimodal 编码得到。
2. **精排**：`--model-path` 须与训练时的 MultiModalFusion 检查点一致；**不要**混用不同训练轮次、不同数据预处理的权重。若权重由 `train_multimodal.py` 保存，文件内为 `{state_dict, meta}`：`meta.use_dfg` 与特征中是否含 `multimodal.dfg` 应对齐；含 DFG 训练权重而查询/库特征无 `dfg` 键时，张量侧会以空 DFG 走前向（行为见 `docs/dfg_fusion_design.md`）。
3. **CVE 来源**：CVE **仅**来自库嵌入 JSON 中每条目的 `cve` 字段（字符串或字符串列表）；粗筛 → 精排只传递 `function_id`，**不修改** CVE。报告阶段按 `candidate_function_id` 查表附加。

---

## 扩展真实漏洞库（自备 ELF / 第三方固件样本等）

离线构建嵌入 JSON 时，在 `functions[]` 中写入可选字段 `cve`：

- 类型：字符串（单个 CVE）或 **字符串列表**（同一函数关联多个 CVE）。
- 与 `eval_bcsd.py --mode cve`、本 Demo 报告使用同一 schema（见 `memory-bank/@architecture.md` 漏洞库嵌入格式）。

将第三方元数据（二进制 → 函数 → CVE）映射到每条 `EmbeddingItem` 即可；无需改 `TwoStagePipeline` 检索逻辑。

---

## 最简命令示例

在仓库根目录、已具备 `output/best_model.pth`、库 SAFE 嵌入与 `library_features.json` 时：

```bash
PYTHONPATH=src python scripts/demo_cve_match.py \
  --query-binary path/to/query.elf \
  --library-emb data/two_stage/library_safe_embeddings.json \
  --library-features data/two_stage/library_features.json \
  --output-dir output/demo_run
```

无 Ghidra、仅验证报告管线（使用预计算查询特征）：

```bash
PYTHONPATH=src python scripts/demo_cve_match.py \
  --query-features data/two_stage/query_features.json \
  --library-emb data/two_stage/library_safe_embeddings.json \
  --library-features data/two_stage/library_features.json \
  --output-dir output/demo_run
```

---

## 预检清单（B.7）

- [ ] **环境**：Python 3.8+、`requirements.txt` 已安装；若用 `--query-binary`：Java 17+、Ghidra、`GHIDRA_HOME` 或 `sempatch.cfg` 中 `ghidra_home` 正确。
- [ ] **权重**：`output/best_model.pth`（精排）存在；若使用训练后的 SAFE 库嵌入，则准备 `output/safe_best_model.pt` 并传入 `--safe-model-path`。
- [ ] **数据**：`--library-emb` 与 `--library-features` 的 `function_id` 集合一致或可检索；库嵌入至少含一条函数。
- [ ] **耗时**：单小 ELF 首次分析取决于 Ghidra；缓存命中后主要耗时在推理与函数数量 × 精排。
- [ ] **验收**：命令 `exit 0`；`output-dir/matches.json` 可被 `python -m json.tool` 或 `jq` 解析；`report.md` 非空；每条候选含 `cve` 数组（可为 `[]`）。

`matches.json` 顶层结构示例：

```json
{
  "config": {
    "git_rev": "abc1234",
    "library_emb": "/path/to/library_safe_embeddings.json",
    "library_features": "/path/to/library_features.json",
    "rerank_model_path": "/path/to/output/best_model.pth",
    "safe_model_path": null,
    "coarse_k": 100,
    "top_k": 10,
    "query_mode": "binary",
    "query_binary": "data/binkit_subset/foo.elf"
  },
  "queries": [
    {
      "query_function_id": "data/binkit_subset/foo.elf|0x401000",
      "query_binary": "data/binkit_subset/foo.elf",
      "top_k": 10,
      "candidates": [
        {
          "rank": 1,
          "candidate_function_id": "lib|0x1",
          "candidate_name": "vuln_func",
          "similarity": 0.87,
          "cve": ["CVE-2021-1234"]
        }
      ]
    }
  ]
}
```
