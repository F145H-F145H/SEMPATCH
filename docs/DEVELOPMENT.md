# SemPatch 开发指南

**产品入口**：仓库根目录 `python sempatch.py match ...`（TwoStage CVE 匹配）。训练、评估、库构建等脚本位于 `scripts/sidechain/`，根目录 `scripts/*.py` 多为转发桩（见 `scripts/sidechain/README.md`）。

## 开发流程强制规范

1. **写任何代码前**必须完整阅读 `memory-bank/@architecture.md`（包含完整数据结构）
2. **写任何代码前**必须完整阅读 `memory-bank/@design-document.md`
3. **每完成一个重大功能或里程碑后**，必须更新 `memory-bank/@architecture.md`

<a id="synthetic-short-path"></a>

## 研究原型：合成数据最短路径（非 CVE Demo）

本节与 **[CVE 匹配 Demo](DEMO.md)** 分章：**Demo** 面向真实 ELF → Ghidra/缓存 → `TwoStagePipeline` → CVE 报告；本节面向 **无 Ghidra** 的训练与评估冒烟（CI、本地快速验证）。

在仓库根目录执行（参数以各脚本 `--help` 为准）。

**1.（可选）生成合成 multimodal 对并落盘**

```bash
PYTHONPATH=src python scripts/generate_synthetic_features.py -o data/synthetic_pairs.json -n 100
```

**2. 短训多模态精排（合成数据）**

```bash
PYTHONPATH=src python scripts/train_multimodal.py --synthetic --epochs 2
```

`--synthetic` 时默认读取 `data/synthetic_pairs.json`，可用 `--synthetic-file` 覆盖。若该文件不存在或无法解析，`PairwiseSyntheticDataset` 会在内存中按对数随机生成样本，故**仅执行步骤 2 也可完成短训**。

**2b.（可选）survey 5.1 工程对照：仅序列 vs CFG+序列（各 1 epoch）**

与 [CVE Demo](DEMO.md) 无关；用于管线级冒烟，**非**同一 `MultiModalFusionModel` 权重内的消融开关。

```bash
PYTHONPATH=src python scripts/train_safe.py --synthetic --epochs 1
PYTHONPATH=src python scripts/train_multimodal.py --synthetic --epochs 1
```

前者为 SAFE 孪生路径（序列侧）；后者为 `MultiModalFusionModel`（CFG 图分支 + 序列 + 跨模态注意力）。说明见 `memory-bank/@design-document.md` §4.5。

**3.（可选）轻量 BCSD 评估冒烟**

与步骤 2 产出的权重**无强制一致**，仅验证「嵌入 JSON → 指标」链路：

```bash
PYTHONPATH=src python scripts/eval_bcsd.py \
  --firmware-emb data/vulnerability_db/test_embeddings.json \
  --db-emb data/vulnerability_db/test_embeddings.json \
  -k 1 5 10
```

**4.（可选）同一 `TwoStagePipeline` 的两阶段指标**

若已准备 `data/two_stage/`（见下文「两阶段流水线」与 [two_stage_split.md](two_stage_split.md)），或仅需 CLI 冒烟，可使用 `scripts/eval_two_stage.py`；与 Demo 脚本关系见 [DEMO.md「主流程与相关脚本」](DEMO.md#主流程与相关脚本)。

<a id="testing-pytest"></a>

## 测试

- **全量**：在仓库根执行 `pytest`（配置见根目录 `pyproject.toml` 中 `[tool.pytest.ini_options]`）。
- **M1 / CI 推荐子集**（排除依赖真实 Ghidra 安装的用例）：

```bash
pytest -m "not ghidra"
```

要求 **0 failed**（与 `memory-bank/@prototype-survey-alignment-plan.md` 执行原则一致）。标记为 `@pytest.mark.ghidra` 的测试需本地安装并配置 Ghidra 后再跑全量套件。

## 两阶段流水线（含过滤）

过滤低质量样本后，推荐顺序：

```text
scripts/filter_index_by_pcode_len.py -i data/binkit_functions.json -o data/binkit_functions_filtered.json --min-pcode-len 16 --filtered-features-output data/two_stage/filtered_features.jsonl
scripts/prepare_two_stage_data.py --index-file data/binkit_functions_filtered.json
scripts/build_library_features.py --precomputed-multimodal data/two_stage/filtered_features.jsonl ...
scripts/train_safe.py --index-file data/binkit_functions_filtered.json --precomputed-features data/two_stage/library_features.json ...
```

侧车文件体积会随保留函数数线性增长；在超大数据集上请预留磁盘空间。

详见 [docs/two_stage_split.md](two_stage_split.md) 第六节、[docs/filter_features_pipeline.md](filter_features_pipeline.md) 与 [docs/memory_and_oom.md](memory_and_oom.md)（大程序内存与 cgroup）。

## 相关文档

- [memory-bank/@architecture.md](../memory-bank/@architecture.md)：完整数据结构、节点契约、模块接口
- [memory-bank/@design-document.md](../memory-bank/@design-document.md)：技术栈、目录结构、流程、方案、评估计划
- [memory-bank/@implementation-plan.md](../memory-bank/@implementation-plan.md)：面向 AI 的分步实施指令
