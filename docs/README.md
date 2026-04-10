# SemPatch 文档导航

## 快速上手

- [QUICKSTART.md](QUICKSTART.md) — 三步上手（smoke demo / two-stage match / 自建库）
- [ENVIRONMENT.md](ENVIRONMENT.md) — 环境快照与复现步骤
- [DOWNLOAD_HELP.md](DOWNLOAD_HELP.md) — Ghidra / 数据集下载指引

## 工作流程

- [WORKFLOWS.md](WORKFLOWS.md) — A: 建库 / B: 两阶段匹配 / C: 单阶段匹配
- [DEMO.md](DEMO.md) — CVE 匹配完整演示（M1）

## 架构设计

- [design.md](design.md) — 整体设计、研究目标、流水线、节点、ctx 约定
- [dfg_fusion_design.md](dfg_fusion_design.md) — DFG 嵌入融合设计
- [two_stage_split.md](two_stage_split.md) — 两阶段流水线拆分设计
- [asm_normalization_research.md](asm_normalization_research.md) — 汇编归一化调研

## 数据与评估

- [DATA.md](DATA.md) — 数据集概览
- [DATASET_STATUS.md](DATASET_STATUS.md) — 数据集构建状态
- [BASELINE_AND_EVAL.md](BASELINE_AND_EVAL.md) — 基线对比与评估
- [VULNERABILITY_LIBRARY.md](VULNERABILITY_LIBRARY.md) — 漏洞库管理
- [build_vulnerability_db.md](build_vulnerability_db.md) — 建库流程
- [filter_features_pipeline.md](filter_features_pipeline.md) — 特征过滤流水线

## 开发

- [DEVELOPMENT.md](DEVELOPMENT.md) — 开发工作流、CI、评测
- [api_reference.md](api_reference.md) — sempatch.py API 与 ctx 键表
- [memory_and_oom.md](memory_and_oom.md) — 内存管理与 OOM 规避

## 参考配置

- [train_default.yaml](train_default.yaml) — 训练默认参数（`--config` 用）

## ADR

- [adr/001-dag-as-executor-only.md](adr/001-dag-as-executor-only.md) — DAG 仅作执行引擎

## 其他

- [survey.md](survey.md) — 语义对比固件漏洞分析综述
