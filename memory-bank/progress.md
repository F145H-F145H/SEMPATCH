# SemPatch 实施进度

本文档为**实施进度、回归记录与已知工程限制**的权威事实来源之一；与源码及 `pytest`/文档中的可复现命令对照使用。根目录 `TODO.md`、`项目现状.md` 已废弃，请勿再以其为准。

## Phase 1 完成（@architecture.md）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 1.1 | 从 specs.py 提取 TypedDict 类及字段 | 已完成，与源码一致 |
| 1.2 | 整理 NODE_INPUT_KEYS、NODE_OUTPUT_KEYS 表格 | 已完成，与 specs.py 一致 |
| 1.3 | 归纳 test_vuln_lsir.json 的 lsir_raw 顶层结构 | 已完成，文档与 JSON 兼容 |
| 1.4 | 归纳 test_embeddings.json 的 embeddings 格式 | 已完成，与格式一致 |
| 1.5 | 创建 @architecture.md（强制规范、数据结构、节点映射、JSON schema、模块契约） | 已存在且完整 |
| 1.6 | 运行 pytest 验证无回归 | 通过，6 passed |

### 校验结果

- **specs.py**：15 个 TypedDict、NODE_INPUT_KEYS、NODE_OUTPUT_KEYS 与 @architecture.md 表格完全一致
- **test_vuln_lsir.json**：functions、basic_blocks（含 start/end）、instructions（含 address、mnemonic、operands、pcode、source_line、source_file）与文档描述一致
- **test_embeddings.json**：functions 含 name、vector，cve 可选，与文档一致

### 供后续开发者参考

1. 写代码前必读 `memory-bank/@architecture.md` 和 `memory-bank/@design-document.md`
2. 新增 TypedDict 或节点类型时，同步更新 @architecture.md
3. 修改 LSIR/Features/Embeddings JSON 格式时，更新架构文档第四、五节

---

## Phase 2 完成（@design-document.md）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 2.1 | 阅读 README、docs/design、survey 5、TODO | 已完成，项目功能、流水线、5.1/5.3 聚焦已整合 |
| 2.2 | 技术栈表格与 README、requirements 对照 | 已完成；faiss-cpu 已加入 requirements |
| 2.3 | src 目录结构、超 300 行文件标注 | 已完成，与 find 输出一致；builders.py/ghidra_runner 已标注 |
| 2.4 | 绘制 Mermaid flowchart | 已完成，与 README 策略表对应 |
| 2.5 | 整理方案 A/B/C 概要 | 已完成，与 survey 5.1–5.3 对应 |
| 2.6 | 整理评估计划 | 已完成，eval_bcsd.py 已标注待实现 |
| 2.7 | 创建 @design-document.md | 已存在且完整 |
| 2.8 | 运行 pytest 验证无回归 | 通过，6 passed |

### 校验结果

- **技术栈**：反汇编、IR、GNN、FAISS、数据集等与 README、requirements.txt 无矛盾
- **eval_bcsd.py**：已在设计文档 5.5 中标注「待实现」

---

## Phase 3 完成（模块化原则与规范落地）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 3.1 | @architecture.md 末尾新增「模块化原则」小节 | 已完成，第六节已含多文件、300 行阈值、职责边界 |
| 3.2 | @design-document 目录结构与模块职责补充模块化原则与职责边界 | 已完成，2.1 模块化原则、2.3 模块职责边界 |
| 3.3 | 扫描 src/ 生成文件行数清单，超 250 行文件标注 | 已完成，builders.py(368)、ghidra_runner.py(264) 已标注 |
| 3.4 | 运行 pytest 验证无回归 | 通过，6 passed |

### 校验结果

- **模块化原则**：已写入 @architecture.md 第六节、@design-document.md 2.1
- **职责边界**：frontend、utils、features、matcher、dag 各有清晰描述，无职责重叠
- **超 250 行文件**：builders.py 建议拆分，ghidra_runner.py 需关注

---

## Phase 4 完成（可选增强）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 4.1 | README.md 新增「开发流程」小节，引用 memory-bank 强制规范 | 已完成，位于「评估」与「依赖」之间 |
| 4.2 | 规则文件：已创建 `.cursor/rules/sempatch-memory-bank.mdc`；`docs/DEVELOPMENT.md` 已存在 | 已完成，二者均含写代码前读 @architecture、@design-document 的要求 |
| 4.3 | 运行 pytest 验证无回归 | 通过，6 passed |
| 4.4 | @implementation-plan.md 存在且含 Phase 6–10 全文 | 已完成，文档完整 |

### 校验结果

- **README**：开发流程小节引用 memory-bank 文档，详见 docs/DEVELOPMENT.md
- **.cursor/rules**：sempatch-memory-bank.mdc（alwaysApply: true）供 Cursor AI 遵循
- **docs/DEVELOPMENT.md**：含强制规范三条，供人类与 AI 参考

---

## Phase 5 完成（终验）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 5.1 | 确认 @architecture、@design-document、@implementation-plan 存在且内容完整 | 已完成，三个文档均可读且含所需章节 |
| 5.2 | 运行 pytest 最终确认全部通过 | 通过，6 passed，0 failed |

### 校验结果

- **@architecture.md**：含强制规范、数据结构、NODE 键映射、lsir_raw JSON schema、漏洞库格式、模块接口契约、模块化原则、各文件作用
- **@design-document.md**：含技术栈与数据集、目录结构与模块职责、Mermaid flowchart、方案 A/B/C、评估计划
- **@implementation-plan.md**：含 Phase 6–10 全文及分步指令与验证要求
- **pytest**：0 failed，与 Phase 6.2 基线一致

---

## Phase 6 完成（环境准备与数据获取）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 6.1 | 确认开发环境（torch、faiss、sempatch --help） | 已完成，venv 下验证通过 |
| 6.2 | 准备训练数据集（BinKit 子集） | 沿用 data/binkit_subset（50 个 OpenWrt 二进制） |
| 6.3 | 创建函数列表文件 | 已完成，scripts/build_binkit_index.py + extract_function_list.java |

### 新增/修改文件

- `src/frontend/ghidra_scripts/extract_function_list.java`：轻量函数列表导出（name + entry）
- `scripts/build_binkit_index.py`：遍历 binkit_subset，调用 Ghidra 生成 data/binkit_functions.json
- `src/utils/ghidra_runner.py`：新增 `script_output_name` 参数，支持 extract_function_list 输出 function_list.json

### 校验结果

- **data/binkit_functions.json**：50 个二进制、9689 个函数；binary 为相对路径，entry 为 0x 前缀十六进制
- **pytest**：6 passed，0 failed

---

## Phase 7 完成（BCSD 评估脚本 eval_bcsd.py）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 7.1 | 创建脚本骨架与参数解析（--firmware-emb, --db-emb, -k, --output） | 已完成 |
| 7.2 | 实现 load_embeddings(path) 返回 names 与 vectors | 已完成 |
| 7.3 | 实现 compute_top_k 暴力 Top-K 检索（余弦相似度） | 已完成 |
| 7.4 | 实现 compute_metrics（Recall@K、Precision@K、MRR） | 已完成 |
| 7.5 | 集成主流程：加载 → Top-K → 指标 → 输出 | 已完成 |
| 7.6 | 添加 tests/test_eval_bcsd.py 单元测试 | 已完成 |

### 新增/修改文件

- `scripts/eval_bcsd.py`：BCSD 评估脚本，支持 Recall@K、Precision@K、MRR
- `tests/test_eval_bcsd.py`：load_embeddings、compute_top_k、compute_metrics 单元测试

### 校验结果

- `python scripts/eval_bcsd.py --help` 正常显示参数
- 使用 test_embeddings.json 作为 firmware 与 db 时输出合理（baseline 嵌入 Recall@1 约 0.4）
- `pytest tests/test_eval_bcsd.py` 6 个测试全部通过
- `pytest` 全项目 12 passed，无回归

### 待用户验证

完成验证后再进行 Phase 8（训练流程）。

---

## Phase 8 完成（训练流程）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 8.1 | PairwiseFunctionDataset（dataset.py） | 已完成 |
| 8.2 | ContrastiveLoss（losses.py） | 已完成 |
| 8.3 | Trainer 类（trainer.py） | 已完成 |
| 8.4 | train_multimodal.py 集成模型与数据流 | 已完成 |
| 8.5 | 训练配置与日志（argparse、TensorBoard 可选） | 已完成 |
| 8.6 | generate_synthetic_features.py、PairwiseSyntheticDataset | 已完成 |

### 新增/修改文件

- `src/features/dataset.py`：PairwiseFunctionDataset、PairwiseSyntheticDataset、辅助函数
- `src/features/losses.py`：ContrastiveLoss（余弦相似度）
- `src/features/trainer.py`：Trainer（train_epoch、validate、fit、TensorBoard）
- `scripts/train_multimodal.py`：孪生网络训练入口
- `scripts/generate_synthetic_features.py`：合成 multimodal 特征对
- `tests/test_dataset.py`：数据集、损失、Trainer 单元测试

### 校验结果

- `python scripts/generate_synthetic_features.py -o data/synthetic_pairs.json -n 100` 正常
- `python scripts/train_multimodal.py --synthetic --epochs 2` 完成并生成 output/best_model.pth
- `pytest` 全项目 15 passed，无回归

### 待用户验证

完成验证后再进行 Phase 9（集成训练模型到推理）。

---

## Phase 9 完成（集成训练模型到推理）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 9.1 | inference.py 支持 model_path、SEMPATCH_MODEL_PATH、load_state_dict | 已完成 |
| 9.2 | verify_embedding_consistency.py 验证同函数/不同函数嵌入相似度 | 已完成 |

### 新增/修改文件

- `src/features/inference.py`：embed_batch 增加 model_path 参数，_resolve_model_path，加载训练权重
- `src/dag/nodes/embed_node.py`：从 params/ctx 读取 model_path 并传入 embed_batch
- `scripts/build_embeddings_db.py`：增加 --model-path 参数
- `scripts/verify_embedding_consistency.py`：嵌入一致性验证脚本
- `tests/test_inference.py`：embed_batch 单元测试（5 个用例）

### 校验结果

- `pytest tests/test_inference.py` 5 passed
- `python scripts/verify_embedding_consistency.py --model-path output/verify_model.pth` 同函数相似度 1.0
- `pytest` 全项目 20 passed，无回归

### 待用户验证

完成验证后再进行 Phase 10（端到端基础流程验证）。

---

## Phase 10 完成（端到端基础流程验证）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 10.1 | build_embeddings_db.py 支持 --input-dir、--index-file 批量构建 | 已完成 |
| 10.2 | 使用 eval_bcsd.py 对 BinKit 嵌入库自评 | 已完成 |
| 10.3 | 记录结果并更新 memory-bank 文档 | 已完成 |

### 新增/修改文件

- `scripts/build_embeddings_db.py`：新增 --input-dir、--index-file 批量模式，支持遍历多二进制合并输出
- `data/binkit_embeddings.json`：50 个二进制、9689 个函数嵌入（训练模型 output/best_model.pth）
- `output/eval_bcsd_result.json`：BCSD 评估结果

### 评估结果（BinKit 子集自评，firmware-emb = db-emb）

| K | Recall@K | Precision@K | MRR |
|---|----------|-------------|-----|
| 1 | 0.4883 | 0.4883 | 0.4883 |
| 5 | 0.5389 | 0.1206 | 0.5076 |
| 10 | 0.5630 | 0.0704 | 0.5107 |

- 训练模型（output/best_model.pth，合成数据训练 2 epoch）相较 baseline（Recall@1 约 0.4）有提升

### 校验结果

- `python scripts/build_embeddings_db.py --index-file data/binkit_functions.json -o data/binkit_embeddings.json --model-path output/best_model.pth` 成功，输出 9689 个函数嵌入
- `python scripts/eval_bcsd.py --firmware-emb data/binkit_embeddings.json --db-emb data/binkit_embeddings.json -k 1 5 10` 输出合理

---

## Phase 11 完成（模型训练改进）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 11.1 | 验证 PairwiseFunctionDataset 真实数据路径，支持 use_disk_cache（默认启用） | 已完成 |
| 11.2 | 增加真实数据训练规模（num-pairs 默认 2000、epochs 默认 20） | 已完成 |
| 11.3 | 添加超参数配置（embed-dim、hidden-dim、num-gnn-layers、num-transformer-layers、output-dim） | 已完成 |
| 11.4 | 评估 DFG 融合状态，记录「DFG 分支扩展」为后续任务 | 已完成 |

### 修改文件

- `scripts/train_multimodal.py`：--use-disk-cache（默认 True）、--no-disk-cache；num-pairs 默认 2000、epochs 默认 20；新增 embed-dim、hidden-dim、num-gnn-layers、num-transformer-layers、output-dim
- `memory-bank/@design-document.md`：六、总结与后续步骤 新增「DFG 分支扩展」条目

### DFG 评估结论（11.4）

- `extract_graph_features` 已提取 CFG 与 DFG
- `fuse_features` 的 `_build_graph_for_model` 仅使用 CFG，未融合 DFG
- 当前图分支基于 CFG/ACFG；DFG 分支扩展列为后续任务

### 待用户验证

完成验证后再进行 Phase 12（评估能力增强）。

---

## Phase 12 完成（评估能力增强）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 12.1 | load_embeddings 扩展 CVE 字段，返回 (names, vectors, cves) | 已完成 |
| 12.2 | build_relevant_pairs_by_cve，CVE 模式相关性 | 已完成 |
| 12.3 | eval_bcsd --mode name/cve 参数 | 已完成 |
| 12.4 | scripts/build_library_binary_index.py 输出自备库清单 JSON；CVE 用 annotate_library_embeddings_cve.py | 已完成 |
| 12.5 | scripts/eval_challenge.py 支持 --query-index/--db-index | 已完成 |
| 12.6 | SAFE 基线、build_embeddings_db --model、run_baseline_comparison | 已完成 |

### 新增/修改文件

- `scripts/eval_bcsd.py`：load_embeddings 返回 cves；build_relevant_pairs_by_cve；--mode name/cve
- `scripts/build_library_binary_index.py`：自备漏洞 ELF 目录扫描，输出 `--from-index-file` 兼容索引
- `scripts/annotate_library_embeddings_cve.py`：库嵌入 `cve` 字段合并
- `scripts/eval_challenge.py`：跨编译器/优化/架构挑战场景评估
- `scripts/run_baseline_comparison.py`：SemPatch vs SAFE 基线对比
- `scripts/build_embeddings_db.py`：--model sempatch|safe
- `src/features/baselines/safe.py`：SAFE 风格序列编码器
- `tests/test_eval_bcsd.py`：test_load_embeddings_with_cve、test_build_relevant_pairs_by_cve

### 基线来源

- **SAFE**：轻量实现，P-code token embedding + mean 聚合，参考 [gadiluna/SAFE](https://github.com/gadiluna/SAFE)
- 复现：`python scripts/build_embeddings_db.py <lsir.json> -o out.json --model safe`；`python scripts/run_baseline_comparison.py --firmware-emb out.json --db-emb out.json`

### 待用户验证

完成验证后再进行 Phase 14（代码结构优化）。

---

## Phase 13 完成（实验管理与可复现性）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 13.1 | 默认启用 TensorBoard，--tb-dir 未指定时使用 output/tensorboard/<timestamp>，--no-tb 禁用 | 已完成 |
| 13.2 | --wandb、--wandb-project 可选接入 W&B（仅 except ImportError） | 已完成 |
| 13.3 | configs/train_default.yaml，_load_train_config，--config 命令行覆盖 | 已完成 |
| 13.4 | --seed 随机种子管理，torch/random/numpy，DataLoader generator，记录 TB/W&B | 已完成 |
| 13.5 | tests/test_train_reproducibility.py 验证 seed 可复现性 | 已完成 |

### 新增/修改文件

- `configs/train_default.yaml`：训练默认配置（epochs、batch_size、lr、seed 等）
- `scripts/train_multimodal.py`：TensorBoard 默认启用、W&B、config 加载、seed、_load_train_config
- `src/features/trainer.py`：fit 增加可选 on_epoch_end 回调（供 W&B 使用）
- `tests/test_train_reproducibility.py`：seed 可复现性单元测试
- `requirements.txt`：PyYAML>=6.0

### 验证结果

- `python scripts/train_multimodal.py --synthetic --epochs 1` 默认生成 output/tensorboard/<timestamp>/
- `--config configs/train_default.yaml` 加载配置，命令行参数覆盖
- `--seed 42` 连续两次运行 train_loss/val_loss 完全一致
- `pytest tests/test_train_reproducibility.py` 通过
- `pytest` 全项目 23 passed

---

## Phase 14 完成（代码结构优化）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 14.1 | 拆分 builders.py 为 builders/ 包（fusion、traditional、unpack、ghidra） | 已完成 |
| 14.2 | 拆分 ghidra_runner.py，抽取 _ghidra_helpers.py | 已完成 |
| 14.3 | UnpackNode 添加 binwalk 存在性检查，明确错误提示 | 已完成 |

### 新增/修改文件

- `src/dag/builders/`：fusion.py(203)、traditional.py(104)、unpack.py(37)、ghidra.py(40)、__init__.py(37)；删除原 builders.py
- `src/utils/_ghidra_helpers.py`：validate_ghidra_environment、can_skip_ghidra、binary_cache_key、build_ghidra_command、execute_ghidra_process、write_to_binary_cache、read_from_binary_cache
- `src/utils/ghidra_runner.py`：精简为入口编排，调用 helpers
- `src/dag/nodes/unpack_node.py`：调用 binwalk 前 shutil.which 检查，不存在时抛出明确 RuntimeError
- `tests/test_dag/test_nodes.py`：test_unpack_node_binwalk_not_found

### 验证结果

- `pytest` 全项目 24 passed
- `sempatch --help` 正常
- 各 builders 子文件 ≤ 300 行；ghidra_runner.py(186)、_ghidra_helpers.py(182) ≤ 300 行

### 待用户验证

完成验证后再进行 Phase 15（测试与文档补充）。

---

## Phase 15 完成（测试与文档补充）

**完成时间**：2025-03-16

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 15.1 | tests/test_pcode_normalizer.py：normalize_varnode、opcode、lsir_raw 等 | 已完成 |
| 15.2 | tests/test_features/test_feature_extractors.py：extract_graph/sequence/acfg、fuse_features | 已完成 |
| 15.3 | tests/test_dag_mock.py、tests/fixtures/lsir_raw_mock.json：无 Ghidra DAG 流水线 | 已完成 |
| 15.4 | README 项目结构：完整目录树与顶层目录说明 | 已完成 |
| 15.5 | README 快速上手指南：端到端命令序列 | 已完成 |

### 新增/修改文件

- `tests/test_pcode_normalizer.py`：12 个 P-code 规范化单元测试
- `tests/test_features/test_feature_extractors.py`：5 个特征提取模块单元测试
- `tests/test_dag_mock.py`：无 Ghidra DAG 集成测试
- `tests/fixtures/lsir_raw_mock.json`：预生成 lsir_raw fixture
- `README.md`：项目结构目录树、快速上手指南

### 验证结果

- `pytest tests/test_pcode_normalizer.py` 12 passed
- `pytest tests/test_features/test_feature_extractors.py` 5 passed
- `pytest tests/test_dag_mock.py` 1 passed（无需 Ghidra）
- `pytest` 全项目 42 passed

---

## 两阶段框架 阶段 A 完成（数据准备）

**完成时间**：2025-03-17

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| A.1 | 确认 BinKit 数据与索引（50 二进制、9689 函数） | 已完成 |
| A.2 | 定义库/查询划分逻辑，文档 docs/two_stage_split.md | 已完成 |
| A.3 | 脚本 prepare_two_stage_data.py，输出 library/query/ground_truth | 已完成 |
| A.4 | 脚本 build_library_features.py，预计算库 multimodal 特征 | 已完成 |
| A.5 | build_library_features 支持 --query-index，同时输出 query_features.json | 已完成 |

### 新增/修改文件

- `docs/two_stage_split.md`：划分规则、统计量、示例查询
- `scripts/prepare_two_stage_data.py`：按二进制随机划分，输出 library_index、query_index、ground_truth
- `scripts/build_library_features.py`：遍历索引调用 Ghidra + 特征提取，输出 library_features、query_features（可选）
- `data/two_stage/library_index.json`：38 个二进制、6805 个库函数
- `data/two_stage/query_index.json`：12 个二进制、2884 个查询函数
- `data/two_stage/ground_truth.json`：1109 个正样本充足查询 → 正样本列表
- `data/two_stage/library_features.json`：6805 个 function_id → multimodal 特征
- `data/two_stage/query_features.json`：2884 个 function_id → multimodal 特征

### 统计量（seed=42, min-queries=1000）

- 划分比例：库 75% / 查询 25%（20% 不足 1000，自动调整为 25%）
- 正样本充足查询数：1109

### 验证结果

- `python scripts/prepare_two_stage_data.py` 两次运行输出一致（可复现）
- ground_truth 中抽查 5 个查询，正样本均在 library_index 中存在
- library_features、query_features 覆盖索引全部函数，每项含 graph、sequence
- `pytest` 全项目 42 passed，无回归

### 供后续开发者参考

1. 划分脚本使用 `--seed 42` 保证可复现；`--min-queries 1000` 可调整
2. build_library_features 按二进制缓存 lsir_raw，避免重复 Ghidra 调用
3. function_id 格式：`binary_path|entry_address`（如 `data/binkit_subset/addpart.elf|0x401000`）
4. 阶段 B 将使用 library_features.json 构建 SAFE 嵌入与 FAISS 索引

---

## 两阶段框架 阶段 B 完成（粗筛模型与 FAISS 索引）

**完成时间**：2025-03-17

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| B.1 | 扩展 build_embeddings_db 支持 --features-file，输出 library_safe_embeddings.json | 已完成 |
| B.2 | 实现 l2_normalize、l2_normalize_single（matcher/similarity.py） | 已完成 |
| B.3 | LibraryFaissIndex：从库嵌入构建 FAISS IndexFlatIP，search(query_vector, k) | 已完成 |
| B.4 | retrieve_coarse(query_features, library_faiss_index, k) 粗筛检索接口 | 已完成 |

### 新增/修改文件

- `scripts/build_embeddings_db.py`：新增 `--features-file` 参数，从预计算特征直接构建 SAFE 嵌入
- `src/matcher/similarity.py`：新增 `l2_normalize`、`l2_normalize_single`
- `src/matcher/faiss_library.py`：`LibraryFaissIndex` 类、`retrieve_coarse` 函数
- `data/two_stage/library_safe_embeddings.json`：6805 个库函数 SAFE 嵌入（未训练）
- `tests/test_matcher/test_similarity.py`：`test_l2_normalize_single`、`test_l2_normalize`、`test_l2_normalize_zero_vector`
- `tests/test_matcher/test_faiss_library.py`：LibraryFaissIndex、retrieve_coarse 单元测试

### 验证结果

- `python scripts/build_embeddings_db.py --features-file data/two_stage/library_features.json -o data/two_stage/library_safe_embeddings.json` 成功，输出 6805 个嵌入
- 输出格式含 `function_id` 与 `vector`，向量维度 128
- LibraryFaissIndex 自查询、空库、k>库大小 测试通过
- retrieve_coarse 空库返回空、k 限制正确
- `pytest` 全项目 51 passed，无回归

---

## 两阶段框架 阶段 C 完成（精排模型）

**完成时间**：2025-03-17

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| C.1 | 确认精排相似度计算方式（查询+候选分别嵌入，余弦相似度） | 已完成 |
| C.2 | 实现 compute_rerank_scores 批量精排接口 | 已完成 |
| C.3 | 实现 load_candidate_features 候选特征查找 | 已完成 |

### 新增/修改文件

- `src/matcher/rerank.py`：`load_candidate_features`、`compute_rerank_scores`、`_collect_vocab_from_multimodals`
- `tests/test_matcher/test_rerank.py`：8 个单元测试（同函数得分~1、正负对、排序、空候选、特征查找、缺失处理）
- `scripts/verify_rerank.py`：粗筛+精排集成验证脚本（可选）

### 验证结果

- `pytest tests/test_matcher/test_rerank.py` 8 passed
- `pytest` 全项目 59 passed，无回归
- `python scripts/verify_rerank.py --top-k 20` 正常输出 Top-5 精排结果

### 待用户验证

完成验证后再进行阶段 D（流水线整合）。

---

## 两阶段框架 阶段 D 完成（流水线整合）

**完成时间**：2025-03-17

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| D.1 | 实现 TwoStagePipeline 类（matcher/two_stage.py） | 已完成 |
| D.2 | 与现有 DAG/节点解耦，独立可调用 | 已完成 |
| D.3 | 新增 eval_two_stage.py 评估脚本 | 已完成 |

### 新增/修改文件

- `src/matcher/two_stage.py`：TwoStagePipeline 类，retrieve、rerank、retrieve_and_rerank
- `scripts/eval_two_stage.py`：两阶段评估（--ground-truth、--query-features、--library-*、-k、--output、--max-queries）
- `tests/test_matcher/test_two_stage.py`：6 个单元测试

### 验证结果

- `python scripts/eval_two_stage.py --help` 正常显示参数
- `python scripts/verify_rerank.py` 粗筛+精排流程正确，正样本出现于 Top-5
- `python scripts/eval_two_stage.py --max-queries 20` 输出 Recall@K、MRR
- `pytest tests/test_matcher/test_two_stage.py` 6 passed
- `pytest` 全项目 65 passed，无回归

### 待用户验证

完成验证后再进行阶段 E（实验验证）。

---

## P-code 长度索引预过滤完成

**完成时间**：2025-03-18

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 0 | 抽取 extract_multimodal_from_lsir_raw 至 utils，重构 build_library_features 复用 | 已完成 |
| 1 | 新增 filter_index_by_pcode_len.py | 已完成 |
| 2 | 单元测试（extract_multimodal、filter_index） | 已完成 |
| 3 | 更新 @architecture.md 与流水线文档 | 已完成 |

### 新增/修改文件

- `src/utils/feature_extractors/multimodal_extraction.py`：extract_multimodal_from_lsir_raw，lsir_raw → multimodal 单一入口
- `scripts/filter_index_by_pcode_len.py`：按 len(pcode_tokens) >= 16 预过滤索引
- `scripts/build_library_features.py`：重构为调用 extract_multimodal_from_lsir_raw
- `tests/test_filter_index_by_pcode_len.py`：4 个单元测试（mock Ghidra）
- `tests/test_features/test_feature_extractors.py`：新增 3 个 extract_multimodal_from_lsir_raw 测试
- `memory-bank/@architecture.md`：补充 multimodal_extraction、filter_index_by_pcode_len
- `docs/two_stage_split.md`、`docs/DEVELOPMENT.md`：推荐流水线顺序

### 验证结果

- `pytest tests/test_filter_index_by_pcode_len.py tests/test_features/test_feature_extractors.py` 11 passed
- `pytest` 全项目 75 passed，无回归

### 推荐流水线

```text
scripts/filter_index_by_pcode_len.py -i data/binkit_functions.json -o data/binkit_functions_filtered.json --min-pcode-len 16
scripts/prepare_two_stage_data.py --index-file data/binkit_functions_filtered.json
scripts/build_library_features.py ...
scripts/train_safe.py --index-file data/binkit_functions_filtered.json ...
```

---

## 实施计划澄清更新（2025-03-16）

根据开发澄清问答，已更新 @implementation-plan.md 及关联文档：

| 澄清项 | 更新内容 |
|--------|----------|
| 6.2 数据规模 | 明确：至少 50 个二进制、1000 个函数；补充 BinKit 下载与筛选步骤 |
| 6.3 索引格式 | binary 用相对路径；entry 为十六进制字符串 |
| 7.4 指标 | Recall@K（Success Rate）、Precision@K、MRR 精确定义；relevant_pairs 含所有同名对 |
| 8.1 数据集 | 正对跨二进制、负对混合来源；缓存键 (binary_path, entry)；动态按需提取；缓存目录 data/features_cache/ |
| 8.4 孪生网络 | 方式 A：训练脚本中调用模型两次，不修改 MultiModalFusionModel |
| 8.6 合成数据 | 生成 multimodal 格式；脚本名 generate_synthetic_features.py；随机整数填充 |
| 9.1 模型路径 | 优先级：函数参数 > SEMPATCH_MODEL_PATH > baseline |
| 10.1 批量构建 | build_embeddings_db.py 支持 --input-dir 或 --index-file 批量处理 |
| 目录命名 | @design-document.md 中 SemPatch_replain/ 已改为 SemPatch/ |

---

## 过滤侧车特征优化完成（@optimize-filter_index）

**完成时间**：2026-03-20

### 步骤完成情况

| 步骤 | 描述 | 状态 |
|------|------|------|
| 0 | 契约冻结与数据流文档补齐（新增 docs/filter_features_pipeline.md） | 已完成 |
| 1 | filter_index_by_pcode_len 支持侧车特征输出，保留分支复用一次提取结果 | 已完成 |
| 2 | build_library_features 支持 --precomputed-multimodal 合并命中，缺失回退提取 | 已完成 |
| 3 | PairwiseFunctionDataset / train_safe 支持预计算特征优先读取 | 已完成 |
| 4 | 单元测试与开发文档更新（架构、开发流程） | 已完成 |
| 5 | 小规模性能与一致性验收记录 | 已完成 |

### 新增/修改文件

- `scripts/filter_index_by_pcode_len.py`：新增 `--filtered-features-output`；`_pcode_filter_worker` 保留分支返回 `multimodal`；`_filter_index` 支持聚合侧车 map
- `scripts/build_library_features.py`：新增 `--precomputed-multimodal` 与加载逻辑；命中 map 时跳过 `extract_multimodal_from_lsir_raw`
- `src/features/dataset.py`：`PairwiseFunctionDataset` 新增 `precomputed_features_path`，`_get_features` 优先读取全局 map
- `scripts/train_safe.py`：新增 `--precomputed-features` 并传入 `PairwiseFunctionDataset`
- `tests/test_filter_index_by_pcode_len.py`：补充侧车聚合与阈值行为测试
- `tests/test_build_library_features.py`：新增命中/未命中分支测试与 precomputed 加载测试
- `tests/test_dataset.py`：新增预计算特征优先命中测试
- `docs/filter_features_pipeline.md`：新增过滤→合并构建数据流说明与 Mermaid 图
- `docs/DEVELOPMENT.md`：补充过滤侧车与合并构建推荐命令
- `memory-bank/@architecture.md`：补充 filtered_features 产物、build/train 新参数与衔接说明

### 验证结果

- 单测：在仓库根目录执行 `python -m pytest tests/test_filter_index_by_pcode_len.py tests/test_build_library_features.py tests/test_dataset.py`（或 `.venv/bin/python -m pytest …`）
  - 结果：`11 passed`
- 一致性检查（小样本 mock）：
  - `build_library_features` 全量重算与“过滤侧车 + 合并命中”结果 key 集合一致
  - 抽样 `function_id` 的 multimodal 深度相等（`sample_equal=True`）
- 性能对比（小样本 mock，4 函数，同一数据）：
  - baseline（过滤不写侧车 + 构建全量提取）：`0.2505s`
  - optimized（过滤写侧车 + 构建全命中）：`0.1211s`
  - 结论：优化路径约 `51.7%` 更快

### 备注

- 默认 CLI 行为保持兼容：不传新参数时，`filter_index_by_pcode_len` 仅输出过滤索引，`build_library_features` 仍按原路径提取。

### 内存与 OOM 缓解（2026-03-20 增补）

- 侧车默认改为 **JSONL 流式写出**，不再在内存中累积整库 `{function_id: multimodal}`；进程池路径使用 `map(..., chunksize=1)` 并逐条消费，避免 `list(map(...))` 一次性缓存全部子进程结果。
- `build_library_features` / `PairwiseFunctionDataset` 通过 `utils/precomputed_multimodal_io.py` 加载侧车时，**仅读入当前索引中出现的 function_id**（JSONL 可提前结束扫描）；整文件 `.json` 仍会 `json.load` 全量，大库请改用 `.jsonl`。
- 大库若仍内存紧张，可对过滤阶段使用 `--workers 0` 或较小 `--workers`，降低 fork 子进程与结果队列中的并行副本峰值。

### 内存上限与缓解 CLI（2026-03-20 增补）

- 新增 `src/utils/memory_mitigation.py`：`--max-memory-mb` / `SEMPATCH_MAX_MEMORY_MB`（RLIMIT_AS，best-effort）、`--gc-after-each-binary`、大 lsir 函数数告警。`max_tasks_per_child` 与 **fork** 不兼容，过滤脚本固定 fork，故 `--process-pool-recycle-after-tasks` 会被自动忽略并打日志。
- `filter_index_by_pcode_len.py` 与 `build_library_features.py` 已接入上述选项；外部 cgroup/systemd 说明见 `docs/memory_and_oom.md`。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 A · 步骤 A.1（能力清单已冻结）

> 与上文「两阶段框架 阶段 A（数据准备，BinKit/索引）」区分：本节对应 `memory-bank/@prototype-survey-alignment-plan.md` 的**现状审计**步骤 A.1。

### 冻结声明

M1 / CVE 导向 Demo 的固定推荐执行路径为 **`TwoStagePipeline`**（`src/matcher/two_stage.py`），与本文档前节「Demo（M1）固定路径」及 `memory-bank/@architecture.md` 一致。

### Demo 相关能力清单

| 名称 | 路径或入口 | 在「二进制 → 嵌入 → 带 CVE 的检索」中的角色 | 验证方式 |
|------|------------|-----------------------------------------------|----------|
| build_embeddings_db | `scripts/build_embeddings_db.py` | 从单文件 LSIR、`--index-file` / `--input-dir` 索引或 `--features-file` 预计算特征生成嵌入 JSON；`EmbeddingItem` 可含可选 `cve` 列表，供漏洞库侧向量构建 | 见 **步骤 A.2** 冒烟矩阵 |
| eval_bcsd | `scripts/eval_bcsd.py` | 加载 query/db 两侧嵌入 JSON，做 Top-K 与 Recall@K、MRR；`--mode cve` 下按 CVE 列表定义相关对 | 见 **步骤 A.2** 冒烟矩阵 |
| sempatch（DAG / CLI） | 项目根 `sempatch.py`（子命令 `compare` 等，`dag.run_dag`） | 多策略编排（`semantic_embed`、`traditional_*`、`fusion` 等），单二进制 vs 库；**非 M1 Demo 默认推荐路径** | 见 **步骤 A.2** 冒烟矩阵 |
| run_firmware_vs_db | `sempatch.run_firmware_vs_db`（定义于 `sempatch.py`） | 编程入口：构建 compare DAG、执行、写 `diff_result.json`；`scripts/run_benchmark.py` 等引用 | 见 **步骤 A.2** 冒烟矩阵 |
| build_binkit_index | `scripts/build_binkit_index.py` | 批量 Ghidra 导出 lsir_raw、写入 `BINARY_CACHE_DIR`，产出 `binkit_functions.json`，支撑后续批量特征/嵌入流水线 | 见 **步骤 A.2** 冒烟矩阵 |
| TwoStagePipeline | `src/matcher/two_stage.py`；参考 `scripts/eval_two_stage.py` | M1 固定推荐：SAFE 粗筛（FAISS）+ 多模态精排；CVE 字段在结果中的透传与单一 Demo 入口在阶段 B 收敛 | 见 **步骤 A.2** 冒烟矩阵 |

### DAG 与 TwoStagePipeline 的关系

**Demo 文档与验收契约以 `TwoStagePipeline` 为准。** `sempatch compare` / `run_firmware_vs_db` 仍为可选的多策略 DAG 入口，用于历史兼容与对比实验，**不作为 M1 CVE 匹配 Demo 的默认路径**，避免与 `compare --strategy` 选项混淆。

### A.1 实施记录

- **日期**：2026-03-22
- **内容**：完成 `@prototype-survey-alignment-plan` 步骤 A.1：上表写入本文件；修复 `features.inference` 缺失的 `resolve_inference_device` / `_resolve_model_path` / `run_with_cuda_oom_fallback`（供 `matcher/rerank.py` 使用），以及 `EmbedNode` 对 `embed_batch` 传入不存在的 `model_path` 关键字导致嵌入节点失败、DAG mock 集成测试无法得到 `diff_result`。
- **回归**：`pytest -m "not ghidra"` → **97 passed**，1 skipped（本机 `.venv`）。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 A · 步骤 A.3–A.5

**日期**：2026-03-22

### A.3 测试套件基线（`pytest -m "not ghidra"`）

| 指标 | 结果（agent，项目根 `.venv`，Python 3.12） |
|------|---------------------------------------------|
| 选中用例 | 99 |
| **passed** | **98** |
| skipped | 1（`tests/test_matcher/test_faiss_library.py` 中需 faiss 的用例） |
| deselected | 1（`@pytest.mark.ghidra`） |
| **failed** | **0** |

**验证**：与对齐计划一致，要求 **0 failed**。请维护者本地复跑同一命令确认环境一致。

**已知告警（历史）**：A.3 当时 `pytest.mark.ghidra` 未注册会有 `PytestUnknownMarkWarning`；**阶段 C** 已在 `pyproject.toml` 注册 `ghidra`，当前应无此警告。

### A.4 文档与 `TODO.md` 一致性

| 修改项 | 说明 |
|--------|------|
| `TODO.md` Phase 2.2 | 「训练流程」与 `progress.md` Phase 8–11 矛盾已消除：标为 **[x]**，并指向 `train_multimodal.py` / 合成与真实数据路径 |
| `memory-bank/@design-document.md` §4.1、§4.4 | 与 survey 对齐：**已实现** = CFG + P-code 序列 + 跨模态注意力；**DFG 融入嵌入** = 阶段 H，未落地前不得声称已实现 |
| `memory-bank/@design-document.md` §6 | 拆成 **6.1 Demo 交付（M1）** 与 **6.2 DFG 融合（阶段 H）**，待办边界与 `TODO.md`、`@prototype-survey-alignment-plan.md` 一致 |
| `memory-bank/progress.md`（A.1 小节） | 修正对齐计划文件名笔误：`@prototype-survey-alignment.md` → `@prototype-survey-alignment-plan.md` |

**冷读检查**：设计文档「总结」节不再将「完成训练」列为唯一首要项而不区分 Demo；DFG 状态与 `TODO.md` Phase 2.4 一致。

### A.5 `TODO.md` 中阶段 H（DFG）Phase

- 在 **Phase 2.4** 下重写可勾选子项，与 **`@prototype-survey-alignment-plan.md` 阶段 H.1–H.7** 逐条对应，并增补 **H.test**（pytest / 降级路径）。
- **未**将任何 H 子项标为已完成。

### 历史约定（已归档）

原约定：用户确认测试通过后追加阶段 B 纪要并更新 `@architecture.md`。下列 **阶段 B** 小节已在实现合并中写入；**阶段 C** 见下文专节。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 B 完成

**日期**：2026-03-22

### 供后续开发者参考（实施摘要）

| 步骤 | 内容 |
|------|------|
| B.1 / B.4 / B.7 | 新增 [docs/DEMO.md](../docs/DEMO.md)：CLI 与输出契约、SAFE/精排权重一致性、CVE 查表附加说明、预检清单与示例命令 |
| B.2 | [scripts/eval_bcsd.py](../scripts/eval_bcsd.py)：`load_embeddings` 返回每条 `cve` 为 `List[str]`；`build_relevant_pairs_by_cve` 按 **CVE 集合交集** 判相关；[benchmarks/smoke/fake_cve/](benchmarks/smoke/fake_cve/)（`FAKE-CVE-*`、同名不同 CVE 两条库函数） |
| B.3 / B.5 | 新增 [scripts/demo_cve_match.py](../scripts/demo_cve_match.py)：`TwoStagePipeline`、`matches.json` + `report.md`、`git rev` 与路径摘要；候选 **始终含 `cve` 数组**、**不按名/CVE 去重** |
| B.5 / B.3 验证 | [tests/test_demo_cve_match.py](../tests/test_demo_cve_match.py)：元数据单测、CLI 冒烟、`top_k=2` 下两条「dup」候选并存 |
| B.6 | [README.md](../README.md) 增加「漏洞匹配 Demo」链至 `docs/DEMO.md` 与一行最简命令 |
| 文档 | [memory-bank/@architecture.md](@architecture.md) 增补 `demo_cve_match`、`docs/DEMO.md`、夹具表及 **memory-bank 文档索引** |

### 验证（维护者可复跑）

```bash
pytest -m "not ghidra"
```

**Agent 本机（`.venv`，Python 3.12）**：106 passed，1 skipped，1 deselected，0 failed。

### 范围说明

- 阶段 C 已完成，见下一节「阶段 C 完成」。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 C 完成

**日期**：2026-03-25

### 供后续开发者参考

| 步骤 | 内容 |
|------|------|
| C.1 | [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) 锚点 `#synthetic-short-path`：合成 multimodal 对（可选 `generate_synthetic_features.py`）→ `train_multimodal.py --synthetic`（默认/可选 `data/synthetic_pairs.json`，缺失则内存生成）→ 可选 `eval_bcsd` 冒烟 → 可选 `eval_two_stage` 交叉引用 [DEMO.md](../docs/DEMO.md)；与 CVE 真实二进制 Demo 分章 |
| C.2 | [docs/DEMO.md](../docs/DEMO.md) 锚点 `#主流程与相关脚本`：`demo_cve_match.py`（CVE 报告）与 `eval_two_stage.py`（Recall@K / MRR 等指标）并列说明，二者均基于 `TwoStagePipeline`；含 `benchmarks/smoke/two_stage` 最小命令 |
| C.3 | [pyproject.toml](../pyproject.toml) 注册 `ghidra` marker；DEVELOPMENT 锚点 `#testing-pytest`：`pytest` 全量与 `pytest -m "not ghidra"`（M1/CI 门槛，0 failed） |

### 验证

```bash
pytest -m "not ghidra"
```

**Agent 本机（`.venv`，Python 3.12）**：106 passed，1 skipped，1 deselected，0 failed；`ghidra` 已注册，无 UnknownMark 警告。

### 范围说明

- 阶段 D（P-code 规范化审计与 5.3 文档）见下一节「阶段 D 完成」。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 D 完成

**日期**：2026-03-25

### 步骤勾选

| 步骤 | 状态 |
|------|------|
| D.1 P-code 规范化审计 | 已完成：`memory-bank/@architecture.md` **§1.1a**（入口、默认值、`binary_cache` 时机、opcode/varnode 边界） |
| D.2 小型规范化统计 | 已完成：`scripts/pcode_norm_fixture_digest.py` + 本节验证命令 |
| D.3 汇编归一化调研 | 已完成：`docs/asm_normalization_research.md`；`memory-bank/@design-document.md` §4.3 已引用 |

### 供后续开发者参考

| 交付物 | 说明 |
|--------|------|
| §1.1a | 默认在消费 `lsir_raw` 时规范化；`build_pcode_normalize_node` 已导出但默认 fusion 链不单独插入 |
| `scripts/pcode_norm_fixture_digest.py` | 固定 fixture 上可重复的 pcode 条数与 `normalized_sha256` |
| `docs/asm_normalization_research.md` | 汇编侧可选扩展与 P-code 主路径分工 |

### 验证（维护者可复跑）

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pcode_normalizer.py -q
PYTHONPATH=src .venv/bin/python scripts/pcode_norm_fixture_digest.py
```

**Agent 本机（`.venv`，Python 3.12）**：`test_pcode_normalizer` **12 passed**；digest 脚本输出 `pcode_ops_before=2 pcode_ops_after=2`、`normalized_sha256=f68434247d4949b71c9215c5965abb00fb7b0ff7619ea0f69687ce0e8619685d`。

### 范围说明

- 阶段 E（5.1 叙事与对照）见下一节「阶段 E：survey 5.1 对照」及「阶段 E 实施与验证记录」。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 E — survey 5.1 对照

**日期**：2026-03-25

### Survey 5.1 要点 ↔ SemPatch 文件/状态

| Survey 5.1 要点 | SemPatch 文件 / 状态 |
|-----------------|----------------------|
| 多维语义：控制流图结构参与表示 | **已实现**：`src/utils/feature_extractors/fusion.py`（`multimodal.graph` 来自 CFG）、`src/utils/feature_extractors/graph_features.py` |
| 指令 / P-code 序列与跳转语义 | **已实现**：`src/utils/feature_extractors/sequence_features.py`；`multimodal.sequence` 含 `pcode_tokens`、`jump_mask` |
| 跨模态融合（图 ↔ 序列） | **已实现**：`src/features/models/multimodal_fusion.py`（图分支 + 序列分支 + `MultiheadAttention`） |
| 嵌入推理与 DAG `embed` 节点 | **已实现**：`src/features/inference.py`（`embed_batch` 优先 `MultiModalFusionModel`） |
| 数据流图 DFG 作为独立语义模态进入嵌入 | **已实现（阶段 H，2026-03-25）**：`multimodal.dfg`、`MultiModalFusionModel(use_dfg=True)`、`train_multimodal.py --use-dfg`；见 `docs/dfg_fusion_design.md`、`memory-bank/progress.md`「阶段 H 实施与验证记录」 |
| LSIR 中含 DFG 字段、提取器可接触 DFG 相关信息 | **已实现**：LSIR `dfg`；`extract_graph_features`；`fuse_features` → `multimodal.dfg`（可空图） |
| 仅序列侧基线（粗筛 / SAFE 族） | **已实现**：`src/features/baselines/safe.py`（`embed_batch_safe`）；`scripts/train_safe.py` |

### 范围说明

- **阶段 F**（第二基线 jTrans 等）未启动。
- 实施与验证记录见下节。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 E 实施与验证记录

**日期**：2026-03-25

### 交付物

| 项 | 说明 |
|----|------|
| [docs/design.md](../docs/design.md) | E.1：5.1 拆成已实现（CFG+序列+跨模态）与阶段 H（DFG）；节点说明标明 `multimodal.graph` 仅 CFG |
| [memory-bank/@design-document.md](@design-document.md) | E.1：§3.1 流程图脚注；E.2：§4.5 轻量消融命令表 |
| [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) | E.2：`#synthetic-short-path` 下步骤 **2b**（`train_safe` / `train_multimodal` 各 1 epoch） |
| [memory-bank/@architecture.md](@architecture.md) | E 后：§1.3a 5.1 相关文件角色（见该节） |

### 验证命令（维护者可复跑）

```bash
pytest -m "not ghidra"
PYTHONPATH=src python scripts/train_safe.py --synthetic --epochs 1
PYTHONPATH=src python scripts/train_multimodal.py --synthetic --epochs 1
```

### 验证结果（Agent 本机）

- **`pytest -m "not ghidra"`**（Python 3.12，`.venv`）：**106 passed**，1 skipped，1 deselected，**0 failed**。
- **`train_safe.py --synthetic --epochs 1 --no-tb`**：**exit 0**；末行摘要 `Epoch 1/1  train_loss=0.0043  val_loss=0.0003  val_acc=0.8000`；权重写入 `output/safe_best_model.pt`。
- **`train_multimodal.py --synthetic --epochs 1 --no-tb`**：**exit 0**；末行摘要 `Epoch 1/1  train_loss=0.0059  val_loss=0.0000  val_acc=0.8000`；权重写入 `output/best_model.pth`。

---

## CVE Demo / survey 对齐：`@prototype-survey-alignment-plan` 阶段 G（收尾与里程碑判定）

**日期**：2026-03-25

### G.1 / G.3 文档摘要

| 项 | 说明 |
|----|------|
| **G.1** | [memory-bank/@architecture.md](@architecture.md) 在「Demo（M1）固定路径与验收契约」之上新增 **「Demo 推荐数据流（M1）」**（G 当时文案曾写「模型不消费 DFG」；**阶段 H 后**以现版 `@architecture.md`「DFG 与阶段 H（已落地）」为准）；Mermaid、`EmbeddingItem.cve` 查表语义、架构洞察短表、A.1 文件名修正等同原文 |
| **G.3** | [memory-bank/@implementation-plan.md](@implementation-plan.md) 冷读：首段与「Demo 冻结决策」已互见 `@prototype-survey-alignment-plan.md` 且明确 **DFG 融入嵌入模型** 走阶段 H；步骤 11.4 已为阶段 H 语境；**未改该文件**（避免与第 5 行重复） |

### 双里程碑判定（G.2）

M1 前置阶段 A（含 A.5）–E 的完成证据见本文件各「CVE Demo / survey 对齐」小节；**本次 G 为文档同步 + pytest 门槛复验**。

| 里程碑 | 判定条件（摘自 alignment 计划 G.2） | 状态 / 日期 |
|--------|-------------------------------------|-------------|
| **M1（Demo）** | 阶段 A（含 A.5）、B 全部验证通过；允许 `pytest -m "not ghidra"` **0 failed**；验收证据为 exit 0 + 输出可解析 | **门槛已复验**：见下节「实施与验证记录」（2026-03-25） |
| **M2（DFG 融合）** | 阶段 H 全部通过且 `TODO.md` 对应项勾选 | **已达成**（2026-03-25）：见下方「阶段 H 实施与验证记录」 |

### 实施与验证记录

**JSON 与文档对照（G.1 验证抽样）**

- `benchmarks/smoke/fake_cve/library_embeddings.json`：含 `vector` 与非空 `cve`（`FAKE-CVE-*`），与 `@architecture.md` §4.2 / Demo 契约一致。
- `data/vulnerability_db/test_embeddings.json`：仓库大型样例，字段与 §4.2 一致（`cve` 可选）。
- `docs/DEMO.md`：`matches.json` / `report.md` 字段与 `@architecture.md`「Demo（M1）固定路径与验收契约」一致。

**命令与结果（Agent 本机，项目根 `.venv`，Python 3.12）**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not ghidra" -q
```

- **111 passed**，1 skipped，1 deselected，**0 failed**（约 8.8s）。

### 供后续开发者参考

- Demo 端到端数据流与「每一步对应哪个脚本/模块」以 `@architecture.md` **「Demo 推荐数据流」** 与紧随其后的 **「架构洞察：Demo 数据流步骤与关键文件映射」** 为准。
- **阶段 H** 已于 2026-03-25 落地；对外文档以 `@architecture.md`、`docs/design.md`、`docs/dfg_fusion_design.md` 为准。

---

## 阶段 H（DFG 多维语义融合）· H.1：DFG 表示与 LSIR 一致性审计

**日期**：2026-03-25

### CFG（`ir_builder`）

- **节点 ID**：`bb_{块索引}_{basic_block.start}`，与 Ghidra 导出的块顺序一致；**边**：相邻块 fall-through + 对块内最后若干条指令若助记符为 BRANCH/CBRANCH/CALL 等则停止扩展（跳转目标地址**未解析**，CFG 偏保守）。
- **与基本块**：一节点一块；不含指令级控制流细节。

### DFG（`ir_builder._extract_dfg_edges`）

- **节点**：字符串 `"{instruction_address}:{varnode_string}"`，varnode 为 P-code 的 `output` / `inputs` 原文（如 `(register, 0x0, 8)`）。
- **边**：对单条 P-code，若有 `output`，则对每个 `input` 建**有向边** `input_node → output_node`（**仅指令内** def-use，**不**跨指令传播 SSA 定义链）。
- **与基本块**：地址前缀可映射回所属指令及块；图在**函数级**合并所有块内指令后构建。

### 与 CFG 的对应关系

- 二者独立：CFG 块级、DFG 为带地址的 varnode 级；**无**自动的「块节点 ↔ DFG 节点」一对一 ID 对齐。

### 已知噪声与限制

- varnode 字符串随 **P-code 规范化**（`normalize_lsir_raw`）可能变化，DFG 拓扑随之变化。
- 单指令多 P-code、多输入会产生**局部扇入**；未做跨指令 use-def 链接时 DFG 较「碎」。
- **无 networkx** 时 `dfg` 为 `{"edges": [...]}` dict，与 `cfg` 的 dict 形态一致，特征提取已兼容。

### 验证

- 与 [@architecture.md](@architecture.md) §1.1（LSIRFunction 含 `cfg`/`dfg`）及阶段 H 实施后 §1.3a（DFG 进入 multimodal）对照一致。

---

## 阶段 H · H.2：融合形态设计评审

**日期**：2026-03-25

- **结论**：选定 **方案 A — 独立 DFG 图分支**，CFG 图嵌入与 DFG 图嵌入拼接后经线性层压回 `output_dim`，再进入现有跨模态注意力；异构图与块级拼接仅作文档对比，不实现。
- **设计说明**：[docs/dfg_fusion_design.md](../docs/dfg_fusion_design.md)
- **评审**：单人文档评审；范围受控为单一实现路径。

---

## 阶段 H 实施与验证记录（H.3–H.7 与回归）

**日期**：2026-03-25

### 代码与文档变更摘要

| 区域 | 路径 / 要点 |
|------|-------------|
| 特征 | `src/utils/feature_extractors/fusion.py`：`multimodal.dfg`、`_build_dfg_for_model`、`include_dfg`；CFG 边与 `sorted(node_list)` 对齐 |
| 模型 | `src/features/models/multimodal_fusion.py`：`use_dfg`、`graph_fuse`、`dfg_*` 分支；`_tensorize_multimodal` 返回 7 元组；`parse_multimodal_checkpoint` |
| 训练 | `scripts/train_multimodal.py`：`--use-dfg`、`--max-dfg-nodes`；`Trainer(checkpoint_meta=...)` → `{state_dict, meta}` |
| 合成数据 | `scripts/generate_synthetic_features.py`：`--with-dfg` |
| 推理 | `src/features/inference.py`：`embed_batch` 解析包装检查点 |
| 精排 / 两阶段 | `src/matcher/rerank.py`、`src/matcher/two_stage.py`：`use_dfg_model` / `rerank_use_dfg` |
| Demo | `scripts/demo_cve_match.py`：`--use-dfg-model` / `--no-use-dfg-model`，`config.rerank_use_dfg` |
| 测试 | `tests/test_features/test_multimodal_dfg.py`；`tests/test_features/test_feature_extractors.py` 扩展；`tests/test_train_reproducibility.py` 更新 |
| 对外文档 | `docs/DEMO.md`、`docs/design.md`、`memory-bank/@design-document.md`、`memory-bank/@architecture.md` |

### H.7 合成消融（验证集 loss，**无 Recall@K**：合成对无 ground_truth 检索）

在 **CPU/CUDA** 环境各跑 **1 epoch**，`batch_size=4`，`num_pairs=32`：

| 配置 | 命令要点 | Epoch 1 `val_loss`（日志末行摘要） |
|------|----------|--------------------------------------|
| 无 DFG 子模块 | `train_multimodal.py --synthetic --epochs 1 --no-use-dfg` | `val_loss=0.0000`（示例运行） |
| 含 DFG | `generate_synthetic_features.py --with-dfg` + `train_multimodal.py --synthetic-file … --use-dfg --save-path /tmp/best_dfg.pth` | `val_loss=0.0541`（示例运行） |

**说明**：二者**非同一随机种子与数据分布**下的严格消融；仅作「开/关不崩溃」与 loss 可测性记录。严格 Recall@K 需在 `eval_two_stage.py` + `ground_truth` 上另做实验。

### 回归命令

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not ghidra" -q
```

- **119 passed**，1 skipped，1 deselected，**0 failed**（阶段 H 合并后复验）。

### M2

- `TODO.md` Phase **2.4** 已全部勾选；**G.2** 表中 M2 更新为已达成（与本节互见）。

---

## 研究原型生产线落地（输入二进制 → CVE 报告）与 OOM 修复

**日期**：2026-03-25

### 本次目标

- 落地一个**单入口**执行脚本：从查询输入（二进制或特征）到 `matches.json` / `report.md`。
- 不调用任何训练脚本；缺失产物写入状态文件，供后续开发者继续。
- 修复大规模数据上的 OOM（`library_features.json` 16GB 级）导致流程无法完成的问题。

### 新增/修改文件

- `scripts/run_cve_pipeline.py`（新增）
  - 统一执行入口：`--query-binary` 或 `--query-features`。
  - 自动检查并可选补齐 `library_safe_embeddings.json`（仅推理构建，不训练）。
  - 输出 `pipeline_status.json`，显式记录 `missing` / `warnings` / `artifacts` / `mode`。
  - OOM 规避策略：
    1) TwoStage 前置文件体积阈值检查；
    2) TwoStage 失败时（含 OOM/非零退出）自动降级到 SAFE 粗筛报告路径；
    3) `query_features` 为超大 JSON 时拒绝整表加载并给出改用建议（`--query-binary` 或 JSONL）。
- `memory-bank/@architecture.md`（更新）
  - 在 scripts 表中补充 `run_cve_pipeline.py` 角色；
  - 增加「低内存执行洞察」说明 TwoStage 全量加载库特征的 OOM 风险与降级路径。

### 验证结果

1. **fixture 全链路（TwoStage）**
   - 命令：
     - `PYTHONPATH=src .venv/bin/python scripts/run_cve_pipeline.py --query-features benchmarks/smoke/fake_cve/query_features.json --library-emb benchmarks/smoke/fake_cve/library_embeddings.json --library-features benchmarks/smoke/fake_cve/library_features.json --output-dir output/prototype_pipeline_fixture_oomfix --max-queries 1 --top-k 2 --cpu`
   - 结果：
     - `exit 0`
     - 产物存在：`matches.json`、`report.md`、`pipeline_status.json`
     - `pipeline_status.json.ok = true`

2. **真实大文件目录防 OOM 预检**
   - 输入现状：
     - `data/two_stage/library_features.json` 约 **16GB**
     - `data/two_stage/library_safe_embeddings.json` 缺失
   - 命令：
     - `PYTHONPATH=src .venv/bin/python scripts/run_cve_pipeline.py --query-features benchmarks/smoke/fake_cve/query_features.json --output-dir output/prototype_pipeline_data_oomfix --cpu`
   - 结果：
     - 快速失败（非 OOM 崩溃），`exit 1`
     - 产出状态文件并显式标记缺失：
       - `library_safe_embeddings` 缺失
       - 原因：`library_features.json` 超大，直接构建存在 OOM 风险，需 JSONL 侧车或预先提供库嵌入

### 供后续开发者参考

- 首选执行：
  - `scripts/run_cve_pipeline.py`（而非直接手敲多段命令）。
- 若 `library_features.json` 为 GB 级：
  - 不要直接尝试全量 JSON 加载路径；
  - 优先准备/提供 `library_safe_embeddings.json`，或走 JSONL 侧车流程后再执行。
- 交付标准：
  - 无论成功或部分完成，均查看 `output/<run>/pipeline_status.json` 作为事实来源。
