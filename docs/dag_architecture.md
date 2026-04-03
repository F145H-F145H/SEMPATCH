# DAG 架构说明

DAG 作为纯执行引擎，不实现任何流水线。编排逻辑在 sempatch.py。

## 节点类型

- `ghidra`：Ghidra headless 分析
- `lsir_build`：构建 LSIR（CFG/DFG）
- `feature_extract`：特征提取
- `embed`：嵌入（占位）
- `load_db`：加载漏洞库（占位）
- `diff`：固件 vs 漏洞库匹配

## ctx 键约定

- `ghidra_output` / `lsir` / `features` / `embeddings` / `db_embeddings` / `diff_result`
