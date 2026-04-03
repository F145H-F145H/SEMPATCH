# API 参考

## sempatch.py

- `run_firmware_vs_db(binary_path, db_path, output_dir, **kwargs)`：固件 vs 漏洞库
- 子命令：`compare`、`extract`（传统 Ghidra）

## ctx 键

| 键 | 来源 | 说明 |
|----|------|------|
| ghidra_output | GhidraNode | lsir_raw in-memory dict |
| lsir | LSIRBuildNode | LSIR 结构 |
| features | FeatureExtractNode | 特征 |
| embeddings | EmbedNode | 向量 |
| db_embeddings | LoadDBNode | 漏洞库向量 |
| diff_result | DiffNode | 匹配结果 |

## builders

- `build_ghidra_node(dag, node_id, binary_path, output_dir, deps, ...)`
- `build_lsir_build_node(dag, node_id, deps, input_key, output_key)`
- `build_feature_extract_node`, `build_embed_node`, `build_load_db_node`, `build_diff_node`
