# 两阶段数据划分说明

## 一、划分规则

将 BinKit 索引拆分为「函数库」与「查询集」，采用**按二进制随机划分**：

- **库侧**：80% 二进制，用作检索库
- **查询侧**：20% 二进制，用作查询集
- **随机种子**：默认 42，保证可复现

## 二、查询规模与调整逻辑

- **目标**：查询集函数数不少于 1000 个
- **「正样本充足」定义**：该查询函数在库中至少有 1 个同名函数（正样本）
- **仅统计**：仅将「正样本充足」的查询纳入查询集；若某查询在库中无同名函数，不计入有效查询

**调整逻辑**（当 20% 划分后有效查询数 < 1000 时）：

1. 依次尝试 25%、30%、35%、40%、45%、50% 作为查询侧比例
2. 取第一个使有效查询数 ≥ 1000 的比例
3. **上限**：查询侧比例不超过 50%
4. **若仍不足**：接受实际数量，脚本输出提示说明原因

## 三、统计量（seed=42, min-queries=1000）

| 指标 | 值 |
|------|------|
| 划分比例 | 库 75% / 查询 25%（20% 不足 1000，自动调整为 25%） |
| 库二进制数 | 38 |
| 库函数数 | 6805 |
| 查询二进制数 | 12 |
| 查询函数数（原始） | 2884 |
| 正样本充足查询数 | 1109 |

## 四、示例查询

| 查询 function_id | 在库中的正样本数量 |
|------------------|-------------------|
| `netifd.elf\|0x404000` (完整路径见 ground_truth.json) | 38 |
| `netifd.elf\|0x4044fa` | 1 |
| `netifd.elf\|0x4045e4` | 38 |
| `netifd.elf\|0x408bca` | 1 |
| `netifd.elf\|0x40c8ef` | 1 |

详见 `data/two_stage/ground_truth.json`。

## 五、输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| library_index.json | 与 binkit_functions 一致 | 库函数索引 |
| query_index.json | 与 binkit_functions 一致 | 查询集索引 |
| ground_truth.json | `{function_id: [positive_id, ...]}` | 查询→正样本映射 |

**function_id 格式**：`{binary_path}|{entry_address}`，如 `data/binkit_subset/addpart.elf|0x401000`。

## 六、推荐流水线顺序（含 pcode 长度过滤）

若需过滤低质量样本（`len(pcode_tokens) < 16`），推荐在 prepare 前执行：

```text
1. scripts/filter_index_by_pcode_len.py -i data/binkit_functions.json -o data/binkit_functions_filtered.json --min-pcode-len 16
2. scripts/prepare_two_stage_data.py --index-file data/binkit_functions_filtered.json
3. scripts/build_library_features.py --library-index data/two_stage/library_index.json --query-index data/two_stage/query_index.json
4. scripts/train_safe.py --index-file data/binkit_functions_filtered.json ...
```
