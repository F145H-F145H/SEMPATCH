# 训练与测试数据集现状检查

**检查时间**：2025-03-17

---

## 一、当前数据集概况

### 1.1 训练数据（PairwiseFunctionDataset / PairwiseSyntheticDataset）

| 项目 | 当前值 | 说明 |
|------|--------|------|
| 数据源 | `data/binkit_functions.json` | 指向 `data/binkit_subset/` |
| 二进制数 | 50 | OpenWrt 相关 ELF |
| 函数总数 | 9689 | |
| 正样本候选（同名≥2处） | 600 个函数名 | 约 29417 对可能正对 |
| 架构/优化 | 单一（推测 x86_64 + 单一优化） | 缺乏跨优化/跨编译器多样性 |

### 1.2 测试数据（两阶段评估 eval_two_stage）

| 项目 | 当前值 | 说明 |
|------|--------|------|
| 库侧 | 38 二进制、6805 函数 | 来自 75% 划分 |
| 查询侧 | 12 二进制、2884 函数 | 来自 25% 划分 |
| 有效查询（正样本充足） | 1109 | ground_truth 中的查询数 |

### 1.3 其他数据源

| 数据源 | 状态 | 说明 |
|--------|------|------|
| 自备漏洞库 | 按需 | 将 ELF 置于自定义目录后 `build_library_binary_index.py` → 见 `docs/VULNERABILITY_LIBRARY.md` |
| BinKit 完整集 | 未使用 | `data/binaries/binkit/` 仅有源码与编译脚本，无预编译二进制 |

---

## 二、不足与风险

1. **规模偏小**：50 二进制 vs 设计文档中 BinKit 完整集约 37 万二进制。
2. **多样性不足**：单一架构、单一优化，难以评估跨优化（O0 vs O3）、跨编译器（gcc vs clang）。
3. **1-day / CVE 库评估**：依赖自备漏洞 ELF 与 CVE 映射；仅 .arrow 等表格无法直接走 Ghidra 链（见 `docs/VULNERABILITY_LIBRARY.md`）。
4. **正样本多样性**：600 个同名候选中，大量为 `_init`、`_fini`、`entry` 等通用桩，语义多样性有限。

---

## 三、扩展方式与手动确认

### 方式 A：扩展 BinKit 子集（推荐，提升训练规模）

**目标**：将 `binkit_subset` 从 50 增至 100–200 二进制，函数数约 2–4 万。

**步骤**（需您手动确认后执行）：

1. **获取更多预编译二进制**  
   - 使用 `python scripts/expand_datasets.py binkit-precompiled --help` 查看下载帮助  
   - 或直接访问 BinKit 2.0 预编译：<https://drive.google.com/file/d/1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa/view>  
   - 完整说明见 `docs/DOWNLOAD_HELP.md`

2. **下载并解压**（例如 x86_64 架构）后，使用脚本复制到 binkit_subset：
   ```bash
   # 使用 expand_datasets copy-binkit 子命令
   python scripts/expand_datasets.py copy-binkit --extract-dir /path/to/binkit_extracted --target-count 200 --prefer-arch x86_64
   ```

3. **重建索引并重新划分两阶段**：
   ```bash
   python scripts/build_binkit_index.py
   python scripts/prepare_two_stage_data.py
   python scripts/build_library_features.py --library-index data/two_stage/library_index.json --query-index data/two_stage/query_index.json
   ```

**请您确认**：是否已下载 BinKit 预编译数据？拟新增多少二进制（目标 100/150/200）？

---

### 方式 B：增加跨优化/跨编译器变体（提升评估多样性）

**目标**：在保持同源函数的前提下，引入不同优化（O0、O3）或编译器（gcc、clang）变体。

**步骤**（需您手动确认后执行）：

1. 从 BinKit 预编译集中选取**同源不同配置**的二进制，例如：
   - `x86_64/gcc/O0/` 与 `x86_64/gcc/O3/` 对比
   - `x86_64/gcc/O2/` 与 `x86_64/clang/O2/` 对比

2. 将选中的 ELF 复制到 `data/binkit_subset/`（或新建 `data/binkit_cross_opt/` 并相应修改索引路径）。

3. 重建索引与两阶段数据（同方式 A 第 3 步）。

**请您确认**：是否已获取包含多优化/多编译器的 BinKit 子集？打算采用哪些配置组合？

---

### 方式 C：自备漏洞二进制库（CVE / 1-day 向评估）

**目标**：在自有 ELF 集合上构建 `library_features.json` + `library_safe_embeddings.json`，并写入 `cve` 元数据。

**步骤**：见 **`docs/VULNERABILITY_LIBRARY.md`**（`build_library_binary_index.py` → `build_binkit_index --from-index-file` → `build_library_features` → `build_embeddings_db --features-file` → `annotate_library_embeddings_cve.py`）。

**请您确认**：是否已收集可 Ghidra 分析的二进制（非仅 .arrow/Parquet）？

---

### 方式 D：从 BinKit 源码编译（耗时较长）

**目标**：使用 `data/binaries/binkit/` 中的源码和脚本，自行编译得到更多二进制。

**步骤**（需您手动确认后执行）：

1. 运行 `python scripts/expand_datasets.py binkit-compile` 查看编译帮助
2. 进入 `data/binaries/binkit/`，安装依赖后执行 `./do_compile_busybox.sh` 等
3. 将编译产物复制到 `data/binkit_subset/`
4. 运行 `build_binkit_index.py` 与两阶段数据准备流程
5. 完整说明见 `docs/DOWNLOAD_HELP.md`

**请您确认**：是否打算采用自编译方式？对编译时间和磁盘空间（可能数 GB）是否可接受？

---

## 四、建议执行顺序

1. **优先**：方式 A，将二进制数扩展到约 100–150，以提升训练规模。
2. **其次**：方式 B，在扩展后的集中加入跨优化/跨编译器样本，用于挑战性评估。
3. **可选**：方式 C（自备漏洞库）、方式 D（自编译），按评估和资源需求决定。

---

## 五、确认清单

请在完成手动操作或做出决策后，勾选以下项（可复制到 issue 或回复中）：

- [ ] 方式 A：已下载 BinKit 预编译数据，拟新增 _____ 个二进制
- [ ] 方式 B：已获取多优化/多编译器 BinKit 子集，拟使用 _____ 种配置
- [ ] 方式 C：已准备漏洞 ELF 目录，将按 `docs/VULNERABILITY_LIBRARY.md` 建库
- [ ] 方式 D：拟从 BinKit 源码自行编译
- [ ] 暂不扩展，保持当前 50 二进制继续训练

提供确认后，可协助您编写具体的索引合并脚本、路径配置或数据准备流程调整。
