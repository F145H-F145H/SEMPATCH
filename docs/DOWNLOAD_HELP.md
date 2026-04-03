# 数据集下载与扩展指南

**磁盘预算**：100GB  
**扩展目标**：A（BinKit 预编译）、D（BinKit 自编译），确保训练集充足。自备漏洞 ELF 库见 **[VULNERABILITY_LIBRARY.md](VULNERABILITY_LIBRARY.md)**（不经过本子命令）。

**代理**：若需走代理（如公司网络、境内访问 Google Drive/GitHub），可使用 `--proxy http://127.0.0.1:7890` 或设置环境变量 `HTTP_PROXY`/`HTTPS_PROXY`。

**空间/数量限制**：`--max-total-size 20G` 限制复制或下载的总大小（如 20G、500M）；`copy-binkit` 的 `--target-count` 限制复制文件数量。

---

## 一、预算分配建议（100GB）

| 用途 | 预估空间 | 说明 |
|------|----------|------|
| BinKit 2.0 预编译（A） | 30-60GB | 建议仅下载/解压 x86_64 子集；全量约 80-150GB |
| 解压临时 | 80GB 峰值 | 解压需临时空间，完成后可删源包 |
| BinKit 自编译输出（D） | 5-20GB | 取决于编译包数量 |
| 现有 binkit_subset + 特征缓存 | ~5GB | 已占用 |
| 缓冲 | 5-10GB | 留给特征、嵌入、模型等 |

**策略**：优先 A；D 可与 A 共用部分二进制或作为补充。若空间紧张，可只做 A 的 x86_64 子集（约 20-40GB）。

---

## 二、A：BinKit 预编译数据集

### 2.1 获取方式

**Google Drive（官方）**
- BinKit 2.0：<https://drive.google.com/file/d/1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa/view?usp=share_link>
- 约 371,928 个二进制，8 架构 × 6 优化 × 23 编译器

### 2.2 命令行下载（不自动执行，需手动运行）

```bash
# 1. 安装 gdown
pip install gdown

# 2. 下载（约 80-150GB 解压后，建议先确保有空间）
mkdir -p data/downloads
gdown 1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa -O data/downloads/binkit2_dataset.7z

# 使用代理时:
# export https_proxy=http://127.0.0.1:7890 && gdown 1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa -O data/downloads/binkit2_dataset.7z

# 3. 解压（需安装 p7zip）
# Ubuntu: sudo apt install p7zip-full
7z x data/downloads/binkit2_dataset.7z -odata/downloads/binkit_extracted

# 4. 复制 x86_64 子集到 binkit_subset（控制规模以节省空间）
# 解压后目录结构通常为: 架构/编译器/优化/包名.elf
# 示例（按实际结构调整）:
find data/downloads/binkit_extracted -path "*x86_64*gcc*O2*" -name "*.elf" | head -200 | xargs -I{} cp {} data/binkit_subset/
# 或仅复制部分包目录:
# cp -r data/downloads/binkit_extracted/x86_64/gcc/O2/* data/binkit_subset/
```

### 2.3 使用本仓库脚本（仅打印帮助，不下载）

```bash
python scripts/expand_datasets.py binkit-precompiled --help
# 若要实际下载（需 gdown）:
python scripts/expand_datasets.py binkit-precompiled --confirm-download
# 使用代理:
python scripts/expand_datasets.py binkit-precompiled --confirm-download --proxy http://127.0.0.1:7890
# 限制下载大小（超出则中止并删除）:
python scripts/expand_datasets.py binkit-precompiled --confirm-download --max-total-size 20G
```

### 2.4 复制到 binkit_subset（本仓库脚本）

解压后，可用 `copy-binkit` 子命令将指定数量的 ELF 复制到 `data/binkit_subset/`：

```bash
python scripts/expand_datasets.py copy-binkit \
  --extract-dir data/downloads/binkit_extracted \
  --target-count 200 \
  --prefer-arch x86_64
# 同时限制总空间（二选一或组合使用）:
python scripts/expand_datasets.py copy-binkit --extract-dir data/downloads/binkit_extracted --target-count 300 --max-total-size 20G
```

### 2.5 扩展后重建索引

```bash
python scripts/build_binkit_index.py
python scripts/prepare_two_stage_data.py
python scripts/build_library_features.py --library-index data/two_stage/library_index.json --query-index data/two_stage/query_index.json
```

---

## 三、D：BinKit 自编译

### 3.1 前置条件

- 已克隆 BinKit 仓库：`python scripts/download_binkit.py`（可用 `--proxy http://127.0.0.1:7890`）
- 或手动：`git clone https://github.com/SoftSec-KAIST/BinKit data/binaries/binkit`（可用 `export https_proxy=...`）

### 3.2 依赖安装

```bash
cd data/binaries/binkit
./scripts/install_default_deps.sh   # 安装编译依赖
source scripts/env.sh
```

### 3.3 编译（二选一）

**快速：单包编译（如 BusyBox）**
```bash
./do_compile_busybox.sh
```

**完整：全量编译（耗时数小时至数天）**
```bash
./scripts/install_gnu_deps.sh
./compile_packages.sh
```

### 3.4 复制编译产物到 binkit_subset

编译输出通常在 `dataset/` 或 `gnu/` 下，按架构/编译器/优化组织。将需要的 ELF 复制到 `data/binkit_subset/` 后，再执行 `build_binkit_index.py`。

### 3.5 使用本仓库脚本

```bash
python scripts/expand_datasets.py binkit-compile
```

---

## 四、统一入口

```bash
# 列出所有扩展方式
python scripts/expand_datasets.py --list

# 各方式详细帮助
python scripts/expand_datasets.py binkit-precompiled --help
python scripts/expand_datasets.py binkit-compile

# 复制解压后的 BinKit 到 binkit_subset
python scripts/expand_datasets.py copy-binkit --extract-dir /path/to/extracted --target-count 200
```

---

## 五、训练集充足性检查

扩展完成后，建议满足：

- **binkit_subset**：≥ 100 二进制，≥ 15000 函数
- **正样本候选**：≥ 800 个同名函数（跨二进制）
- **两阶段**：有效查询 ≥ 1500

验证命令：

```bash
python -c "
import json
with open('data/binkit_functions.json') as f:
    d = json.load(f)
bins = len(d)
funcs = sum(len(x.get('functions',[])) for x in d)
from collections import defaultdict
n2s = defaultdict(list)
for it in d:
    for fn in it.get('functions',[]):
        if fn.get('name') and fn.get('entry'):
            n2s[fn['name']].append(1)
pos = sum(1 for s in n2s.values() if len(s)>=2)
print(f'binaries={bins}, functions={funcs}, positive_candidates={pos}')
print('OK' if bins>=100 and funcs>=15000 and pos>=800 else 'Need more data')
"
```
