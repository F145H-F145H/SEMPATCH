# 数据集管理

本文档列出 SemPatch 使用的所有数据集、获取方式和用途。

## 项目内 Benchmark（已提交）

| 目录 | 用途 | 校验命令 |
|------|------|----------|
| `benchmarks/smoke/` | CI 冒烟测试（fake_cve + two_stage） | `make check-benchmarks` |
| `benchmarks/dev_binkit/` | 开发期评测（小规模 BinKit 数据） | `make check-benchmarks` |
| `benchmarks/real_cve/` | 真实 CVE 评测数据 | `make check-benchmarks` |

完整性校验：

```bash
make check-benchmarks          # 逐目录校验 SHA256
make check-manifest            # 统一 MANIFEST 校验
```

## 外部数据集

### BinKit

| 属性 | 值 |
|------|------|
| 来源 | 公开数据集 |
| 用途 | 大规模二进制函数相似度训练与评测 |
| 预期路径 | `data/binkit/` |
| 预处理 | 见 `scripts/` 目录下的预处理脚本 |

### Firmware

| 属性 | 值 |
|------|------|
| 来源 | 设备固件镜像 |
| 用途 | 真实固件 1-day 漏洞匹配 |
| 预期路径 | `data/firmware/` |
| 预处理 | binwalk 解包 → Ghidra 反编译 → LSIR 提取 |

### BinCodex

| 属性 | 值 |
|------|------|
| 来源 | 学术数据集 |
| 用途 | 跨编译器/跨优化级别评测 |
| 预期路径 | `data/bincodex/` |
| 预处理 | 见文档 |

## 添加新数据

1. 将数据放入对应的 `benchmarks/` 子目录
2. 更新对应的 `CHECKSUMS.sha256`（`cd benchmarks && sha256sum <file>`）
3. 更新 `benchmarks/MANIFEST.txt`（`make freeze` 或手动合并）
4. 提交 PR 并在 PR 描述中说明数据来源和变更原因
