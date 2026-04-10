# 评测基准目录（固定契约）

三套目录对应三把「尺子」，路径与语义不要随意改；变更划分、清单或 smoke 数据应走 PR 并更新 CHECKSUMS。

| 目录 | 用途 | 大小 | make 命令 |
|------|------|------|-----------|
| [`smoke/`](smoke/) | Commit 前快速验证 | < 5 KB | `make eval-smoke` |
| [`dev_binkit/`](dev_binkit/) | 调参唯一对比组（固定 seed=42） | ~ 26 KB | `make eval-dev` |
| [`real_cve/`](real_cve/) | 最终价值判断（CVE 交叉引用） | ~ 11 KB（示例） | `make eval-real` |

## 快速使用

```bash
source .venv/bin/activate

# 冒烟测试（无外部依赖）
make eval-smoke

# 开发评测（固定 BinKit 子集）
make eval-dev

# 真实 CVE 评测（需要嵌入文件）
make eval-real
```

## 完整性校验

```bash
# 校验所有基准数据
make check-benchmarks

# 或单独校验
sha256sum -c benchmarks/smoke/CHECKSUMS.sha256
sha256sum -c benchmarks/dev_binkit/CHECKSUMS.sha256
sha256sum -c benchmarks/real_cve/CHECKSUMS.sha256
```

## 修改规则

1. **禁止随意修改** — 任何对 `smoke/`、`dev_binkit/split/`、`real_cve/example_data/` 下 JSON 文件的修改必须走 PR
2. **更新 CHECKSUMS** — 修改数据后须重新生成对应的 `CHECKSUMS.sha256`
3. **记录变更原因** — PR 中说明为何需要修改（如：特征提取格式变更、ground truth 修正）
4. **保持小体积** — 提交的 fixture 总大小不超过 100 KB；大体积数据通过脚本本地生成并 gitignore

详见各子目录 README。
