# 大程序 / 大索引下的内存与 OOM

Ghidra 导出的 `lsir_raw` 会整份驻留在内存中（按二进制串行处理，但**单个超大二进制**仍可能占用数 GB）。函数级多进程会额外复制返回的 multimodal。以下分层缓解。

## 1. 脚本内选项（SemPatch）

### `filter_index_by_pcode_len.py`

| 选项 | 作用 |
|------|------|
| `--filtered-features-format jsonl`（默认） | 侧车流式写出，避免内存中攒全表 |
| `--workers 0` | 函数级提取改为主进程串行，降低多进程峰值 |
| `--process-pool-recycle-after-tasks N` | **与当前过滤脚本的 fork 池不兼容**（CPython 限制），会被忽略；保留参数仅作占位 |
| `--gc-after-each-binary` | 每个二进制结束后 `gc.collect()`，略降速 |
| `--max-memory-mb M` | 类 Unix 上尝试 `RLIMIT_AS` 限制虚拟地址空间约 M MiB（见下文限制） |

### `build_library_features.py`

| 选项 | 作用 |
|------|------|
| `--workers 1` | 二进制级仅单线程，避免多份 `lsir_raw` 同时常驻（默认多线程并行会放大峰值） |
| `--precomputed-multimodal *.jsonl` | 只读入当前索引需要的 `function_id`，减少加载侧车时的内存 |
| `--gc-after-each-binary` | 每个二进制任务完成后 `gc.collect()` |
| `--max-memory-mb M` | 同上，RLIMIT_AS |

### 环境变量

- `SEMPATCH_MAX_MEMORY_MB`：与 `--max-memory-mb` 相同含义；**命令行优先**。

## 2. RLIMIT_AS（`--max-memory-mb`）说明与风险

- 限制的是**进程虚拟地址空间**，不是精确的 RSS；设得过低会导致 `mmap` / 分配失败，进程可能 `MemoryError` 退出，但有利于**避免拖垮整台机器**。
- 在 **Linux** 上较常见；**macOS** 等行为可能不同或设置失败（脚本会打日志，不中断）。
- 建议比观测到的峰值 RSS **略宽裕**（例如 +20%～50%），否则易误杀；更稳妥可用下一节的 **cgroup**。

## 3. 系统级硬限制（推荐生产环境）

在 Linux 上用 **systemd** 或 **cgroups** 限制该次运行的最大内存，OOM 时只杀该服务，不影响桌面会话：

```bash
systemd-run --user --pty \
  -p MemoryMax=12G \
  -p MemorySwapMax=0 \
  python scripts/filter_index_by_pcode_len.py -i ... -o ...
```

或使用 `ulimit` 限制虚拟内存（与 RLIMIT_AS 类似，shell 级）：

```bash
ulimit -v $((14 * 1024 * 1024))   # 约 14GiB 虚拟内存上限（单位因 shell 而异，请查 man bash）
```

## 4. 架构层说明

当前流水线是 **「一二进制一整份 lsir_raw」**；单二进制含**极多函数**时，内存下界由该 JSON 结构大小决定。若仍不足，长期方案包括：按函数分片导出 lsir、或磁盘缓存按函数懒加载（需改 Ghidra 导出与缓存格式）。

更多数据流见 [filter_features_pipeline.md](filter_features_pipeline.md)。
