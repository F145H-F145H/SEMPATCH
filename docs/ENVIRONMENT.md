# 环境快照

本文档记录 SemPatch Replain 的完整运行环境，用于复现和 Bug 报告。

## 系统

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.14.0-37-generic |
| Python | 3.12.3 |
| CUDA | 12.4 |
| GPU Driver | NVIDIA (CUDA 12.4 runtime) |

## Python 依赖

精确版本锁见 `requirements_frozen.txt`，安装方式：

```bash
pip install -r requirements_frozen.txt
```

核心依赖：

| 包 | 版本 |
|---|------|
| PyTorch | 2.6.0+cu124 |
| FAISS | 1.13.2 (faiss-cpu) |
| networkx | 3.6.1 |
| scipy | 1.17.1 |
| pytest | 9.0.2 |
| tensorboard | 2.20.0 |

## 外部工具

| 工具 | 版本 | 用途 |
|------|------|------|
| Ghidra | 12.0 build 20251205 | 二进制反编译，P-code IR 提取 |
| Java | 21 | Ghidra 运行依赖 |
| Binwalk | 系统包 | 固件解包（DAG 节点） |

## CUDA 组件（自动安装）

| 组件 | 版本 |
|------|------|
| nvidia-cublas-cu12 | 12.4.5.8 |
| nvidia-cuda-cupti-cu12 | 12.4.127 |
| nvidia-cudnn-cu12 | 9.1.0.70 |
| nvidia-cufft-cu12 | 11.2.1.3 |
| nvidia-nvjitlink-cu12 | 12.4.127 |
| triton | 3.2.0 |

## 复现步骤

```bash
git clone <repo> && cd SemPatch
./scripts/setup.sh               # Python + Ghidra + Binwalk
source .venv/bin/activate
make eval-smoke                  # 验证基本流程
make test-fast                   # 跳过 Ghidra 测试
```
