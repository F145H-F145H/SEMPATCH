#!/usr/bin/env python3
"""
数据集扩展脚本：A（BinKit 预编译）、D（BinKit 自编译）、copy-binkit。

本脚本默认不执行任何下载，仅打印下载帮助与可用命令。
需显式传入 --confirm-download 且指定子命令时才执行实际下载。
支持 --proxy 设置代理（如 http://127.0.0.1:7890），也可通过环境变量 HTTP_PROXY/HTTPS_PROXY 配置。

漏洞库自建流程见 docs/VULNERABILITY_LIBRARY.md（不依赖任何单一外部基准包）。

用法:
  python scripts/expand_datasets.py --help
  python scripts/expand_datasets.py binkit-precompiled --help   # A
  python scripts/expand_datasets.py binkit-compile --help       # D
  python scripts/expand_datasets.py binkit-precompiled --confirm-download  # 实际下载 A
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINKIT_PRECOMPILED_URL = "https://drive.google.com/uc?id=1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa"
BINKIT_REPO = "https://github.com/SoftSec-KAIST/BinKit"
BINKIT_DIR = os.path.join(PROJECT_ROOT, "data", "binaries", "binkit")
BINKIT_SUBSET_DIR = os.path.join(PROJECT_ROOT, "data", "binkit_subset")


def _parse_size(s: str) -> int:
    """解析容量字符串（如 20G、500M、1T）为字节数。"""
    s = s.strip().upper()
    m = re.match(r"^([\d.]+)\s*([KMGTP])?B?$", s)
    if not m:
        raise ValueError(f"无法解析容量: {s}，期望格式如 20G、500M")
    val = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "B": 1}
    if unit == "B" and "B" not in s.upper().replace(" ", ""):
        mult["B"] = 1
    return int(val * mult.get(unit, 1))


def _print_binkit_precompiled_help() -> None:
    print("=" * 60)
    print("A: BinKit 预编译数据集 下载帮助")
    print("=" * 60)
    print()
    print("目标: 扩展训练集至 100-200+ 二进制，确保训练数据充足")
    print("预算: 约 30-60GB（仅 x86_64 子集），总数据集约 80-150GB")
    print()
    print("方式一: 使用 gdown（推荐）")
    print("  1. 安装: pip install gdown")
    print("  2. 下载到临时目录:")
    print("     gdown 1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa -O /tmp/binkit2_dataset.7z")
    print("     使用代理: export https_proxy=http://127.0.0.1:7890 && gdown ...")
    print("  3. 解压: 7z x /tmp/binkit2_dataset.7z -o/tmp/binkit_extracted")
    print("  4. 复制 x86_64 子集到 binkit_subset:")
    print("     解压后目录通常为 arch/compiler/opt/pkg.elf")
    print("     选取 x86_64/gcc/O2/ 等目录下 ELF，复制到 data/binkit_subset/")
    print()
    print("方式二: 浏览器手动下载")
    print("  1. 打开: https://drive.google.com/file/d/1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa/view")
    print("  2. 下载后解压，按方式一第 4 步复制")
    print()
    print("扩展后执行:")
    print("  python scripts/build_binkit_index.py")
    print("  python scripts/prepare_two_stage_data.py")
    print("  python scripts/build_library_features.py --library-index data/two_stage/library_index.json --query-index data/two_stage/query_index.json")
    print()


def _print_binkit_compile_help() -> None:
    print("=" * 60)
    print("D: BinKit 自编译 帮助")
    print("=" * 60)
    print()
    print("目标: 从 BinKit 源码编译得到可控规模的二进制")
    print("预算: 工具链+源码约 5-10GB，编译输出约 5-20GB")
    print("耗时: 单架构单包约 10-30 分钟，全量需数小时至数天")
    print()
    print("前置: 需已克隆 BinKit 仓库到 data/binaries/binkit/")
    print("  python scripts/download_binkit.py  # 若尚未克隆")
    print()
    print("步骤:")
    print("  1. cd data/binaries/binkit")
    print("  2. 安装依赖: ./scripts/install_default_deps.sh")
    print("  3. 配置: source scripts/env.sh")
    print("  4. 编译单个包（示例 BusyBox）:")
    print("     ./do_compile_busybox.sh")
    print("  5. 编译输出通常在 dataset/ 或 build/ 下，复制到 data/binkit_subset/")
    print()
    print("注意: 完整工具链搭建需执行 setup_ctng.sh、setup_gcc.sh 等，耗时较长")
    print("      若已有预编译 BinKit，建议优先使用方式 A")
    print()


def _apply_proxy(proxy: str | None) -> None:
    """设置代理环境变量，供 gdown/urllib 使用。"""
    if proxy:
        proxy = proxy.rstrip("/")
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy


def _run_binkit_precompiled(
    confirm: bool,
    dest: str,
    extract_to: str,
    proxy: str | None,
    max_download_size: int | None,
) -> bool:
    """下载 BinKit 预编译数据集。仅当 confirm=True 时执行。"""
    if not confirm:
        print("未传入 --confirm-download，跳过实际下载。仅打印帮助。", file=sys.stderr)
        _print_binkit_precompiled_help()
        return False

    try:
        import gdown
    except ImportError:
        print("错误: 需安装 gdown。运行: pip install gdown", file=sys.stderr)
        print("或手动下载: https://drive.google.com/file/d/1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa/view", file=sys.stderr)
        return False

    _apply_proxy(proxy)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"正在下载 BinKit 2.0 预编译数据集到 {dest} ...")
    if proxy:
        print(f"使用代理: {proxy}")
    if max_download_size:
        print(f"限制下载大小: {max_download_size / (1024**3):.1f} GB（超出将中止）")
    print("（文件较大，请确保磁盘空间充足，约 80-150GB 解压后）")

    if max_download_size:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import gdown; gdown.download(id='1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa', output={repr(dest)}, quiet=False, fuzzy=True)",
            ],
            env=os.environ,
        )
        try:
            while proc.poll() is None:
                time.sleep(3)
                if os.path.isfile(dest) and os.path.getsize(dest) > max_download_size:
                    proc.terminate()
                    proc.wait(timeout=10)
                    try:
                        os.remove(dest)
                    except OSError:
                        pass
                    print(
                        f"错误: 下载文件已超过限制 {max_download_size / (1024**3):.1f} GB，已中止并删除",
                        file=sys.stderr,
                    )
                    return False
            if proc.returncode != 0:
                return False
        except Exception:
            proc.kill()
            raise
    else:
        gdown.download(id="1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa", output=dest, quiet=False, fuzzy=True)

    if not os.path.isfile(dest):
        print("下载失败或文件不存在", file=sys.stderr)
        return False
    print(f"下载完成。请手动解压到 {extract_to} 后，将 x86_64 等子集复制到 data/binkit_subset/")
    return True


def _run_copy_binkit(
    extract_dir: str,
    target_count: int,
    prefer_arch: str,
    max_total_size: int | None,
) -> None:
    """从解压后的 BinKit 目录复制 ELF 到 binkit_subset，控制数量、架构与总大小。"""
    extract_dir = os.path.abspath(extract_dir)
    if not os.path.isdir(extract_dir):
        print(f"错误: 目录不存在 {extract_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(BINKIT_SUBSET_DIR, exist_ok=True)
    collected = []
    for root, _dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".elf"):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, extract_dir)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                collected.append((path, rel, size))
    pref = [p for p in collected if prefer_arch.lower() in p[1].lower()]
    rest = [p for p in collected if p not in pref]
    ordered = pref + rest

    to_copy = []
    total_size = 0
    for item in ordered:
        if len(to_copy) >= target_count:
            break
        path, rel, size = item[0], item[1], item[2]
        if max_total_size and total_size + size > max_total_size:
            continue
        to_copy.append(item)
        total_size += size

    for item in to_copy:
        src, rel, _ = item
        safe_name = rel.replace("/", "_").replace("\\", "_")
        if len(safe_name) > 200:
            safe_name = safe_name[-200:]
        dst = os.path.join(BINKIT_SUBSET_DIR, safe_name)
        if os.path.normpath(src) != os.path.normpath(dst):
            shutil.copy2(src, dst)
    print(f"已复制 {len(to_copy)} 个 ELF 到 {BINKIT_SUBSET_DIR}（总约 {total_size / (1024**3):.2f} GB）")
    if len(collected) > len(to_copy):
        reason = []
        if len(to_copy) >= target_count:
            reason.append("数量")
        if max_total_size and total_size >= max_total_size:
            reason.append("空间")
        print(f"（共发现 {len(collected)} 个，因{'/'.join(reason)}限制截断至 {len(to_copy)}）")


def _run_binkit_compile(confirm: bool) -> bool:
    """检查编译环境并打印编译命令。"""
    _print_binkit_compile_help()
    if os.path.isdir(BINKIT_DIR):
        print(f"BinKit 源码已存在于 {BINKIT_DIR}")
        if shutil.which("bash"):
            print("可执行: cd data/binaries/binkit && ./do_compile_busybox.sh")
    else:
        print(f"BinKit 源码不存在。请先运行: python scripts/download_binkit.py")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="数据集扩展：A(BinKit预编译) D(BinKit自编译) copy-binkit。默认不下载，仅打印帮助。"
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=["binkit-precompiled", "binkit-compile", "copy-binkit"],
        help="A=binkit-precompiled, D=binkit-compile, copy-binkit=从解压目录复制到 binkit_subset",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="确认执行下载（仅对 binkit-precompiled 有效）",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        metavar="URL",
        help="代理 URL，如 http://127.0.0.1:7890（也可用 HTTP_PROXY/HTTPS_PROXY 环境变量）",
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(PROJECT_ROOT, "data", "downloads", "binkit2_dataset.7z"),
        help="BinKit 预编译下载保存路径",
    )
    parser.add_argument(
        "--extract-to",
        default=os.path.join(PROJECT_ROOT, "data", "downloads", "binkit_extracted"),
        help="BinKit 解压目标目录（下载后需手动解压）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有扩展方式及帮助摘要",
    )
    # copy-binkit 专用
    parser.add_argument(
        "--extract-dir",
        default=os.path.join(PROJECT_ROOT, "data", "downloads", "binkit_extracted"),
        help="BinKit 解压后的根目录（copy-binkit 用）",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="复制到 binkit_subset 的 ELF 数量（copy-binkit 用）",
    )
    parser.add_argument(
        "--prefer-arch",
        default="x86_64",
        help="优先复制的架构名（copy-binkit 用）",
    )
    parser.add_argument(
        "--max-total-size",
        default=None,
        metavar="SIZE",
        help="copy-binkit: 复制总大小上限，如 20G、500M；binkit-precompiled: 下载文件大小上限，超出则中止",
    )
    args = parser.parse_args()

    max_size_bytes = None
    if args.max_total_size:
        try:
            max_size_bytes = _parse_size(args.max_total_size)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

    if args.list:
        print("数据集扩展方式（100GB 预算建议分配: BinKit~60–70GB, 编译输出~20GB, 缓冲~10GB）\n")
        print("  binkit-precompiled (A): 下载 BinKit 2.0 预编译，扩展训练规模")
        print("  binkit-compile (D): 从 BinKit 源码编译，可控规模")
        print("  copy-binkit: 从解压目录复制 ELF 到 binkit_subset（A 解压后使用）")
        print("  自备漏洞库: 见 docs/VULNERABILITY_LIBRARY.md（不经过本子命令）")
        print()
        print("详细帮助: python scripts/expand_datasets.py <subcommand> --help")
        print("完整文档: docs/DOWNLOAD_HELP.md")
        return

    if not args.subcommand:
        parser.print_help()
        print()
        print("示例:")
        print("  python scripts/expand_datasets.py --list")
        print("  python scripts/expand_datasets.py binkit-precompiled --help")
        print("  python scripts/expand_datasets.py binkit-precompiled --confirm-download")
        print("  python scripts/expand_datasets.py copy-binkit --extract-dir /path/to/binkit_extracted --target-count 200 --max-total-size 20G")
        return

    if args.subcommand == "binkit-precompiled":
        _run_binkit_precompiled(
            args.confirm_download,
            args.dest,
            args.extract_to,
            args.proxy,
            max_size_bytes,
        )
    elif args.subcommand == "binkit-compile":
        _run_binkit_compile(args.confirm_download)
    elif args.subcommand == "copy-binkit":
        _run_copy_binkit(
            args.extract_dir,
            args.target_count,
            args.prefer_arch,
            max_size_bytes,
        )


if __name__ == "__main__":
    main()
