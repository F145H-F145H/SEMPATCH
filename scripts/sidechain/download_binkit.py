#!/usr/bin/env python3
"""
下载 BinKit (SoftSec-KAIST BCSA Benchmark) 到 data/binaries/binkit/。

BinKit 提供源码和编译脚本，需本地编译生成 ELF 二进制用于算法评估。
预编译数据集需从论文/项目页获取，或自行运行 do_compile_*.sh。
支持 --proxy 设置代理，也可通过 HTTP_PROXY/HTTPS_PROXY 环境变量配置。

用法:
  python scripts/download_binkit.py              # 克隆仓库
  python scripts/download_binkit.py --proxy http://127.0.0.1:7890
  python scripts/download_binkit.py --list        # 列出获取方式
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINKIT_DIR = os.path.join(PROJECT_ROOT, "data", "binaries", "binkit")
BINKIT_REPO = "https://github.com/SoftSec-KAIST/BinKit"
BINKIT_ZIP = "https://github.com/SoftSec-KAIST/BinKit/archive/refs/heads/master.zip"


def _apply_proxy(proxy: str | None) -> None:
    """设置代理环境变量，供 git/urllib 使用。"""
    if proxy:
        proxy = proxy.rstrip("/")
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["ALL_PROXY"] = proxy
        os.environ["all_proxy"] = proxy


def clone_via_git(dest: str, proxy: str | None) -> bool:
    """使用 git clone 获取仓库。"""
    if shutil.which("git"):
        try:
            env = os.environ.copy()
            if proxy:
                env.update({
                    "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy,
                    "http_proxy": proxy, "https_proxy": proxy,
                })
            subprocess.run(
                ["git", "clone", "--depth", "1", BINKIT_REPO, dest],
                check=True,
                capture_output=True,
                timeout=120,
                env=env,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return False


def clone_via_zip(dest: str, proxy: str | None) -> bool:
    """下载 zip 并解压。"""
    if proxy:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy, "https": proxy,
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
    os.makedirs(dest, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        try:
            urllib.request.urlretrieve(BINKIT_ZIP, tmp.name)
            with zipfile.ZipFile(tmp.name, "r") as zf:
                # zip 根目录为 BinKit-master/
                names = zf.namelist()
                prefix = next((n for n in names if "/" in n), "").split("/")[0] + "/"
                for n in names:
                    if n.endswith("/"):
                        continue
                    data = zf.read(n)
                    rel = n[len(prefix) :] if n.startswith(prefix) else n
                    out = os.path.join(dest, rel)
                    os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(data)
            return True
        except Exception as e:
            print(f"  警告: zip 下载失败: {e}", file=sys.stderr)
            return False
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 BinKit BCSA Benchmark")
    parser.add_argument("--list", action="store_true", help="列出获取方式与编译说明")
    parser.add_argument("--out", default=BINKIT_DIR, help="输出目录")
    parser.add_argument(
        "--proxy",
        default=None,
        metavar="URL",
        help="代理 URL，如 http://127.0.0.1:7890（也可用 HTTP_PROXY/HTTPS_PROXY 环境变量）",
    )
    args = parser.parse_args()

    if args.list:
        print("=== BinKit (BCSA Benchmark) 获取方式 ===")
        print("1. 克隆仓库（本脚本）: python scripts/download_binkit.py")
        print("2. 编译二进制: 进入 data/binaries/binkit 后运行")
        print("   - ./do_compile_busybox.sh    # BusyBox 多配置")
        print("   - ./do_compile_coreutils_oldv.sh  # Coreutils")
        print("   - ./do_compile_openssl.sh    # OpenSSL")
        print("3. 预编译数据集 (BinKit 2.0):")
        print("   https://drive.google.com/file/d/1TrjFnv6BMpVEXYukVxrhlQ78S0NPKEXa/view")
        return

    out = os.path.abspath(args.out)
    if os.path.isdir(out) and any(
        f for f in os.listdir(out) if not f.startswith(".")
    ):
        print(f"已存在非空目录，跳过: {out}")
        return

    if os.path.exists(out):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    _apply_proxy(args.proxy)
    if args.proxy:
        print(f"使用代理: {args.proxy}")
    print("下载 BinKit ...")
    if clone_via_git(out, args.proxy):
        print(f"已克隆到 {out}")
    elif clone_via_zip(out, args.proxy):
        print(f"已解压到 {out}")
    else:
        print("错误: 无法获取 BinKit（请安装 git 或检查网络）", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
