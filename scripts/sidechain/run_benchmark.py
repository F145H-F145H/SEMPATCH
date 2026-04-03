#!/usr/bin/env python3
"""
运行 SemPatch 基准测试：快速冒烟、批量提取、compare 流水线。

用法:
  python scripts/run_benchmark.py --quick
  python scripts/run_benchmark.py --extract-dir data/unpacked/.../squashfs-root/bin
  python scripts/run_benchmark.py --compare /bin/true output/test_db_embeddings.json
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _find_elfs(path: str) -> list:
    bins = []
    for root, _, names in os.walk(path):
        for n in names:
            fp = os.path.join(root, n)
            if os.path.isfile(fp) and os.access(fp, os.R_OK):
                try:
                    with open(fp, "rb") as f:
                        if f.read(4) == b"\x7fELF":
                            bins.append(fp)
                except OSError:
                    pass
    return bins


def run_quick() -> dict:
    """快速冒烟：extract /bin/true + compare vs test_db。"""
    from sempatch import run_firmware_vs_db

    binary = "/bin/true"
    db = os.path.join(PROJECT_ROOT, "output", "test_db_embeddings.json")
    out = os.path.join(PROJECT_ROOT, "output", "benchmark_quick")
    if not os.path.isfile(db):
        return {"ok": False, "reason": f"test_db 不存在: {db}"}
    if not os.path.isfile(binary):
        return {"ok": False, "reason": f"二进制不存在: {binary}"}

    t0 = time.perf_counter()
    try:
        ctx = run_firmware_vs_db(binary, db, out, strategy="traditional_cfg")
        elapsed = time.perf_counter() - t0
        diff = ctx.get("diff_result", {})
        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 2),
            "diff_keys": list(diff.keys()) if isinstance(diff, dict) else [],
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def run_extract_dir(path: str, out_dir: str, limit: int) -> dict:
    """批量对目录中 ELF 执行 Ghidra 提取。"""
    from utils.ghidra_runner import run_ghidra_analysis

    elfs = _find_elfs(path)[:limit]
    if not elfs:
        return {"ok": False, "reason": f"未在 {path} 中发现 ELF"}

    os.makedirs(out_dir, exist_ok=True)
    ok_count = 0
    t0 = time.perf_counter()
    for i, fp in enumerate(elfs):
        sub = os.path.join(out_dir, f"bin_{i:04d}")
        os.makedirs(sub, exist_ok=True)
        try:
            run_ghidra_analysis(fp, sub, timeout=60)
            ok_count += 1
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return {
        "ok": True,
        "total": len(elfs),
        "success": ok_count,
        "elapsed_sec": round(elapsed, 2),
    }


def run_compare(binary: str, db_path: str, out_dir: str, strategy: str) -> dict:
    """运行 compare 流水线并返回摘要。"""
    from sempatch import run_firmware_vs_db

    t0 = time.perf_counter()
    try:
        ctx = run_firmware_vs_db(binary, db_path, out_dir, strategy=strategy)
        elapsed = time.perf_counter() - t0
        diff = ctx.get("diff_result", {})
        n_matches = 0
        if isinstance(diff, dict):
            if "matches" in diff:
                n_matches = len(diff["matches"])
            elif "pairs" in diff:
                n_matches = len(diff["pairs"])
        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 2),
            "matches": n_matches,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SemPatch 基准测试")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="快速冒烟测试")
    g.add_argument("--extract-dir", metavar="DIR", help="批量提取目录内 ELF")
    g.add_argument("--compare", nargs=2, metavar=("BINARY", "DB"), help="compare 流水线")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("--strategy", default="traditional_cfg", help="compare 策略")
    parser.add_argument("--limit", type=int, default=5, help="extract-dir 最多处理数量")
    args = parser.parse_args()

    out = args.output or os.path.join(PROJECT_ROOT, "output", "benchmark")
    os.makedirs(out, exist_ok=True)

    if args.quick:
        r = run_quick()
        print("quick:", r)
        sys.exit(0 if r.get("ok") else 1)

    if args.extract_dir:
        r = run_extract_dir(args.extract_dir, out, args.limit)
        print("extract:", r)
        sys.exit(0 if r.get("ok") else 1)

    if args.compare:
        binary, db = args.compare
        r = run_compare(binary, db, out, args.strategy)
        print("compare:", r)
        sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    main()
