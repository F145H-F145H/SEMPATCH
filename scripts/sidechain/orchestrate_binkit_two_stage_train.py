#!/usr/bin/env python3
"""
从 BinKit 子集目录一键跑通：建索引 → pcode 过滤（默认排除 main/CRT）+ JSONL 侧车
→ （可选）两阶段划分与库/查询特征 → train_multimodal → train_safe。

子脚本均通过当前 Python 解释器以列表参数调用（不经 shell）。
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(cmd: list[str]) -> None:
    line = " ".join(shlex.quote(c) for c in cmd)
    print(f"+ {line}", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="binkit_subset → 过滤索引 + 侧车 → 可选两阶段数据 → 两阶段训练",
    )
    p.add_argument(
        "--input-dir",
        default=os.path.join(PROJECT_ROOT, "data", "binkit_subset"),
        help="BinKit 二进制目录（默认 data/binkit_subset）",
    )
    p.add_argument(
        "--work-dir",
        default=os.path.join(PROJECT_ROOT, "data", "binkit_two_stage_work"),
        help="工作目录：索引、过滤结果、侧车、可选 two_stage/ 子目录",
    )
    p.add_argument("--min-pcode-len", type=int, default=16)
    p.add_argument(
        "--index-workers",
        type=int,
        default=None,
        help="build_binkit_index --workers（默认脚本内置）",
    )
    p.add_argument(
        "--filter-workers",
        type=int,
        default=None,
        help="filter_index_by_pcode_len --workers",
    )
    p.add_argument(
        "--prepare-two-stage",
        action="store_true",
        help="额外运行 prepare_two_stage_data + build_library_features（输出 work-dir/two_stage/）",
    )
    p.add_argument(
        "--min-queries",
        type=int,
        default=1000,
        help="prepare_two_stage_data --min-queries",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="仅准备数据，不运行 train_multimodal / train_safe",
    )
    p.add_argument(
        "--stage1-path-contains",
        default=None,
        metavar="SUBSTR",
        help="若指定：先过滤索引子串再跑第一阶段 multimodal，全量索引第二阶段并 --init-weights（需非 skip-train）",
    )
    p.add_argument(
        "--stage1-epochs",
        type=int,
        default=10,
        help="第一阶段 train_multimodal epoch 数（与 --stage1-path-contains 联用）",
    )
    p.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="第二阶段 epoch 数（默认与 --multimodal-epochs 相同）",
    )
    p.add_argument("--multimodal-epochs", type=int, default=20)
    p.add_argument("--safe-epochs", type=int, default=10)
    p.add_argument("--multimodal-num-pairs", type=int, default=20000)
    p.add_argument("--safe-num-pairs", type=int, default=10000)
    p.add_argument("--multimodal-batch-size", type=int, default=8)
    p.add_argument(
        "--multimodal-num-workers",
        type=int,
        default=2,
        help="train_multimodal --num-workers（默认 2；内存紧张或调试时用 0）",
    )
    p.add_argument(
        "--multimodal-memory-cache-max-items",
        type=int,
        default=None,
        help="若指定则传入 train_multimodal --memory-cache-max-items",
    )
    p.add_argument("--safe-batch-size", type=int, default=8)
    p.add_argument(
        "--no-exclude-runtime-symbols",
        action="store_true",
        help="传给 filter_index_by_pcode_len：关闭 main/CRT 排除（与默认训练管线相反，仅用于对照）",
    )
    p.add_argument(
        "--features-workers",
        type=int,
        default=None,
        help="build_library_features --workers（仅 --prepare-two-stage 时）",
    )
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    input_dir = os.path.abspath(args.input_dir)

    py = sys.executable
    scr = os.path.join(PROJECT_ROOT, "scripts")

    index_json = os.path.join(work_dir, "binkit_functions.json")
    filtered_json = os.path.join(work_dir, "binkit_functions_filtered.json")
    features_jsonl = os.path.join(work_dir, "filtered_features.jsonl")
    two_stage_dir = os.path.join(work_dir, "two_stage")

    binkit_cmd = [
        py,
        os.path.join(scr, "build_binkit_index.py"),
        "--input-dir",
        input_dir,
        "-o",
        index_json,
    ]
    if args.index_workers is not None:
        binkit_cmd.extend(["--workers", str(args.index_workers)])
    # _run(binkit_cmd)

    filt_cmd = [
        py,
        os.path.join(scr, "filter_index_by_pcode_len.py"),
        "-i",
        index_json,
        "-o",
        filtered_json,
        "--filtered-features-output",
        features_jsonl,
        "--min-pcode-len",
        str(args.min_pcode_len),
        "--project-root",
        PROJECT_ROOT,
    ]
    if args.filter_workers is not None:
        filt_cmd.extend(["--workers", str(args.filter_workers)])
    else:
        filt_cmd.extend(["--workers", str(12)])

    if args.no_exclude_runtime_symbols:
        filt_cmd.append("--no-exclude-runtime-symbols")
    # _run(filt_cmd)

    if args.prepare_two_stage:
        _run(
            [
                py,
                os.path.join(scr, "prepare_two_stage_data.py"),
                "--index-file",
                filtered_json,
                "--output-dir",
                two_stage_dir,
                "--min-queries",
                str(args.min_queries),
            ]
        )
        feat_cmd = [
            py,
            os.path.join(scr, "build_library_features.py"),
            "--library-index",
            os.path.join(two_stage_dir, "library_index.json"),
            "--query-index",
            os.path.join(two_stage_dir, "query_index.json"),
            "--output-dir",
            two_stage_dir,
            "--precomputed-multimodal",
            features_jsonl,
        ]
        if args.features_workers is not None:
            feat_cmd.extend(["--workers", str(args.features_workers)])
    #    _run(feat_cmd)

    if args.skip_train:
        print("已跳过训练（--skip-train）。产物:", filtered_json, features_jsonl, flush=True)
        return

    mm_base = [
        py,
        os.path.join(scr, "train_multimodal.py"),
        "--precomputed-features",
        features_jsonl,
        "--num-pairs",
        str(args.multimodal_num_pairs),
        "--batch-size",
        str(args.multimodal_batch_size),
        "--pairing-mode",
        "binkit_refined",
    ]
    mm_perf: list[str] = ["--num-workers", str(args.multimodal_num_workers)]
    if args.multimodal_memory_cache_max_items is not None:
        mm_perf.extend(
            ["--memory-cache-max-items", str(args.multimodal_memory_cache_max_items)]
        )

    save_mm = os.path.join(PROJECT_ROOT, "output", "best_model.pth")
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else args.multimodal_epochs

    if args.stage1_path_contains:
        stage1_index = os.path.join(work_dir, "binkit_stage1_index.json")
        _run(
            [
                py,
                os.path.join(scr, "filter_binkit_index_subset.py"),
                "-i",
                filtered_json,
                "-o",
                stage1_index,
                "--path-contains",
                args.stage1_path_contains,
            ]
        )
        _run(
            mm_base + mm_perf
            + [
                "--index-file",
                stage1_index,
                "--epochs",
                str(args.stage1_epochs),
                "--save-path",
                save_mm,
            ]
        )
        _run(
            mm_base + mm_perf
            + [
                "--index-file",
                filtered_json,
                "--epochs",
                str(stage2_epochs),
                "--init-weights",
                save_mm,
                "--save-path",
                save_mm,
            ]
            + (
                [
                    "--retrieval-val-dir",
                    two_stage_dir,
                ]
                if args.prepare_two_stage
                else []
            )
        )
    else:
        mm_cmd = mm_base + mm_perf + [
            "--index-file",
            filtered_json,
            "--epochs",
            str(args.multimodal_epochs),
            "--save-path",
            save_mm,
        ]
        if args.prepare_two_stage:
            mm_cmd.extend(["--retrieval-val-dir", two_stage_dir])
        _run(mm_cmd)

    safe_cmd = [
        py,
        os.path.join(scr, "train_safe.py"),
        "--index-file",
        filtered_json,
        "--vocab-from-features",
        features_jsonl,
        "--precomputed-features",
        features_jsonl,
        "--epochs",
        str(args.safe_epochs),
        "--num-pairs",
        str(args.safe_num_pairs),
        "--batch-size",
        str(args.safe_batch_size),
    ]
    if args.prepare_two_stage:
        safe_cmd.extend(["--data-dir", two_stage_dir])
    else:
        safe_cmd.append("--skip-validation")
    _run(safe_cmd)

    print("完成。精排权重: output/best_model.pth ，SAFE: output/safe_best_model.pt", flush=True)


if __name__ == "__main__":
    main()
