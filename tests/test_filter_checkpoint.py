"""utils.filter_checkpoint 单元测试。"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.filter_checkpoint import (  # noqa: E402
    CHECKPOINT_VERSION,
    build_default_checkpoint_path,
    compute_file_sha256,
    load_checkpoint,
    save_checkpoint_atomic,
    validate_checkpoint_meta,
)


def test_build_default_checkpoint_path():
    p = build_default_checkpoint_path("/tmp/out.json")
    assert p.endswith("/tmp/out.json.filter_checkpoint.json")


def test_save_load_and_validate_checkpoint(tmp_path):
    input_file = tmp_path / "index.json"
    input_file.write_text('[{"binary":"a","functions":[]}]', encoding="utf-8")
    sha = compute_file_sha256(str(input_file))
    meta = {
        "input_path": str(input_file),
        "input_sha256": sha,
        "project_root": "/tmp/proj",
        "min_pcode_len": 16,
        "filtered_features_output": "/tmp/features.jsonl",
        "filtered_features_format": "jsonl",
    }
    payload = {
        "version": CHECKPOINT_VERSION,
        "meta": meta,
        "completed_binaries": ["/tmp/proj/bin1"],
        "slots": [None, {"binary": "a", "functions": []}],
        "counters": {"total_original": 10, "total_kept": 6, "binaries_dropped": 1},
    }
    ckpt = tmp_path / "f.ckpt.json"
    save_checkpoint_atomic(str(ckpt), payload)
    assert ckpt.is_file()

    loaded = load_checkpoint(str(ckpt))
    ok, msg = validate_checkpoint_meta(loaded, meta)
    assert ok is True
    assert msg == "ok"

    bad_meta = dict(meta)
    bad_meta["min_pcode_len"] = 32
    ok2, msg2 = validate_checkpoint_meta(loaded, bad_meta)
    assert ok2 is False
    assert "min_pcode_len" in msg2


def test_save_checkpoint_is_valid_json_after_replace(tmp_path):
    ckpt = tmp_path / "replace.ckpt.json"
    payload = {"version": CHECKPOINT_VERSION, "meta": {}, "completed_binaries": [], "slots": [], "counters": {}}
    save_checkpoint_atomic(str(ckpt), payload)
    raw = json.loads(ckpt.read_text(encoding="utf-8"))
    assert raw["version"] == CHECKPOINT_VERSION
