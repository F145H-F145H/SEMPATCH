# 人造 `FAKE-CVE-*` 漏洞库（离线 JSON）

与 [`examples/fake_cve_demo/`](../../../examples/fake_cve_demo/) 中 `manifest.json` 使用同一伪 CVE 命名空间；本目录为**无 Ghidra**的最小库（`library_features.json` + `library_embeddings.json` + `query_features.json`），供 CI 与 `pytest -m fake_cve` 验收 `sempatch.py match` / `run_cve_match_pipeline`。
