# Examples

Sample binaries and manifests for testing the pipeline.

## `mvp_library/`

Three compiled ELF binaries with distinct vulnerability patterns, used by the CVE library extractor and e2e tests.

| File | Vulnerability | CVE |
|------|---------------|-----|
| `mvp_vuln_01.c` | strcpy stack overflow | CVE-MVP-2024-0001 |
| `mvp_vuln_02.c` | unbounded loop write | CVE-MVP-2024-0002 |
| `mvp_vuln_03.c` | sprintf format string overflow | CVE-MVP-2024-0003 |

Build: `make -C examples/mvp_library`

## `mvp_vulnerable/`

Original MVP vulnerable binary for testing the match pipeline.

## `fake_cve_demo/`

Pre-computed features for testing without Ghidra. JSON smoke data lives under `benchmarks/smoke/fake_cve` (`pytest -m fake_cve` / `make eval-smoke`).
