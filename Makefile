# 评测命令封装：固定路径与默认 K，减少手敲长参数误差。
# 使用前：source .venv/bin/activate（或确保 PYTHON 指向带依赖的解释器）。

.PHONY: eval-smoke eval-dev eval-real eval-all check-benchmarks check-manifest test test-fast freeze env-info reproduce

PYTHON ?= .venv/bin/python
PY := PYTHONPATH=src $(PYTHON)

EVAL_DEV_DATA := benchmarks/dev_binkit/split

eval-smoke:
	$(PYTHON) -m pytest -m fake_cve

eval-dev:
	@test -f $(EVAL_DEV_DATA)/ground_truth.json || (echo "缺少 $(EVAL_DEV_DATA)/ground_truth.json，见 benchmarks/dev_binkit/split/README.md"; exit 1)
	@test -f $(EVAL_DEV_DATA)/query_features.json || (echo "缺少 $(EVAL_DEV_DATA)/query_features.json"; exit 1)
	@test -f $(EVAL_DEV_DATA)/library_features.json || (echo "缺少 $(EVAL_DEV_DATA)/library_features.json"; exit 1)
	@test -f $(EVAL_DEV_DATA)/library_safe_embeddings.json || (echo "缺少 $(EVAL_DEV_DATA)/library_safe_embeddings.json"; exit 1)
	@mkdir -p output/benchmarks
	$(PY) scripts/eval_two_stage.py \
		--data-dir $(EVAL_DEV_DATA) \
		--coarse-k 100 \
		-k 1 5 10 \
		--output output/benchmarks/dev_eval.json

eval-real:
	@test -f benchmarks/real_cve/query_embeddings.json || (echo "缺少 benchmarks/real_cve/query_embeddings.json，见 benchmarks/real_cve/README.md"; exit 1)
	@test -f benchmarks/real_cve/library_embeddings.json || (echo "缺少 benchmarks/real_cve/library_embeddings.json"; exit 1)
	@mkdir -p output/benchmarks
	$(PY) scripts/eval_bcsd.py \
		--firmware-emb benchmarks/real_cve/query_embeddings.json \
		--db-emb benchmarks/real_cve/library_embeddings.json \
		--mode cve \
		-k 1 5 10 \
		--output output/benchmarks/real_eval.json

eval-all: eval-smoke eval-dev eval-real

check-benchmarks:
	@echo "=== smoke ==="
	cd benchmarks && sha256sum -c smoke/CHECKSUMS.sha256
	@echo "=== dev_binkit ==="
	cd benchmarks && sha256sum -c dev_binkit/CHECKSUMS.sha256
	@echo "=== real_cve ==="
	cd benchmarks && sha256sum -c real_cve/CHECKSUMS.sha256
	@echo "All benchmark checksums OK."

check-manifest:
	cd benchmarks && sha256sum -c MANIFEST.txt
	@echo "MANIFEST checksums OK."

test:
	$(PYTHON) -m pytest tests/ -x -q

test-fast:
	$(PYTHON) -m pytest tests/ -x -q -m "not ghidra"

freeze:
	$(PYTHON) -m pip freeze > requirements_frozen.txt
	@echo "已生成 requirements_frozen.txt"

env-info:
	@echo "=== Python ==="
	$(PYTHON) --version
	@echo "=== PyTorch ==="
	$(PYTHON) -c "import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')"
	@echo "=== FAISS ==="
	$(PYTHON) -c "import faiss; print(f'FAISS {faiss.__version__}')" 2>/dev/null || echo "(not installed)"
	@echo "=== CUDA ==="
	nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
	@echo "=== Ghidra ==="
	test -f third_party/ghidra_12.0_PUBLIC/support/analyzeHeadless && echo "Ghidra 12.0 (installed)" || echo "(not installed)"
	@echo "=== Binwalk ==="
	binwalk --version 2>/dev/null || echo "(not installed)"

reproduce: check-manifest test-fast
	$(PY) scripts/sidechain/train_multimodal.py --synthetic --epochs 2 --seed 42 --no-tb
	$(PY) scripts/sidechain/train_safe.py --synthetic --epochs 1 --seed 145 --no-tb --skip-validation
	$(PYTHON) -m pytest -m fake_cve -x -q
	@echo "=== 复现完成 ==="
