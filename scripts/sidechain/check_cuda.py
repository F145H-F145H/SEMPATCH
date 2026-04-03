#!/usr/bin/env python3
"""Diagnostic script to check CUDA availability and performance bottlenecks."""

import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

print("=" * 60)
print("CUDA Diagnostic Report")
print("=" * 60)

# Check PyTorch and CUDA
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("   ⚠️  WARNING: CUDA is NOT available! Training will run on CPU.")
    print("   Possible reasons:")
    print("   - PyTorch CPU-only version installed")
    print("   - CUDA drivers not installed")
    print("   - GPU out of memory")
    print("   - CUDA_VISIBLE_DEVICES not set")

# Check device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n2. Selected Device: {device}")

# Test tensor on device
try:
    test_tensor = torch.randn(100, 100).to(device)
    result = test_tensor @ test_tensor.t()
    print(f"   ✓ Tensor operation successful on {device}")
except Exception as e:
    print(f"   ✗ Tensor operation failed: {e}")

# Check data loading bottleneck
print(f"\n3. Data Loading Configuration:")
print(f"   num_workers in DataLoader: 0 (main thread)")
print(f"   ⚠️  This can cause bottlenecks! Consider using num_workers=4 or higher.")

# Check if running the actual command would use CUDA
print(f"\n4. For your training command:")
print(f"   PYTHONPATH=src python scripts/train_safe.py \\")
print(f"     --index-file data/binkit_functions_50_filtered.json \\")
print(f"     --vocab-from-features data/two_stage_50/library_features.json \\")
print(f"     --data-dir data/two_stage_50 \\")
print(f"     --num-pairs 2000 \\")
print(f"     --epochs 10 \\")
print(f"     --batch-size 16 \\")
print(f"     --lr 1e-3 \\")
print(f"     --save-path output/safe_best_model_50.pt \\")
print(f"     --coarse-k 50 \\")
print(f"     --target-coarse-recall 0.30 \\")
print(f"     --target-recall-at-1 0.25")
print(f"\n   Will use device: {device}")

if not torch.cuda.is_available():
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS TO ENABLE CUDA:")
    print("=" * 60)
    print("1. Install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n2. Check NVIDIA driver:")
    print("   nvidia-smi")
    print("\n3. Set CUDA visible devices (if multiple GPUs):")
    print("   export CUDA_VISIBLE_DEVICES=0")
    print("\n4. Verify installation:")
    print("   python -c 'import torch; print(torch.cuda.is_available())'")

print(f"\n{'=' * 60}")