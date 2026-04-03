"""测试 sempatch 编排逻辑。"""
import os
import sys

# 确保可导入 sempatch（项目根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def test_run_firmware_vs_db_exists():
    """run_firmware_vs_db 函数存在。"""
    import sempatch
    assert hasattr(sempatch, "run_firmware_vs_db")
    assert callable(sempatch.run_firmware_vs_db)
