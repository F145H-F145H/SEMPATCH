"""
Phase 2 预留：动态验证（符号执行/污点）后的成对标签侧车。

推荐 JSON 形状示例::
    {
      "version": 1,
      "pairs": [
        {
          "function_id_a": "path/a.elf|0x401000",
          "function_id_b": "path/b.elf|0x401000",
          "verified": true,
          "method": "concolic|manual|..."
        }
      ]
    }

当前训练管线不读取本模块；接入时在 PairwiseFunctionDataset 或独立脚本中解析即可。
"""

PAIR_LABELS_VERSION = 1
