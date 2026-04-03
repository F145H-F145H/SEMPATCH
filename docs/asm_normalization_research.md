# 汇编归一化调研（survey 5.3 / 方案 B）

本文档支撑「架构中立表示」中 **P-code 默认路径** 与 **可选汇编归一化** 的分工，与 `memory-bank/@design-document.md` 第四节方案 B 一致。

## 1. 当前默认：Ghidra P-code

- **输入**：已由 `extract_lsir_raw` 导出 `lsir_raw`，指令级带 `pcode` 列表。
- **处理**：`utils/pcode_normalizer.py` 在构建 LSIR / 提取 multimodal 前默认执行（见 `memory-bank/@architecture.md` §1.1a）。
- **动机**：P-code 已在 ISA 之上抽象寄存器与内存访问模式，比原始机器码字节更适合跨架构对比学习；与 survey 5.3「架构中立」叙事一致。

## 2. 汇编级归一化（未默认开启）

| 路线 | 说明 | 与 SemPatch 的关系 |
|------|------|-------------------|
| 线性反汇编 + 规范化 | Capstone 等库输出助记符与操作数，可做立即数折叠、寄存器别名统一 | 设计文档中列为可选；**当前主链路不依赖** |
| VEX / 其它.lift | 另一套 IR，可与 P-code 并存但需独立 tokenizer 与训练契约 | 若引入，应单独定义 schema 与 `fuse_features` 输入，避免与 P-code 序列混用同一词表而无文档 |
| 文献中的「归一化汇编」 | 常针对 x86 变长指令与别名 | 可作为阶段 F/扩展评估的子课题，**不阻塞 M1 Demo** |

## 3. 为何优先 P-code 而非汇编

1. **单一事实来源**：仓库已从 Ghidra 统一导出 P-code，无需为每种 ISA 维护汇编解码差异。
2. **与现有多模态链一致**：`sequence` 特征来自 pcode token；切换汇编需重定义 `extract_sequence_features` 与词表构建。
3. **成本**：汇编归一化要覆盖 ARM/MIPS/x86 等 BinKit 维度时，测试与回归面显著大于扩展 `pcode_normalizer`。

## 4. 若后续引入汇编归一化的建议接口位置

- **实现落点**：新建 `utils/asm_normalizer.py`（或 `feature_extractors` 子模块），输入为 `LSIRInstruction.mnemonic` / `operands`（或原始字节流，若扩展 frontend）。
- **接入点**：仅在 `extract_sequence_features` 之前增加可选分支（配置开关），并在 `@architecture.md` 的 `multimodal.sequence` 说明中标注「token 源 = pcode | asm_norm」。
- **文档**：更新 `docs/DEMO.md` 与训练脚本帮助文本，明确「汇编模式」与 P-code 模式 **不得混用同一检查点**。

## 5. 结论

- **当前阶段**：以 P-code 规范化为默认、可测路径（`tests/test_pcode_normalizer.py`）。
- **汇编归一化**：保留为方案 B 的扩展项，落地前须完成词表、特征与评估契约的独立设计评审。
