// SPDX-License-Identifier: Apache-2.0
// SemPatch - LSIR Raw Extractor
//
// Stable version: handles CancelledException
//
// Author: SemPatch Project

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.block.*;
import ghidra.program.model.pcode.*;
import ghidra.program.model.address.Address;

import java.io.FileWriter;
import java.io.IOException;

public class extract_lsir_raw extends GhidraScript {

    private FileWriter writer;

    @Override
    public void run() throws Exception {

        String outputPath = getScriptArgs().length > 0
                ? getScriptArgs()[0]
                : "lsir_raw.json";

        println("[SemPatch] Exporting LSIR Raw to: " + outputPath);

        writer = new FileWriter(outputPath);
        writer.write("{\n  \"functions\": [\n");

        FunctionManager functionManager = currentProgram.getFunctionManager();
        BasicBlockModel bbModel = new BasicBlockModel(currentProgram);

        boolean firstFunction = true;

        for (Function function : functionManager.getFunctions(true)) {
            if (!firstFunction) {
                writer.write(",\n");
            }
            firstFunction = false;

            try {
                exportFunction(function, bbModel);
            } catch (ghidra.util.exception.CancelledException e) {
                println("[SemPatch WARNING] Function iteration cancelled: " + function.getName());
            } catch (Exception e) {
                println("[SemPatch WARNING] Exception exporting function " + function.getName() + ": " + e.getMessage());
            }
        }

        writer.write("\n  ]\n}");
        writer.flush();
        writer.close();

        println("[SemPatch] LSIR Raw export finished");
    }

    private void exportFunction(Function function, BasicBlockModel bbModel)
            throws IOException, ghidra.util.exception.CancelledException {

        writer.write("    {\n");
        writer.write("      \"name\": \"" + function.getName() + "\",\n");
        writer.write("      \"entry\": \"" + function.getEntryPoint() + "\",\n");
        writer.write("      \"basic_blocks\": [\n");

        CodeBlockIterator blocks;
        try {
            blocks = bbModel.getCodeBlocksContaining(function.getBody(), monitor);
        } catch (ghidra.util.exception.CancelledException e) {
            println("[SemPatch WARNING] getCodeBlocksContaining cancelled for function: " + function.getName());
            return;
        }

        boolean firstBlock = true;

        while (blocks.hasNext()) {
            CodeBlock block;
            try {
                block = blocks.next();
            } catch (ghidra.util.exception.CancelledException e) {
                println("[SemPatch WARNING] Block iteration cancelled in function: " + function.getName());
                continue; // skip this block
            }

            if (!firstBlock) {
                writer.write(",\n");
            }
            firstBlock = false;
            exportBasicBlock(block);
        }

        writer.write("\n      ]\n");
        writer.write("    }");
    }

    private void exportBasicBlock(CodeBlock block) throws IOException {

        writer.write("        {\n");
        writer.write("          \"start\": \"" + block.getFirstStartAddress() + "\",\n");
        writer.write("          \"end\": \"" + block.getMaxAddress() + "\",\n");
        writer.write("          \"instructions\": [\n");

        Listing listing = currentProgram.getListing();
        InstructionIterator instructions = listing.getInstructions(block, true);

        boolean firstInst = true;

        while (instructions.hasNext()) {
            Instruction inst = instructions.next();

            if (!firstInst) {
                writer.write(",\n");
            }
            firstInst = false;

            exportInstruction(inst);
        }

        writer.write("\n          ]\n");
        writer.write("        }");
    }

    private void exportInstruction(Instruction inst) throws IOException {

        writer.write("            {\n");
        writer.write("              \"address\": \"" + inst.getAddress() + "\",\n");
        writer.write("              \"mnemonic\": \"" + inst.getMnemonicString() + "\",\n");

        // Gather all operands as one string
        StringBuilder operands = new StringBuilder();
        for (int i = 0; i < inst.getNumOperands(); i++) {
            if (i > 0) operands.append(", ");
            operands.append(inst.getDefaultOperandRepresentation(i));
        }

        writer.write("              \"operands\": \"" + operands.toString().replace("\"", "\\\"") + "\",\n");
        writer.write("              \"pcode\": [\n");

        PcodeOp[] pcodeOps = inst.getPcode();
        for (int i = 0; i < pcodeOps.length; i++) {
            if (i > 0) {
                writer.write(",\n");
            }
            exportPcode(pcodeOps[i]);
        }

        writer.write("\n              ]\n");

        // Optional: source line and file from debug info (Ghidra SourceFileManager when available)
        int sourceLine = getSourceLineForAddress(inst.getAddress());
        String sourceFile = getSourceFileForAddress(inst.getAddress());
        if (sourceLine >= 0) {
            writer.write(",\n              \"source_line\": " + sourceLine);
        } else {
            writer.write(",\n              \"source_line\": null");
        }
        if (sourceFile != null && !sourceFile.isEmpty()) {
            writer.write(",\n              \"source_file\": \"" + sourceFile.replace("\\", "\\\\").replace("\"", "\\\"") + "\"");
        } else {
            writer.write(",\n              \"source_file\": null");
        }

        writer.write("\n            }");
    }

    /**
     * Get source line number for an address if debug info is available.
     * Uses SourceFileManager when present (Ghidra master/newer); otherwise returns -1.
     */
    private int getSourceLineForAddress(Address address) {
        try {
            java.lang.reflect.Method m = currentProgram.getClass().getMethod("getSourceFileManager");
            Object sfm = m.invoke(currentProgram);
            if (sfm == null) return -1;
            java.lang.reflect.Method getLine = sfm.getClass().getMethod("getLineNumber", Address.class);
            Object result = getLine.invoke(sfm, address);
            if (result instanceof Number) {
                int line = ((Number) result).intValue();
                return line > 0 ? line : -1;
            }
        } catch (Exception e) {
            // API not available or no debug info
        }
        return -1;
    }

    /**
     * Get source file path for an address if debug info is available.
     */
    private String getSourceFileForAddress(Address address) {
        try {
            java.lang.reflect.Method m = currentProgram.getClass().getMethod("getSourceFileManager");
            Object sfm = m.invoke(currentProgram);
            if (sfm == null) return null;
            java.lang.reflect.Method getFile = sfm.getClass().getMethod("getSourceFile", Address.class);
            Object result = getFile.invoke(sfm, address);
            return result != null ? result.toString() : null;
        } catch (Exception e) {
            return null;
        }
    }

    private void exportPcode(PcodeOp op) throws IOException {

        writer.write("                {\n");
        writer.write("                  \"opcode\": \"" + op.getMnemonic() + "\",\n");

        if (op.getOutput() != null) {
            writer.write("                  \"output\": \"" + op.getOutput().toString() + "\",\n");
        } else {
            writer.write("                  \"output\": null,\n");
        }

        writer.write("                  \"inputs\": [");
        Varnode[] inputs = op.getInputs();
        for (int i = 0; i < inputs.length; i++) {
            if (i > 0) {
                writer.write(", ");
            }
            writer.write("\"" + inputs[i].toString() + "\"");
        }
        writer.write("]\n");
        writer.write("                }");
    }
}
