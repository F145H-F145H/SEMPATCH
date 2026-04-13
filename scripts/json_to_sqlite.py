#!/usr/bin/env python3
"""
将 library_features.json 转换为 SQLite 数据库（流式处理，内存友好）
用法: python scripts/preprocess/json_to_sqlite.py \
        --input data/two_stage/library_features.json \
        --output data/two_stage/library_features.db
"""
import argparse
import json
import sqlite3
import ijson
import os
from tqdm import tqdm

def convert_json_to_sqlite(input_path, db_path):
    # 连接数据库（若存在则覆盖）
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # 提升写入性能
    conn.execute("""CREATE TABLE IF NOT EXISTS features
                    (function_id TEXT PRIMARY KEY, features_json TEXT)""")
    conn.execute("DELETE FROM features")  # 清空旧数据

    # 判断 JSON 顶层类型：字典 or 数组
    with open(input_path, 'rb') as f:
        first_byte = f.read(1)
        f.seek(0)
        if first_byte == b'{':
            # 字典格式: {"func_id": {...}, ...}
            parser = ijson.kvitems(f, '')
            total = None  # 无法预知总数，不显示进度百分比
            print("检测到 JSON 字典格式，开始流式转换...")
            for func_id, value in tqdm(parser, desc="写入数据库", unit="条"):
                features_json = json.dumps(value, ensure_ascii=False)
                conn.execute(
                    "INSERT OR REPLACE INTO features VALUES (?, ?)",
                    (func_id, features_json)
                )
        elif first_byte == b'[':
            # 数组格式: [{"function_id": "...", ...}, ...]
            parser = ijson.items(f, 'item')
            print("检测到 JSON 数组格式，开始流式转换...")
            for obj in tqdm(parser, desc="写入数据库", unit="条"):
                func_id = obj.get('function_id') or obj.get('id')
                if not func_id:
                    print(f"警告: 跳过无 function_id 的对象: {obj}")
                    continue
                features_json = json.dumps(obj, ensure_ascii=False)
                conn.execute(
                    "INSERT OR REPLACE INTO features VALUES (?, ?)",
                    (func_id, features_json)
                )
        else:
            raise ValueError("无法识别的 JSON 格式（顶层不是对象也不是数组）")

    conn.commit()
    # 创建索引加速查询（可选）
    conn.execute("CREATE INDEX IF NOT EXISTS idx_function_id ON features(function_id)")
    conn.close()
    print(f"转换完成，数据库保存至 {db_path}")

def main():
    parser = argparse.ArgumentParser(description="将 library_features.json 转为 SQLite")
    parser.add_argument("--input", required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出的 SQLite 数据库路径")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    convert_json_to_sqlite(args.input, args.output)
    return 0

if __name__ == "__main__":
    exit(main())