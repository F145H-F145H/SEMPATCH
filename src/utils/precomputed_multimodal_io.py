"""
预计算 multimodal 侧车文件读写约定。

- JSONL（推荐）：每行一个对象 {"function_id": "...", "multimodal": {...}}，可流式扫描。
- JSON（兼容）：整文件为 {function_id: multimodal, ...}。
- 大侧车：build_jsonl_sidecar_lazy_index 只保留 (字节偏移, 行长)，按 function_id 再读行解析，避免整表进内存。
"""
from __future__ import annotations

import json
import os
from typing import Any, BinaryIO, Dict, Iterable, Iterator, Optional, Set, Tuple


def _skip_json_string(line: bytes, i: int) -> Optional[int]:
    if i >= len(line) or line[i] != 34:  # "
        return None
    i += 1
    while i < len(line):
        c = line[i]
        if c == 34:  # "
            return i + 1
        if c == 92:  # \
            i += 2
            continue
        i += 1
    return None


def _skip_json_value(line: bytes, i: int) -> Optional[int]:
    if i >= len(line):
        return None
    c = line[i]
    if c == 34:  # "
        return _skip_json_string(line, i)
    if c in b"-0123456789":
        j = i + 1
        while j < len(line) and line[j] in b"0123456789+-.eE":
            j += 1
        return j
    if line.startswith(b"true", i):
        return i + 4
    if line.startswith(b"false", i):
        return i + 5
    if line.startswith(b"null", i):
        return i + 4
    if c not in (91, 123):  # [ or {
        return None

    stack = [93 if c == 91 else 125]  # ] or }
    j = i + 1
    while j < len(line):
        ch = line[j]
        if ch == 34:  # "
            nxt = _skip_json_string(line, j)
            if nxt is None:
                return None
            j = nxt
            continue
        if ch == 91:  # [
            stack.append(93)
        elif ch == 123:  # {
            stack.append(125)
        elif ch == stack[-1]:
            stack.pop()
            if not stack:
                return j + 1
        j += 1
    return None


def _extract_function_id_from_jsonl_line_bytes(line: bytes) -> Optional[str]:
    """
    仅扫描 JSONL 单行前缀字段，尽量在不反序列化 multimodal 的前提下抽取 function_id。
    若不满足预期结构，返回 None 交由上层回退 json.loads。
    """
    n = len(line)
    i = 0
    while i < n and line[i] in b" \t\r\n":
        i += 1
    if i >= n or line[i] != 123:  # {
        return None
    i += 1
    while i < n:
        while i < n and line[i] in b" \t\r\n":
            i += 1
        if i >= n:
            return None
        if line[i] == 125:  # }
            return None
        key_start = i
        key_end = _skip_json_string(line, key_start)
        if key_end is None:
            return None
        try:
            key = json.loads(line[key_start:key_end].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        i = key_end
        while i < n and line[i] in b" \t\r\n":
            i += 1
        if i >= n or line[i] != 58:  # :
            return None
        i += 1
        while i < n and line[i] in b" \t\r\n":
            i += 1
        if i >= n:
            return None

        if key == "function_id":
            val_start = i
            val_end = _skip_json_string(line, val_start)
            if val_end is None:
                return None
            try:
                val = json.loads(line[val_start:val_end].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
            return val if isinstance(val, str) else None

        # 按既有写出格式，multimodal 位于 function_id 之后；若提前遇到，交给 fallback。
        if key == "multimodal":
            return None

        val_end = _skip_json_value(line, i)
        if val_end is None:
            return None
        i = val_end
        while i < n and line[i] in b" \t\r\n":
            i += 1
        if i < n and line[i] == 44:  # ,
            i += 1
            continue
        if i < n and line[i] == 125:  # }
            return None
        return None
    return None


def _parse_jsonl_record(obj: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(obj, dict):
        return None
    fid = obj.get("function_id")
    mm = obj.get("multimodal")
    if isinstance(fid, str) and isinstance(mm, dict):
        return fid, mm
    return None


def iter_jsonl_sidecar(path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            rec = _parse_jsonl_record(obj)
            if rec is not None:
                yield rec


def is_jsonl_sidecar_path(path: str) -> bool:
    p = path.lower()
    if p.endswith(".jsonl"):
        return True
    if p.endswith(".json"):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    return False
                return _parse_jsonl_record(obj) is not None
    except OSError:
        return False
    return False


class JsonlSidecarLazyIndex:
    """
    JSONL 侧车懒加载：构造时单遍扫描只记录 needed_ids 对应行的字节偏移与行长，
    get(function_id) 时再读盘 json 解析该行。适合 GB 级侧车 + 百万级函数，避免 OOM。

    bulk_get：对一批 function_id 按文件偏移排序后**单次打开、顺序 read**，避免多线程
    每函数 random seek + 反复 open 造成页缓存抖动与解析峰值叠加（build_library_features 主路径）。
    """

    __slots__ = ("path", "_index", "_read_lock", "_rb_fp", "_reuse_read_file_handle")

    def __init__(
        self,
        path: str,
        index: Dict[str, Tuple[int, int]],
        *,
        read_lock: Optional[Any] = None,
        reuse_read_file_handle: bool = True,
    ) -> None:
        self.path = os.path.abspath(path)
        self._index = index
        self._read_lock = read_lock
        self._rb_fp: Optional[BinaryIO] = None
        self._reuse_read_file_handle = bool(reuse_read_file_handle)

    def __len__(self) -> int:
        return len(self._index)

    def close_read_handle(self) -> None:
        """关闭懒复用的二进制读句柄（可选，长进程结束时释放 FD）。"""
        fp = self._rb_fp
        self._rb_fp = None
        if fp is not None:
            try:
                fp.close()
            except OSError:
                pass

    def _read_one(self, function_id: str, offset: int, length: int) -> Optional[Dict[str, Any]]:
        def _parse(raw: bytes) -> Optional[Dict[str, Any]]:
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            rec = _parse_jsonl_record(obj)
            if rec is None or rec[0] != function_id:
                return None
            return rec[1]

        try:
            if self._reuse_read_file_handle:
                if self._rb_fp is None:
                    self._rb_fp = open(self.path, "rb")
                self._rb_fp.seek(offset)
                raw = self._rb_fp.read(length)
                return _parse(raw)
            with open(self.path, "rb") as f:
                f.seek(offset)
                raw = f.read(length)
                return _parse(raw)
        except OSError:
            return None

    def bulk_get(self, function_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量读取：按 (offset) 排序后顺序读行，减少磁盘寻道与并发随机读。
        未出现在索引中的 id 不会出现在返回 dict 中。
        """
        triples: list[Tuple[int, int, str]] = []
        for fid in function_ids:
            pos = self._index.get(fid)
            if pos is None:
                continue
            off, ln = pos
            triples.append((off, ln, fid))
        triples.sort(key=lambda t: t[0])
        out: Dict[str, Dict[str, Any]] = {}
        if not triples:
            return out

        def _do_read() -> None:
            pos_end = -1
            with open(self.path, "rb") as f:
                for offset, length, fid in triples:
                    if offset != pos_end:
                        f.seek(offset)
                    raw = f.read(length)
                    pos_end = offset + length
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    rec = _parse_jsonl_record(obj)
                    if rec is None or rec[0] != fid:
                        continue
                    out[fid] = rec[1]

        if self._read_lock is not None:
            with self._read_lock:
                _do_read()
        else:
            _do_read()
        return out

    def bulk_get_iter(self, function_ids: Iterable[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        批量读取并逐条产出 (function_id, multimodal)，避免构建大临时 dict。
        """
        triples: list[Tuple[int, int, str]] = []
        for fid in function_ids:
            pos = self._index.get(fid)
            if pos is None:
                continue
            off, ln = pos
            triples.append((off, ln, fid))
        triples.sort(key=lambda t: t[0])
        if not triples:
            return

        def _iter_read() -> Iterator[Tuple[str, Dict[str, Any]]]:
            pos_end = -1
            with open(self.path, "rb") as f:
                for offset, length, fid in triples:
                    if offset != pos_end:
                        f.seek(offset)
                    raw = f.read(length)
                    pos_end = offset + length
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    rec = _parse_jsonl_record(obj)
                    if rec is None or rec[0] != fid:
                        continue
                    yield fid, rec[1]

        if self._read_lock is not None:
            with self._read_lock:
                for item in _iter_read():
                    yield item
        else:
            for item in _iter_read():
                yield item

    def get(self, function_id: str) -> Optional[Dict[str, Any]]:
        pos = self._index.get(function_id)
        if pos is None:
            return None
        offset, length = pos

        def _one() -> Optional[Dict[str, Any]]:
            return self._read_one(function_id, offset, length)

        if self._read_lock is not None:
            with self._read_lock:
                return _one()
        return _one()


def build_jsonl_sidecar_lazy_index(
    path: str,
    needed_ids: Set[str],
    *,
    read_lock: Optional[Any] = None,
    reuse_read_file_handle: bool = True,
) -> JsonlSidecarLazyIndex:
    """
    流式扫描 JSONL，仅为 needed_ids 建立 {function_id -> (file_offset, line_length_bytes)}。
    优先走轻量 function_id 扫描，减少整行反序列化 multimodal 的峰值；非规范行回退 json.loads。
    """
    if not needed_ids:
        return JsonlSidecarLazyIndex(
            os.path.abspath(path),
            {},
            read_lock=read_lock,
            reuse_read_file_handle=reuse_read_file_handle,
        )
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"预计算特征文件不存在: {abs_path}")

    needed = set(needed_ids)
    index: Dict[str, Tuple[int, int]] = {}
    with open(abs_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            fid = _extract_function_id_from_jsonl_line_bytes(line)
            if fid is None:
                try:
                    obj = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                rec = _parse_jsonl_record(obj)
                if rec is None:
                    continue
                fid, _mm = rec
            if fid not in needed:
                continue
            index[fid] = (offset, len(line))
            needed.discard(fid)
            if not needed:
                break
    return JsonlSidecarLazyIndex(
        abs_path,
        index,
        read_lock=read_lock,
        reuse_read_file_handle=reuse_read_file_handle,
    )


def load_precomputed_multimodal_map(
    path: Optional[str],
    needed_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    加载预计算 multimodal。

    needed_ids 非 None 时：
      - JSONL：只保留需要的键，扫描完 needed 集为空时可提前结束。
      - JSON：全量解析后再按 needed_ids 过滤（大文件仍可能 OOM）。
    needed_ids 为 None 时加载全部（大 JSONL 仍可能 OOM，慎用）。
    """
    if not path:
        return {}
    if needed_ids is not None and len(needed_ids) == 0:
        return {}
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"预计算特征文件不存在: {abs_path}")

    if is_jsonl_sidecar_path(abs_path):
        out: Dict[str, Dict[str, Any]] = {}
        missing: Optional[Set[str]] = None
        if needed_ids is not None:
            missing = set(needed_ids)
        for fid, mm in iter_jsonl_sidecar(abs_path):
            if needed_ids is not None and fid not in needed_ids:
                continue
            out[fid] = mm
            if missing is not None:
                missing.discard(fid)
                if not missing:
                    break
        return out

    with open(abs_path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("预计算特征文件必须是 JSON 对象：{function_id: multimodal_dict}")
    out = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, dict)}
    if needed_ids is not None:
        out = {k: v for k, v in out.items() if k in needed_ids}
    return out


def write_jsonl_sidecar_line(fp: Any, function_id: str, multimodal: Dict[str, Any]) -> None:
    rec = {"function_id": function_id, "multimodal": multimodal}
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
