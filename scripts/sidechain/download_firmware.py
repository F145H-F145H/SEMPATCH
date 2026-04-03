#!/usr/bin/env python3
"""
下载测试用固件：OpenWrt（直接下载）及厂商固件（文档化链接）。

用法:
  python scripts/download_firmware.py              # 下载 OpenWrt x86_64 rootfs
  python scripts/download_firmware.py --list       # 仅列出可用固件及链接
"""
import argparse
import gzip
import os
import shutil
import sys
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIRMWARE_DIR = os.path.join(PROJECT_ROOT, "data", "firmware")

# 可直接下载的固件（OpenWrt 等）
DOWNLOADABLE = [
    {
        "name": "openwrt-22.03.5-x86-64-squashfs-rootfs",
        "url": "https://downloads.openwrt.org/releases/22.03.5/targets/x86/64/openwrt-22.03.5-x86-64-generic-squashfs-rootfs.img.gz",
        "arch": "x86_64",
        "note": "标准 SquashFS rootfs，binwalk 可直接解包",
        "decompress": True,
    },
    {
        "name": "openwrt-22.03.5-ath79-generic-rootfs",
        "url": "https://downloads.openwrt.org/releases/22.03.5/targets/ath79/generic/openwrt-22.03.5-ath79-generic-tplink_archer-c7-v2-squashfs-sysupgrade.bin",
        "arch": "MIPS",
        "note": "TP-Link Archer C7 固件，含 SquashFS",
        "decompress": False,
    },
]

# 需手动下载的厂商固件（链接可能变更）
MANUAL = [
    {
        "name": "D-Link DIR-816 A2",
        "url": "https://support.dlink.com/ProductInfo.aspx?m=DIR-816",
        "arch": "MIPS",
        "note": "中国区: support.dlink.com.cn，搜索 DIR-816 A2 固件",
    },
    {
        "name": "TP-Link WR841N",
        "url": "https://www.tp-link.com/support/download/",
        "arch": "MIPS",
        "note": "搜索 WR841N 选择对应硬件版本",
    },
]


def download_url(url: str, dest: str, decompress: bool = False) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        if decompress and dest.endswith(".gz"):
            out_path = dest[:-3] if dest.endswith(".img.gz") else dest
            with gzip.open(tmp, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(tmp)
            print(f"  已解压 -> {out_path}")
        else:
            shutil.move(tmp, dest)
            print(f"  已保存 -> {dest}")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"下载失败 {url}: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 SemPatch 测试固件")
    parser.add_argument("--list", action="store_true", help="仅列出可用固件")
    parser.add_argument("--out", default=FIRMWARE_DIR, help="输出目录")
    parser.add_argument("--name", choices=[d["name"] for d in DOWNLOADABLE], help="指定下载项")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.list:
        print("=== 可直接下载 ===")
        for d in DOWNLOADABLE:
            print(f"  {d['name']} ({d['arch']}) - {d['note']}")
        print("\n=== 需手动下载（厂商官网） ===")
        for m in MANUAL:
            print(f"  {m['name']}: {m['url']}")
            print(f"    {m['note']}")
        return

    to_download = [d for d in DOWNLOADABLE if args.name is None or d["name"] == args.name]
    for d in to_download:
        ext = ".img.gz" if d["decompress"] else ""
        base = d["name"] + ext
        dest = os.path.join(args.out, base)
        out_final = dest[:-3] if d["decompress"] and dest.endswith(".gz") else dest
        if os.path.exists(out_final):
            print(f"已存在，跳过: {out_final}")
            continue
        print(f"下载 {d['name']} ...")
        download_url(d["url"], dest, decompress=d["decompress"])

    print(f"\n固件目录: {args.out}")


if __name__ == "__main__":
    main()
