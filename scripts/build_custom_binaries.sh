#!/bin/bash
# 自编译 BusyBox 可控样本：同一源码不同优化 (-O0 vs -O3) 用于相似性检测测试。
# 依赖: gcc-mipsel-linux-gnu (或 gcc-arm-linux-gnueabi 用于 ARM)
# 输出: data/binaries/custom/busybox-O0, busybox-O3

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${PROJECT_ROOT}/data/binaries/custom"
BUSYBOX_VERSION="${BUSYBOX_VERSION:-1.36.1}"
URL="https://busybox.net/downloads/busybox-${BUSYBOX_VERSION}.tar.bz2"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -d "busybox-${BUSYBOX_VERSION}" ]; then
  echo "下载 BusyBox ${BUSYBOX_VERSION} ..."
  wget -q -O- "$URL" | tar xj
fi

cd "busybox-${BUSYBOX_VERSION}"

# 检测可用交叉编译器
if command -v mipsel-linux-gnu-gcc &>/dev/null; then
  CROSS="mipsel-linux-gnu-"
  ARCH="mips"
elif command -v mips-linux-gnu-gcc &>/dev/null; then
  CROSS="mips-linux-gnu-"
  ARCH="mips"
elif command -v arm-linux-gnueabi-gcc &>/dev/null; then
  CROSS="arm-linux-gnueabi-"
  ARCH="arm"
else
  echo "使用原生 gcc (x86_64)"
  CROSS=""
  ARCH="x86"
fi

echo "交叉编译前缀: ${CROSS:-无}"

make distclean 2>/dev/null || true
make defconfig

# 编译 O0
make CROSS_COMPILE="$CROSS" CFLAGS="-O0" -j$(nproc) 2>/dev/null
cp busybox "${OUT_DIR}/busybox-O0-${ARCH}"
make clean

# 编译 O3
make CROSS_COMPILE="$CROSS" CFLAGS="-O3" -j$(nproc) 2>/dev/null
cp busybox "${OUT_DIR}/busybox-O3-${ARCH}"

echo "完成: ${OUT_DIR}/busybox-O{0,3}-${ARCH}"
