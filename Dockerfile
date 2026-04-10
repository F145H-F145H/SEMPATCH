FROM python:3.12-slim

LABEL maintainer="SemPatch"
LABEL description="SemPatch Replain — firmware 1-day vulnerability discovery prototype"

# 系统依赖：JDK 21（Ghidra 需要）、binwalk（固件解包）、git、make
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        binwalk \
        git \
        make \
        wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制依赖清单（利用 Docker 层缓存）
COPY requirements.txt requirements_frozen.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -f requirements_frozen.txt ]; then pip install --no-cache-dir -r requirements_frozen.txt; fi

# 下载并安装 Ghidra 12.0
ARG GHIDRA_VERSION=12.0
ARG GHIDRA_BUILD=20251205
ARG GHIDRA_SHA256=af43e8cfb2fa4490cf6020c3a2bde25c159d83f45236a0542688a024e8fc1941
ARG GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_BUILD}.zip

RUN mkdir -p third_party \
    && wget -q -O /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA256}  /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q /tmp/ghidra.zip -d third_party/ \
    && rm /tmp/ghidra.zip \
    && ls third_party/ghidra_${GHIDRA_VERSION}_PUBLIC/support/analyzeHeadless

# 复制项目源码
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY docs/ docs/
COPY Makefile sempatch.cfg.example ./

# 初始化配置
RUN cp sempatch.cfg.example sempatch.cfg \
    && sed -i "s|^ghidra_home =.*|ghidra_home = /app/third_party/ghidra_${GHIDRA_VERSION}_PUBLIC|" sempatch.cfg

# 环境变量
ENV PYTHONPATH=src
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# 默认入口：跑最小复现子集
CMD ["make", "reproduce"]
