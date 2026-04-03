#!/bin/bash

# SemPatch 环境设置脚本
# 1. Python 虚拟环境 (.venv) + pip install
# 2. Ghidra 下载与安装（可选，--skip-ghidra 跳过）
# 3. Binwalk 安装（可选，--skip-binwalk 跳过）
# 4. sempatch.cfg 初始化
# 用法: ./scripts/setup.sh [--skip-ghidra] [--skip-python] [--skip-binwalk] [--proxy HOST:PORT]

set -e  # 遇到错误时退出

# 解析参数
SKIP_GHIDRA=""
SKIP_PYTHON=""
SKIP_BINWALK=""
PROXY_URL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-ghidra) SKIP_GHIDRA=1; shift ;;
    --skip-python) SKIP_PYTHON=1; shift ;;
    --skip-binwalk) SKIP_BINWALK=1; shift ;;
    --proxy)
      shift
      if [ -n "$1" ]; then
        PROXY_URL="$1"
        shift
      else
        echo "错误: --proxy 需要指定地址，例如 --proxy 127.0.0.1:7890"
        exit 1
      fi
      ;;
    --proxy=*)
      PROXY_URL="${1#--proxy=}"
      shift
      ;;
    *) shift ;;
  esac
done

# 设置代理环境变量（支持 pip、curl、wget、apt、brew）
if [ -n "$PROXY_URL" ]; then
  case "$PROXY_URL" in
    http://*|https://*) ;;
    *) PROXY_URL="http://${PROXY_URL}" ;;
  esac
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export NO_PROXY="localhost,127.0.0.1"
  export no_proxy="localhost,127.0.0.1"
fi

# 配置变量
GHIDRA_VERSION="12.0"
GHIDRA_BUILD="20251205"
GHIDRA_DIR_NAME="ghidra_${GHIDRA_VERSION}_PUBLIC"
GHIDRA_ZIP="${GHIDRA_DIR_NAME}_${GHIDRA_BUILD}.zip"
GHIDRA_URL="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_BUILD}.zip"

# SHA-256 哈希验证
EXPECTED_SHA256="af43e8cfb2fa4490cf6020c3a2bde25c159d83f45236a0542688a024e8fc1941"

# 项目目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
GHIDRA_INSTALL_DIR="${THIRD_PARTY_DIR}/${GHIDRA_DIR_NAME}"
GHIDRA_ZIP_PATH="${THIRD_PARTY_DIR}/${GHIDRA_ZIP}"
SHA256_FILE="${GHIDRA_ZIP_PATH}.sha256"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Python 虚拟环境与依赖
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"

setup_python_venv() {
    log_info "设置 Python 虚拟环境..."
    if [ ! -d "$VENV_DIR" ]; then
        log_info "创建 .venv..."
        python3 -m venv "$VENV_DIR" || {
            log_error "创建虚拟环境失败，请确保已安装 python3-venv"
            return 1
        }
        log_success "已创建 .venv"
    else
        log_info ".venv 已存在"
    fi
    log_info "激活虚拟环境并安装依赖..."
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip -q
    if [ -f "$REQUIREMENTS" ]; then
        pip install -r "$REQUIREMENTS"
        log_success "已安装 requirements.txt"
    else
        log_warning "requirements.txt 不存在，跳过 pip install -r"
    fi
    # 可选依赖（YAML 配置、ML 增强），--timeout 避免长时间无响应
    pip install pyyaml -q --timeout 60 2>/dev/null || true
    log_info "安装可选依赖 xgboost、imbalanced-learn（可跳过）..."
    pip install xgboost -q --timeout 120 2>/dev/null || log_warning "xgboost 可选，train_ml_fp_filter --backend xgboost 时需手动安装"
    pip install imbalanced-learn -q --timeout 120 2>/dev/null || log_warning "imbalanced-learn 可选，--balance smote/undersample 时需手动安装"
    log_success "Python 环境就绪"
    echo ""
    echo "  激活虚拟环境: source ${VENV_DIR}/bin/activate"
    echo "  或使用: . ${VENV_DIR}/bin/activate"
    echo "  之后运行脚本需在项目根目录且 PYTHONPATH=src，例如:"
    echo "    PYTHONPATH=src python scripts/eval/run_eval.py --dataset datasets/example_complex_plus"
    echo ""
}

# 获取文件大小（跨平台：BSD stat -f%z / GNU stat -c%s）
get_file_size() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo 0
        return
    fi
    stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo 0
}

# 检查下载工具（启用代理时优先 curl，因 axel 代理支持不稳定）
check_download_tools() {
    log_info "检查下载工具..."
    
    if [ -n "$PROXY_URL" ]; then
        if command -v curl &> /dev/null; then
            log_success "找到 curl（代理模式）"
            DOWNLOAD_TOOL="curl"
        elif command -v wget &> /dev/null; then
            log_success "找到 wget（代理模式）"
            DOWNLOAD_TOOL="wget"
        elif command -v axel &> /dev/null; then
            log_warning "axel 代理支持有限，建议安装 curl"
            DOWNLOAD_TOOL="axel"
        else
            log_error "未找到 curl 或 wget"
            return 1
        fi
    else
        if command -v axel &> /dev/null; then
            log_success "找到 axel"
            DOWNLOAD_TOOL="axel"
        elif command -v curl &> /dev/null; then
            log_success "找到 curl"
            DOWNLOAD_TOOL="curl"
        elif command -v wget &> /dev/null; then
            log_success "找到 wget"
            DOWNLOAD_TOOL="wget"
        else
            log_error "未找到任何下载工具 (axel, curl, wget)"
            return 1
        fi
    fi
    
    return 0
}

# 快速验证文件完整性（检查文件大小和缓存）
quick_verify_integrity() {
    local file_path="$1"
    local skip_sha256="${2:-false}"
    
    # 检查文件是否存在
    if [ ! -f "$file_path" ]; then
        return 1
    fi
    
    # 检查文件大小
    local file_size
    file_size=$(get_file_size "$file_path")
    
    # Ghidra 12.0 大约 500MB
    if [ "$file_size" -lt 400000000 ]; then
        log_warning "文件可能不完整或损坏 (小于 400MB)"
        return 1
    fi
    
    # 如果有缓存且不需要重新计算 SHA256，直接返回
    if [ "$skip_sha256" = "true" ] && [ -f "$SHA256_FILE" ]; then
        local stored_sha256
        stored_sha256=$(cut -d' ' -f1 "$SHA256_FILE" 2>/dev/null || echo "")
        if [ "$stored_sha256" = "$EXPECTED_SHA256" ]; then
            log_info "使用缓存的 SHA-256 验证结果"
            return 0
        fi
    fi
    
    # 否则进行完整验证
    return 2
}

# 完整验证文件完整性
full_verify_integrity() {
    local file_path="$1"
    
    # 计算并验证 SHA-256
    log_info "计算 SHA-256 哈希值..."
    
    local actual_sha256=""
    if command -v sha256sum &> /dev/null; then
        actual_sha256=$(sha256sum "$file_path" 2>/dev/null | cut -d' ' -f1)
    elif command -v shasum &> /dev/null; then
        actual_sha256=$(shasum -a 256 "$file_path" 2>/dev/null | cut -d' ' -f1)
    else
        log_warning "无法计算 SHA-256 (缺少 sha256sum 或 shasum 工具)"
        return 1
    fi
    
    if [ -z "$actual_sha256" ]; then
        log_error "无法计算 SHA-256 哈希值"
        return 1
    fi
    
    log_info "预期 SHA-256: $EXPECTED_SHA256"
    log_info "实际 SHA-256: $actual_sha256"
    
    if [ "$actual_sha256" = "$EXPECTED_SHA256" ]; then
        log_success "SHA-256 验证通过"
        
        # 保存哈希值到文件以便后续验证
        echo "$actual_sha256  $(basename "$file_path")" > "$SHA256_FILE"
        return 0
    else
        log_error "SHA-256 验证失败"
        return 1
    fi
}

# 下载 Ghidra
download_ghidra() {
    log_info "下载 Ghidra $GHIDRA_VERSION..."
    log_info "下载链接: $GHIDRA_URL"
    
    # 创建 third_party 目录
    mkdir -p "$THIRD_PARTY_DIR"
    
    # 检查下载工具
    check_download_tools || {
        log_error "没有可用的下载工具"
        return 1
    }
    
    log_info "开始下载 (使用 $DOWNLOAD_TOOL)..."
    
    # 根据下载工具执行下载
    case $DOWNLOAD_TOOL in
        "axel")
            log_info "使用 axel 多线程下载 (8线程)..."
            if ! axel -n 8 -a -o "$GHIDRA_ZIP_PATH" "$GHIDRA_URL"; then
                log_error "axel 下载失败"
                return 1
            fi
            ;;
        "curl")
            log_info "使用 curl 下载..."
            if ! curl -L -o "$GHIDRA_ZIP_PATH" "$GHIDRA_URL" --progress-bar; then
                log_error "curl 下载失败"
                return 1
            fi
            ;;
        "wget")
            log_info "使用 wget 下载..."
            if ! wget -O "$GHIDRA_ZIP_PATH" "$GHIDRA_URL"; then
                log_error "wget 下载失败"
                return 1
            fi
            ;;
        *)
            log_error "未知的下载工具"
            return 1
            ;;
    esac
    
    log_success "下载完成: $GHIDRA_ZIP"
    
    # 验证文件完整性
    log_info "验证文件完整性..."
    local file_size
    file_size=$(get_file_size "$GHIDRA_ZIP_PATH")
    log_info "文件大小: $(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "${file_size} 字节")"
    
    # 进行完整验证
    if full_verify_integrity "$GHIDRA_ZIP_PATH"; then
        return 0
    else
        log_error "文件验证失败"
        return 1
    fi
}

# 重试下载函数
download_ghidra_with_retry() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if [ $retry_count -gt 0 ]; then
            log_info "第 $((retry_count + 1)) 次尝试下载..."
        fi
        
        if download_ghidra; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        
        if [ $retry_count -lt $max_retries ]; then
            log_warning "下载失败，3秒后重试..."
            sleep 3
            
            # 清理可能损坏的文件
            if [ -f "$GHIDRA_ZIP_PATH" ]; then
                log_info "清理损坏的文件..."
                rm -f "$GHIDRA_ZIP_PATH" "$SHA256_FILE"
            fi
        fi
    done
    
    log_error "经过 $max_retries 次尝试后下载仍然失败"
    log_info "请尝试以下手动解决方案:"
    log_info "1. 检查网络连接和代理"
    log_info "2. 手动下载并放置到: $THIRD_PARTY_DIR/"
    log_info "3. 手动验证 SHA-256:"
    log_info "   预期: $EXPECTED_SHA256"
    log_info "4. 访问: https://github.com/NationalSecurityAgency/ghidra/releases"
    log_info "5. 下载 Ghidra $GHIDRA_VERSION 并重命名为: $GHIDRA_ZIP"
    
    return 1
}

# 检查是否有已下载的安装包
check_existing_zip() {
    log_info "检查已下载的安装包..."
    
    # 检查是否有指定版本的 zip 文件
    if [ -f "$GHIDRA_ZIP_PATH" ]; then
        log_success "找到已下载的安装包: $GHIDRA_ZIP"
        
        # 快速验证（只检查文件大小和缓存）
        if quick_verify_integrity "$GHIDRA_ZIP_PATH" true; then
            return 0
        elif [ $? -eq 2 ]; then
            # 需要完整验证
            log_info "进行完整验证..."
            if full_verify_integrity "$GHIDRA_ZIP_PATH"; then
                return 0
            fi
        fi
        
        log_warning "现有文件验证失败，需要重新下载"
        rm -f "$GHIDRA_ZIP_PATH" "$SHA256_FILE"
        return 1
    fi
    
    return 1
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    local missing_deps=()
    
    # 检查下载工具
    if ! check_download_tools; then
        missing_deps+=("curl 或 wget")
    fi
    
    # 检查 unzip
    if ! command -v unzip &> /dev/null; then
        missing_deps+=("unzip")
    fi
    
    # 检查 numfmt（用于格式化文件大小）
    if ! command -v numfmt &> /dev/null; then
        log_warning "numfmt 未找到，文件大小显示可能不友好"
    fi
    
    # 检查 Java (Ghidra 需要 Java 17+ for version 12.0)
    if ! command -v java &> /dev/null; then
        missing_deps+=("java")
    else
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}')
        log_info "Java 版本: $JAVA_VERSION"
        
        # 检查 Java 版本是否 >= 17 (Ghidra 12.0 需要 Java 17+)
        local major_version
        major_version=$(echo "$JAVA_VERSION" | awk -F '.' '{print $1}')
        if [ "$major_version" -lt 17 ]; then
            log_error "Ghidra $GHIDRA_VERSION 需要 Java 17 或更高版本，当前版本: $JAVA_VERSION"
            log_error "请升级 Java 版本"
            missing_deps+=("java>=17")
        fi
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "缺少以下依赖: ${missing_deps[*]}"
        log_info "请使用系统包管理器安装:"
        log_info "  Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
        log_info "  macOS: brew install ${missing_deps[*]}"
        log_info "  Windows: 请手动安装上述工具"
        return 1
    fi
    
    log_success "所有依赖已满足"
    return 0
}

# 安装 Ghidra
install_ghidra() {
    log_info "安装 Ghidra..."
    
    # 检查 zip 文件是否存在
    if [ ! -f "$GHIDRA_ZIP_PATH" ]; then
        log_error "安装包不存在: $GHIDRA_ZIP_PATH"
        return 1
    fi
    
    # 快速验证（跳过 SHA256 计算，使用缓存）
    log_info "验证文件完整性..."
    local file_size
    file_size=$(get_file_size "$GHIDRA_ZIP_PATH")
    log_info "文件大小: $(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "${file_size} 字节")"
    
    # 使用快速验证，如果已经验证过就不重复计算 SHA256
    if quick_verify_integrity "$GHIDRA_ZIP_PATH" true; then
        log_info "使用缓存的验证结果，跳过 SHA-256 计算"
    else
        # 需要完整验证
        if ! full_verify_integrity "$GHIDRA_ZIP_PATH"; then
            log_error "文件完整性验证失败，无法安装"
            return 1
        fi
    fi
    
    # 解压到 third_party 目录
    log_info "解压安装包..."
    
    # 检查目标目录是否已存在
    if [ -d "$GHIDRA_INSTALL_DIR" ]; then
        log_warning "目标目录已存在: $GHIDRA_INSTALL_DIR"
        log_info "备份旧目录..."
        local backup_dir="${GHIDRA_INSTALL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        mv "$GHIDRA_INSTALL_DIR" "$backup_dir" 2>/dev/null || {
            log_error "无法备份旧目录，尝试删除..."
            rm -rf "$GHIDRA_INSTALL_DIR"
        }
    fi
    
    if unzip -q "$GHIDRA_ZIP_PATH" -d "$THIRD_PARTY_DIR"; then
        log_success "解压完成"
        
        # 验证安装
        if [ -d "$GHIDRA_INSTALL_DIR" ]; then
            # 设置执行权限
            if [ -f "${GHIDRA_INSTALL_DIR}/support/analyzeHeadless" ]; then
                chmod +x "${GHIDRA_INSTALL_DIR}/support/analyzeHeadless"
                log_success "Ghidra 安装成功: $GHIDRA_INSTALL_DIR"
            else
                log_error "Ghidra 安装不完整，analyzeHeadless 脚本不存在"
                return 1
            fi
        else
            log_error "解压后未找到预期的 Ghidra 目录"
            log_info "解压内容:"
            ls -la "$THIRD_PARTY_DIR"
            return 1
        fi
    else
        log_error "解压失败，可能文件损坏"
        log_info "请重新下载或检查文件完整性"
        return 1
    fi
    
    return 0
}

# 安装 Binwalk（用于固件解包 DAG 节点）
setup_binwalk() {
    log_info "检查 Binwalk..."
    if command -v binwalk &> /dev/null; then
        log_success "Binwalk 已安装: $(binwalk --version 2>/dev/null | head -n1 || echo 'binwalk')"
        return 0
    fi

    log_info "安装 Binwalk（需 sudo/管理员权限）..."
    case "$(uname -s)" in
        Linux)
            if command -v apt-get &> /dev/null; then
                sudo apt-get update -qq && sudo apt-get install -y binwalk || {
                    log_error "apt 安装 binwalk 失败"
                    return 1
                }
            elif command -v apt &> /dev/null; then
                sudo apt update -qq && sudo apt install -y binwalk || {
                    log_error "apt 安装 binwalk 失败"
                    return 1
                }
            else
                log_error "未找到 apt，请手动安装 binwalk: https://github.com/ReFirmLabs/binwalk"
                return 1
            fi
            ;;
        Darwin)
            if command -v brew &> /dev/null; then
                brew install binwalk || {
                    log_error "brew 安装 binwalk 失败"
                    return 1
                }
            else
                log_error "未找到 brew，请安装 Homebrew 后执行: brew install binwalk"
                return 1
            fi
            ;;
        *)
            log_warning "未知系统，请手动安装 binwalk"
            return 1
            ;;
    esac

    if command -v binwalk &> /dev/null; then
        log_success "Binwalk 安装成功"
        return 0
    fi
    log_error "Binwalk 安装后仍不可用"
    return 1
}

# 初始化 sempatch.cfg
setup_config() {
    local cfg_path="${PROJECT_ROOT}/sempatch.cfg"
    local example_path="${PROJECT_ROOT}/sempatch.cfg.example"

    if [ -f "$cfg_path" ]; then
        log_info "sempatch.cfg 已存在，跳过初始化"
        return 0
    fi

    if [ ! -f "$example_path" ]; then
        log_warning "sempatch.cfg.example 不存在，跳过 config 初始化"
        return 0
    fi

    log_info "初始化 sempatch.cfg..."
    cp "$example_path" "$cfg_path"

    # 写入实际 ghidra_home（相对项目根）
    local ghidra_rel="third_party/${GHIDRA_DIR_NAME}"
    if [ -d "${PROJECT_ROOT}/${ghidra_rel}" ]; then
        if command -v sed &> /dev/null; then
            sed -i "s|^ghidra_home =.*|ghidra_home = ${ghidra_rel}|" "$cfg_path" 2>/dev/null || \
            sed -i '' "s|^ghidra_home =.*|ghidra_home = ${ghidra_rel}|" "$cfg_path" 2>/dev/null || true
        fi
    fi
    log_success "已创建 sempatch.cfg"
}

# 主函数
main() {
    echo "========================================"
    echo "    SemPatch 环境设置脚本"
    echo "    Python .venv + Ghidra $GHIDRA_VERSION + Binwalk"
    if [ -n "$PROXY_URL" ]; then
        echo "    代理: $PROXY_URL"
    fi
    echo "========================================"
    echo ""
    
    # 1. Python 虚拟环境
    if [ -z "$SKIP_PYTHON" ]; then
        setup_python_venv
    else
        log_info "跳过 Python 设置 (--skip-python)"
    fi
    
    # 2. Ghidra（可选）
    if [ -z "$SKIP_GHIDRA" ]; then
        # 检查 Ghidra 依赖
        if ! check_dependencies; then
            log_error "依赖检查失败，请先安装缺少的依赖"
            exit 1
        fi

        echo ""

        # 检查是否已安装 Ghidra
        if [ -d "$GHIDRA_INSTALL_DIR" ] && [ -f "${GHIDRA_INSTALL_DIR}/support/analyzeHeadless" ]; then
            log_success "Ghidra 已安装于: $GHIDRA_INSTALL_DIR"
            log_info "Ghidra 已安装，跳过下载和安装步骤"
        else
            log_info "开始安装 Ghidra..."
            if check_existing_zip; then
                log_info "使用已下载的安装包"
            else
                if ! download_ghidra_with_retry; then
                    log_error "下载失败"
                    exit 1
                fi
            fi
            if ! install_ghidra; then
                log_error "安装失败"
                exit 1
            fi
        fi
    else
        log_info "跳过 Ghidra 设置 (--skip-ghidra)"
    fi

    # 3. Binwalk（可选）
    echo ""
    if [ -z "$SKIP_BINWALK" ]; then
        setup_binwalk || log_warning "Binwalk 安装失败，固件解包 DAG 节点将不可用"
    else
        log_info "跳过 Binwalk 设置 (--skip-binwalk)"
    fi

    # 4. Config 初始化
    echo ""
    setup_config
}

# 运行主函数
main "$@"