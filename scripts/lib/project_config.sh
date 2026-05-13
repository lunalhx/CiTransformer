#!/usr/bin/env bash
# =============================================================================
# scripts/lib/project_config.sh —— Shell 层配置工具函数库
# =============================================================================
# 【作用】
#   这个文件本身不能直接运行，它只定义函数，供其他 Shell 脚本"引入"后调用。
#   使用方式（在其他 .sh 脚本顶部写）：
#     source "$(dirname "$0")/project_config.sh"
#
# 【解决的核心问题】
#   其他脚本想用 Python 来读 YAML 配置，但首先得知道"用哪个 Python"——
#   这个文件就专门负责把这个"鸡生蛋蛋生鸡"的问题解开。
# =============================================================================


# -----------------------------------------------------------------------------
# resolve_bootstrap_python_bin
# -----------------------------------------------------------------------------
# 【作用】找到一个"能用的 Python"，用来启动配置读取。
#         优先顺序：项目 .venv → 系统 python3 → 系统 python
# 【注意】这是"引导阶段"用的，不考虑 YAML 配置，纯粹靠文件系统和 PATH 来找。
# 【返回】成功：打印 Python 路径并 return 0；找不到：return 1（脚本会报错）
# -----------------------------------------------------------------------------
resolve_bootstrap_python_bin() {
  # 第一优先：项目自带的 .venv 虚拟环境（最可靠，依赖一定匹配）
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    echo "${PROJECT_ROOT}/.venv/bin/python"
    return 0
  fi

  # 第二优先：系统 PATH 里的 python3（macOS / Linux 通常有）
  # >/dev/null 2>&1 表示把所有输出（含错误）丢掉，只关心命令是否存在
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  # 第三优先：系统 PATH 里的 python（Windows 或旧系统兜底）
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  # 三种都找不到，返回失败（调用方会触发报错）
  return 1
}


# -----------------------------------------------------------------------------
# resolve_python_bin
# -----------------------------------------------------------------------------
# 【作用】决定"最终用哪个 Python"，并把结果暴露给其他脚本使用。
#         优先级：环境变量 PYTHON_BIN > YAML 配置 runtime.python_bin > 引导 Python
# 【调用方式】PYTHON_BIN="$(resolve_python_bin)"
# -----------------------------------------------------------------------------
resolve_python_bin() {
  # 第一优先：环境变量 PYTHON_BIN 已经设置，直接用，不再查配置
  # ${PYTHON_BIN:-} 表示"取变量值，若未设置则返回空字符串"（避免 set -u 报错）
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi

  # 用引导 Python 来读 YAML 配置（先找到一个能跑的 Python）
  local bootstrap_python
  bootstrap_python="$(resolve_bootstrap_python_bin)" || return 1

  # 第二优先：读 configs/local.yaml 或 default.yaml 里的 runtime.python_bin
  # -m utils.project_config get runtime.python_bin 等同于：
  #   python utils/project_config.py get runtime.python_bin
  # 2>/dev/null || true：读取失败（如 yaml 未安装）时静默忽略，不中断脚本
  local configured_python
  configured_python="$(cd "${PROJECT_ROOT}" && "${bootstrap_python}" -m utils.project_config get runtime.python_bin 2>/dev/null || true)"
  if [[ -n "${configured_python}" ]]; then
    echo "${configured_python}"
    return 0
  fi

  # 第三优先：YAML 里没配，就用刚才找到的引导 Python 本身
  echo "${bootstrap_python}"
}


# -----------------------------------------------------------------------------
# project_config_get <dotted.key>
# -----------------------------------------------------------------------------
# 【作用】从配置系统（default.yaml + local.yaml + 环境变量）读取一个值。
# 【用法示例】
#   DATA_DIR="$(project_config_get paths.data_dir)"
#   DEVICE="$(project_config_get runtime.device)"
# 【注意】调用前必须已经设置好 PROJECT_ROOT 和 PYTHON_BIN。
# -----------------------------------------------------------------------------
project_config_get() {
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -m utils.project_config get "$1")
}


# -----------------------------------------------------------------------------
# project_path <路径值>
# -----------------------------------------------------------------------------
# 【作用】把配置里读出来的路径值统一转成绝对路径，供脚本直接使用。
# 【逻辑】
#   - 以 / 开头  → 绝对路径，原样返回
#   - 以 ~ 开头  → 展开家目录（~/xxx → /Users/yourname/xxx）
#   - 其他        → 视为相对路径，拼接到 PROJECT_ROOT 后面
# 【用法示例】
#   DATA_DIR="$(project_path "$(project_config_get paths.data_dir)")"
# -----------------------------------------------------------------------------
project_path() {
  local value="$1"
  case "${value}" in
    /*)          # 已经是绝对路径，直接用
      echo "${value}"
      ;;
    "~"*)        # 以 ~ 开头，展开为 $HOME
      echo "${value/#\~/${HOME}}"
      ;;
    *)           # 相对路径，拼到项目根目录下
      echo "${PROJECT_ROOT}/${value}"
      ;;
  esac
}


# -----------------------------------------------------------------------------
# setup_matplotlib_cache
# -----------------------------------------------------------------------------
# 【作用】设置 matplotlib 的缓存目录，并确保该目录存在。
# 【为什么需要这个】
#   多进程训练时，多个进程同时写 ~/.config/matplotlib 会冲突报错。
#   把缓存目录改到项目内或 /tmp 下，就能安全并行。
# 【优先级】
#   已设置的 $MPLCONFIGDIR 环境变量 > YAML 配置 paths.matplotlib_cache > /tmp 兜底
# 【调用时机】在训练脚本启动 Python 之前调用一次即可。
# -----------------------------------------------------------------------------
setup_matplotlib_cache() {
  local default_mplconfig
  # 从配置系统读取 paths.matplotlib_cache；读取失败时降级到 /tmp
  default_mplconfig="$(project_config_get paths.matplotlib_cache 2>/dev/null || echo "/tmp/citransformer-matplotlib")"

  # ${MPLCONFIGDIR:-...} 表示：若外部已设置 MPLCONFIGDIR 则保留，否则用上面读到的值
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${default_mplconfig}}"

  # 确保目录存在（-p 表示递归创建，目录已存在时不报错）
  mkdir -p "${MPLCONFIGDIR}"
}
