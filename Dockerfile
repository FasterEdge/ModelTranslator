# ModelTranslator Docker 镜像
# =====================================================================
# 构建（依赖按需安装，默认仅 ONNX 后端，体积轻量）：
#   docker build -t model-translator .
#
# 按需组合后端（空格分隔多个分组）：
#   docker build --build-arg 'UV_EXTRAS=onnx openvino' -t model-translator .
#
# 全部后端（体积很大，一般不需要）：
#   docker build --build-arg 'UV_EXTRAS=all' -t model-translator-full .
# =====================================================================

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 构建参数：按需选择要安装的依赖分组（对应 pyproject.toml 的 optional-dependencies）
ARG UV_EXTRAS="onnx"

# ===== 系统依赖（TensorFlow / OpenCV / coremltools 等需要）=====
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== 先复制依赖清单（利用缓存层）=====
COPY pyproject.toml README.md ./
COPY src ./src

# ===== 按需安装转换后端 + 项目自身 =====
# 把 UV_EXTRAS（空格分隔）转成 uv sync 的多个 --extra 参数
RUN set -eux; \
    if [ -z "$UV_EXTRAS" ]; then \
        uv sync --no-dev; \
    else \
        EXTRAS_ARGS=""; \
        for e in $UV_EXTRAS; do EXTRAS_ARGS="$EXTRAS_ARGS --extra $e"; done; \
        uv sync $EXTRAS_ARGS --no-dev; \
    fi

# ===== 工作目录（挂载模型输入输出）=====
WORKDIR /workspace
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["model-translator"]
CMD ["list"]