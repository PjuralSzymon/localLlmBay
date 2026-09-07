# Official Ollama image ships CUDA + Vulkan runners. Pin matches the last
# known-good node boot (0.33.2). CPU-only hosts still work; GPU needs Docker
# to pass devices (Setup GPU checkbox writes gpus: all into docker-compose-node.yml).
FROM ollama/ollama:0.33.2

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Python worker + Mesa Vulkan ICDs (AMD/Intel when /dev/dri is mapped).
# nvidia-smi is injected at runtime by NVIDIA Container Toolkit, not baked in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    ca-certificates \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /app/venv

COPY requirements.txt .
COPY requirements-flux.txt .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /app/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && /app/venv/bin/pip install --no-cache-dir -r requirements-flux.txt

COPY app.py .
COPY adapt.py .
COPY ollama_messages.py .
COPY ollama_ctx.py .
COPY ollama_image.py .
COPY flux_hf.py .
COPY wan_hf.py .
COPY debug_log.py .
COPY hardware.py .
COPY wire_crypto.py .
COPY tool_parse.py .
COPY tool_budget.py .
COPY start.sh .

RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

EXPOSE 5000
EXPOSE 11434

# Official image ENTRYPOINT is /bin/ollama; we start serve from start.sh.
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
