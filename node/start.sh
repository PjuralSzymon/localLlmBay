#!/bin/bash

# LocalLLMBay Node — start Ollama, then the Python worker.
# Client machine: logs go to container stdout (docker compose logs -f).
# Server Logdy sees queue activity via API (QUEUE.CLAIM / ANSWER), not via a shared volume.

log() {
  echo "[$(date -u +%H:%M:%S)] [node][$1] $2"
}

list_glob() {
  # Print matching paths on one line, or empty.
  local matches
  matches=$(ls -d $1 2>/dev/null | tr '\n' ' ')
  echo "${matches% }"
}

scan_gpus() {
  local phase="$1"
  log gpu "scan ($phase)"

  local nvidia_dev dri_dev kfd_dev smi smi_ok vk
  nvidia_dev=$(list_glob '/dev/nvidia*')
  dri_dev=$(list_glob '/dev/dri/renderD*')
  kfd_dev=$(list_glob '/dev/kfd')
  smi_ok=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    smi=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -n 8 | tr '\n' '; ')
    if [ -n "$smi" ]; then
      log gpu "nvidia-smi: $smi"
      smi_ok=1
    else
      log gpu "nvidia-smi present but returned no GPUs"
    fi
  else
    log gpu "nvidia-smi: not found (expected if Docker did not inject NVIDIA)"
  fi

  if [ -n "$nvidia_dev" ]; then
    log gpu "nvidia devices: $nvidia_dev"
  elif [ -n "$smi_ok" ]; then
    log gpu "nvidia devices: none in /dev (normal on Docker Desktop / WSL when nvidia-smi works)"
  else
    log gpu "nvidia devices: none"
  fi
  if [ -n "$dri_dev" ]; then
    log gpu "drm render: $dri_dev"
  else
    log gpu "drm render: none"
  fi
  if [ -n "$kfd_dev" ]; then
    log gpu "kfd: $kfd_dev"
  else
    log gpu "kfd: none"
  fi

  if command -v vulkaninfo >/dev/null 2>&1; then
    vk=$(vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' | grep -viE 'llvmpipe|swiftshader|lavapipe' | head -n 8 | tr '\n' '; ')
    if [ -n "$vk" ]; then
      log gpu "vulkaninfo: $vk"
    elif [ -n "$smi_ok" ]; then
      log gpu "vulkaninfo: Mesa software only; CUDA path is nvidia-smi (GPU is present)"
    else
      log gpu "vulkaninfo: no discrete adapters"
    fi
  else
    log gpu "vulkaninfo: not installed"
  fi

  if [ -n "$smi_ok" ] || [ -n "$dri_dev" ] || [ -n "$kfd_dev" ]; then
    return
  fi
  log gpu "CPU fallback: no GPU from nvidia-smi and no /dev/dri. Tick GPU on Setup so compose includes gpus: all."
}

log ollama "Starting LocalLLMBay Node…"
# Ollama slog has no WARN setting (unset/0 = INFO). Never raise slog to DEBUG/TRACE.
# GIN_MODE=release plus redirecting serve/pull keeps docker logs on [node] job
# lines and Ollama WARN/ERROR. Full serve chatter stays in /tmp/ollama-serve.log.

mkdir -p /app/models
export GIN_MODE="${GIN_MODE:-release}"
export OLLAMA_MODELS=/app/models
# Graph pack default. Job infer.num_ctx wins over leftover 64k compose.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-${NODE_NUM_CTX:-16384}}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
# Quantize the KV cache so more transformer layers stay on the GPU; leftover weights use RAM.
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
log ollama "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH} flash_attn=${OLLAMA_FLASH_ATTENTION} kv_cache=${OLLAMA_KV_CACHE_TYPE} keep_alive=${OLLAMA_KEEP_ALIVE} parallel=${OLLAMA_NUM_PARALLEL} max_loaded=${OLLAMA_MAX_LOADED_MODELS}"

scan_gpus "before ollama serve"

OLLAMA_SERVE_LOG=/tmp/ollama-serve.log
: > "$OLLAMA_SERVE_LOG"
log ollama "starting Ollama service (WARN/ERROR on stdout; full log $OLLAMA_SERVE_LOG)…"
(
  tail -n +1 -F "$OLLAMA_SERVE_LOG" 2>/dev/null | grep --line-buffered -E 'level=(WARN|WARNING|ERROR)|Listening on |msg="error"|fatal' | while IFS= read -r line; do
    log ollama "$line"
  done
) &
OLLAMA_TAIL_PID=$!
ollama serve >>"$OLLAMA_SERVE_LOG" 2>&1 &
OLLAMA_PID=$!

log ollama "waiting for Ollama to become ready (can take 15–60s)…"
sleep 15

for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log ollama "ready on 127.0.0.1:11434"
        break
    else
        log ollama "still waiting… ($i/30)"
        sleep 2
    fi
done

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log ollama "ERROR: Ollama did not become ready — continuing anyway"
fi

scan_gpus "after ollama ready"
if [ -f "$OLLAMA_SERVE_LOG" ]; then
  snap=$(grep -E 'inference compute|discovering available GPUs|total_vram|library=' "$OLLAMA_SERVE_LOG" | tail -n 12)
  if [ -n "$snap" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && log ollama "$line"
    done <<< "$snap"
  fi
fi
log gpu "Ollama INFO/GIN/pull bars stay in $OLLAMA_SERVE_LOG. nvidia-smi scan above is the GPU check; snapshot lines are what Ollama picked."

MODELS_TO_DOWNLOAD="${NODE_MODELS:-granite3.3:8b,deepseek-coder:6.7b,qwen2.5-coder:7b}"
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
RAM_GB=$(awk '/MemTotal:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null)
log worker "checking models: $MODELS_TO_DOWNLOAD vram_mb=${VRAM_MB:-unknown} ram_gb=${RAM_GB:-unknown}"

if ! FILTERED=$(
  NODE_MODELS="$MODELS_TO_DOWNLOAD" VRAM_MB="${VRAM_MB:-}" RAM_GB="${RAM_GB:-}" /app/venv/bin/python -c "
import os
from adapt import filter_models_for_vram
raw = os.environ.get('NODE_MODELS') or ''
v = (os.environ.get('VRAM_MB') or '').strip()
r = (os.environ.get('RAM_GB') or '').strip()
try:
    vram = float(v) if v else None
except ValueError:
    vram = None
try:
    ram = float(r) if r else None
except ValueError:
    ram = None
print(','.join(filter_models_for_vram(raw, vram, ram)))
"
); then
    log worker "WARN catalog VRAM filter failed — skipping pull"
    FILTERED=""
fi
log worker "models that fit this GPU+RAM: ${FILTERED:-none}"

for model in $(echo $MODELS_TO_DOWNLOAD | tr ',' ' '); do
    case ",$FILTERED," in
        *",$model,"*) ;;
        *)
            log ollama "skip pull model=$model — too big even as GPU+CPU split (vram_mb=${VRAM_MB:-unknown} ram_gb=${RAM_GB:-unknown})"
            ;;
    esac
done
MODELS_TO_DOWNLOAD="${FILTERED}"

NEED_HF_FLUX=0
NEED_HF_WAN=0
for model in $(echo $MODELS_TO_DOWNLOAD | tr ',' ' '); do
    if [ -z "$model" ]; then
        continue
    fi
    if NODE_TAG="$model" /app/venv/bin/python -c "import os,sys; from adapt import is_image_tag; sys.exit(0 if is_image_tag(os.environ.get('NODE_TAG')) else 1)"; then
        NEED_HF_FLUX=1
        log ollama "skip ollama pull model=$model — Flux runs via Hugging Face Diffusers, not Ollama"
        continue
    fi
    if NODE_TAG="$model" /app/venv/bin/python -c "import os,sys; from adapt import is_video_tag; sys.exit(0 if is_video_tag(os.environ.get('NODE_TAG')) else 1)"; then
        NEED_HF_WAN=1
        log ollama "skip ollama pull model=$model — Wan runs via Hugging Face Diffusers, not Ollama"
        continue
    fi
    if ! curl -s http://localhost:11434/api/tags | grep -q "$model"; then
        safe=$(printf '%s' "$model" | tr '/:' '__')
        pull_log="/tmp/ollama-pull-${safe}.log"
        log ollama "pulling model=$model (progress in $pull_log)"
        if ollama pull "$model" >"$pull_log" 2>&1; then
            log ollama "model ready model=$model"
        else
            log ollama "ERROR pull failed model=$model"
            grep -vE '^pulling ' "$pull_log" 2>/dev/null | tail -n 20 | while IFS= read -r line; do
                [ -n "$line" ] && log ollama "pull: $line"
            done
        fi
    else
        log ollama "model already available model=$model"
    fi
done

if [ "$NEED_HF_FLUX" = "1" ]; then
    log worker "downloading FLUX.2 Klein 4B Hugging Face weights (Diffusers)…"
    if /app/venv/bin/python -m flux_hf --ensure; then
        log worker "Flux Hugging Face weights ready"
    else
        log worker "ERROR Flux Hugging Face download failed — image hops will fail until this succeeds"
    fi
fi

if [ "$NEED_HF_WAN" = "1" ]; then
    log worker "downloading Wan 2.1 T2V 1.3B Hugging Face weights (Diffusers)…"
    if /app/venv/bin/python -m wan_hf --ensure; then
        log worker "Wan Hugging Face weights ready"
    else
        log worker "ERROR Wan Hugging Face download failed — video hops will fail until this succeeds"
    fi
fi

log ollama "model pulls done — Python worker will load the active rung (llama.cpp warmup) before the first heartbeat"
log worker "starting Python application…"
/app/venv/bin/python -u app.py
EXIT_CODE=$?

log worker "Python application stopped exit=$EXIT_CODE — shutting down Ollama…"
kill "$OLLAMA_TAIL_PID" 2>/dev/null || true
kill "$OLLAMA_PID" 2>/dev/null || true
exit $EXIT_CODE
