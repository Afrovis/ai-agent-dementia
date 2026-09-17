#!/usr/bin/env bash
# Serve the agent's local text model with MLX on the host. Metal is not
# available inside Docker, so this runs beside Ollama, not in compose.
#
#   python3.12 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm
#   tools/mlx_server/serve.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
server="${MLX_LM_SERVER:-$root/.venv-mlx/bin/mlx_lm.server}"
model="${AGENT_LLM_MODEL:-mlx-community/gemma-4-e4b-it-4bit}"

exec "$server" --model "$model" --host "${MLX_HOST:-127.0.0.1}" --port "${MLX_PORT:-11435}" \
  --chat-template-args '{"enable_thinking":false}' --log-level WARNING
