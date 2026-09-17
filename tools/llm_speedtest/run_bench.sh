#!/usr/bin/env bash
# Run the 50-scenario dialogue bench against MLX models (one mlx_lm.server
# per model) and optional Ollama tags, saving JSON with generated text.
#
#   tools/llm_speedtest/run_bench.sh mlx:mlx-community/gemma-4-e4b-it-4bit ollama:gemma4:e4b-mlx
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
py="${PYTHON:-$root/.venv-mlx/bin/python}"  # override for a venv elsewhere
out="$root/tools/llm_speedtest/results/bench-$(date +%Y%m%d-%H%M%S)"
port=11435  # 8080 is often taken; 127.0.0.1 avoids an IPv6 localhost clash
mkdir -p "$out"

unload_ollama() {
  curl -s localhost:11434/api/ps | "$py" -c 'import json,sys; [print(m["name"]) for m in json.load(sys.stdin).get("models", [])]' |
    while read -r name; do curl -s localhost:11434/api/generate -d "{\"model\":\"$name\",\"keep_alive\":0}" >/dev/null; done
}

for spec in "$@"; do
  kind="${spec%%:*}"
  model="${spec#*:}"
  file="$out/$(echo "$spec" | tr '/:' '__').json"
  unload_ollama
  echo "== $spec" >&2
  if [[ "$kind" == "mlx" ]]; then
    "$(dirname "$py")/mlx_lm.server" --model "$model" --port "$port" --log-level WARNING \
      --chat-template-args '{"enable_thinking":false}' >"$out/server.log" 2>&1 &
    server=$!
    until curl -sf "127.0.0.1:$port/v1/models" >/dev/null; do
      kill -0 "$server" || { echo "server died, see $out/server.log" >&2; exit 1; }
      sleep 1
    done
    # Warm-up so the first scenario doesn't pay the model load.
    curl -s "127.0.0.1:$port/v1/chat/completions" \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" >/dev/null
    "$py" -m dialogue_bench --backend openai --base-url "http://127.0.0.1:$port" \
      --model "$model" --json --include-text >"$file"
    kill "$server"
    wait "$server" 2>/dev/null || true
  else
    curl -s localhost:11434/api/generate -d "{\"model\":\"$model\",\"prompt\":\"\"}" >/dev/null
    "$py" -m dialogue_bench --model "$model" --json --include-text >"$file"
  fi
done
echo "$out"
