#!/usr/bin/env bash
# Download Qwen model files from Hugging Face into ./models/<NAME>.
#
# Usage:
#   ./download-models Qwen3.5-4B                 # config + tokenizer + weights (~9.3 GB)
#   ./download-models Qwen3.8-27B                # config + tokenizer + weights (~55.6 GB)
#   ./download-models --config-only Qwen3.5-4B   # config + tokenizer + index only (no weights)
#   ./download-models --dry-run Qwen3.8-27B      # print the download plan, download nothing
#
# --config-only is enough for `inspect`, `tokenize`, and the M0/M4 oracles.
# Downloads resume automatically (curl -C -); re-running skips completed files.
# Requires: bash, curl, python3 (to read the shard index).
set -euo pipefail

CONFIG_ONLY=0
DRY_RUN=0
NAME=""
for arg in "$@"; do
  case "$arg" in
    --config-only) CONFIG_ONLY=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
    -*)            echo "unknown option: $arg" >&2; exit 2 ;;
    *)             NAME="$arg" ;;
  esac
done
if [ -z "$NAME" ]; then
  echo "usage: ./download-models [--config-only] [--dry-run] MODEL_NAME" >&2
  echo "  e.g. ./download-models Qwen3.5-4B   (or Qwen3.8-27B)" >&2
  exit 2
fi

BASE="https://huggingface.co/Qwen/${NAME}/resolve/main"
DIR="models/${NAME}"
SMALL_FILES=(config.json tokenizer.json tokenizer_config.json model.safetensors.index.json)

# Download one file with resume; treat "already complete" (curl exit 33) as OK.
dl() {
  local url="$1" dest="$2" rc=0
  curl -fL --retry 3 --continue-at - --progress-bar -o "$dest" "$url" || rc=$?
  if [ "$rc" -eq 33 ]; then
    echo "  already complete: $dest"
  elif [ "$rc" -ne 0 ]; then
    echo "  FAILED ($rc): $url" >&2
    return "$rc"
  fi
}

# Enumerate shard filenames from the index JSON (read from a path or stdin).
shards_from_index() {
  python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(sorted(set(d['weight_map'].values()))))"
}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Plan for ${NAME} -> ${DIR}/"
  for f in "${SMALL_FILES[@]}"; do echo "  ${BASE}/${f}"; done
  echo "  ${BASE}/chat_template.jinja   (optional)"
  if [ "$CONFIG_ONLY" -eq 0 ]; then
    echo "  ...plus weight shards listed in the index:"
    curl -fsSL "${BASE}/model.safetensors.index.json" | shards_from_index | sed "s#^#    ${BASE}/#"
  fi
  exit 0
fi

mkdir -p "$DIR"
echo "Downloading ${NAME} into ${DIR}/"
for f in "${SMALL_FILES[@]}"; do
  echo "  $f"
  dl "${BASE}/${f}" "${DIR}/${f}"
done
# chat_template.jinja exists for some models (e.g. 27B); ignore if absent.
curl -fL --retry 3 -o "${DIR}/chat_template.jinja" "${BASE}/chat_template.jinja" 2>/dev/null \
  && echo "  chat_template.jinja" || true

if [ "$CONFIG_ONLY" -eq 1 ]; then
  echo "Done (config only). Enough for inspect/tokenize and the M0/M4 oracles."
  exit 0
fi

echo "Weight shards:"
mapfile -t SHARDS < <(shards_from_index < "${DIR}/model.safetensors.index.json")
for s in "${SHARDS[@]}"; do
  echo "  $s"
  dl "${BASE}/${s}" "${DIR}/${s}"
done

echo "Done. ${NAME} is ready in ${DIR}/"
echo "Run: naive-infer inspect -model ${NAME}"
