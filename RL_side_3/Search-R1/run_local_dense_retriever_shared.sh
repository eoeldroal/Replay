#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Shared-GPU assumption:
# - Retriever sees all 8 GPUs, but we reorder so that the encoder hotspot lands
#   on physical GPU7 first.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7,0,1,2,3,4,5,6}"

INDEX_PATH="${INDEX_PATH:-$HOME/data/searchR1_retriever/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$HOME/data/searchR1_retriever/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
PORT="${PORT:-8000}"

cd "$ROOT_DIR"

python examples/search_r1_like/local_dense_retriever/retrieval_server.py \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk "$TOPK" \
  --retriever_name e5 \
  --retriever_model "$RETRIEVER_MODEL" \
  --faiss_gpu

