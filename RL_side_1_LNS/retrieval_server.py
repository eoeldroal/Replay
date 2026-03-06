#!/usr/bin/env python3
"""ColQwen2 retrieval server — position-based search for LNS RL training.

Startup:
  1. Load deck_index.pt → row_index → 20 × (ds_idx, shard_idx, local_idx)
  2. Load all 877 embedding shards into CPU RAM (~35-38 GB)
  3. Load ColQwen2 model onto GPU for query encoding

Request flow:
  POST /search  ← [{"query": "...", "request_idx": 0, "row_index": 42}]
  1. Micro-batcher collects queries (max_wait=20ms or max_batch=64)
  2. Batch query encoding on GPU → [N, T, 128]
  3. Per-query MaxSim scoring against 20 deck embeddings
  4. Return [{"results": [{"position": 3, "score": 0.95}, ...]}]

Usage:
    conda run -n verl python RL_side_1_LNS/retrieval_server.py
    conda run -n verl python RL_side_1_LNS/retrieval_server.py --port 5002 --device cuda:0
"""

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("retrieval_server")

# ── Paths ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_DECK_INDEX = os.path.join(
    VERL_ROOT, "data", "Visual_Document_Rag", "VDR_final", "deck_index.pt"
)
DEFAULT_EMBEDDING_BASE = os.path.join(
    VERL_ROOT, "data", "Visual_Document_Rag", "VDR_processed_filtered_2",
    "embeddings", "colqwen2",
)
DEFAULT_MODEL = "vidore/colqwen2-v1.0-hf"


# ══════════════════════════════════════════════════════════════════════
# EmbeddingStore — loads all shards into CPU RAM for O(1) lookup
# ══════════════════════════════════════════════════════════════════════

class EmbeddingStore:
    """Holds all embedding shards in CPU RAM and resolves row_index → 20 embeddings."""

    def __init__(self, deck_index_path: str, embedding_base: str):
        logger.info(f"Loading deck_index from {deck_index_path}")
        t0 = time.time()
        di = torch.load(deck_index_path, map_location="cpu", weights_only=False)
        self.deck_locations: list = di["deck_locations"]
        self.datasets: list[str] = di["datasets"]
        self.shard_paths: dict[str, list[str]] = di["shard_paths"]
        logger.info(f"  deck_index loaded in {time.time() - t0:.1f}s — {len(self.deck_locations)} rows")

        # emb_store[ds_idx][shard_idx] = list of Tensor (one per image in shard)
        self.emb_store: dict[int, dict[int, list]] = defaultdict(dict)
        self._load_all_shards(embedding_base)

    def _load_all_shards(self, embedding_base: str):
        """Load all 877 shards into CPU RAM."""
        total_shards = sum(len(sp) for sp in self.shard_paths.values())
        logger.info(f"Loading {total_shards} embedding shards from {embedding_base}")
        t0 = time.time()
        loaded = 0

        for ds_idx, ds_name in enumerate(self.datasets):
            ds_dir = os.path.join(embedding_base, ds_name)
            shard_files = self.shard_paths[ds_name]
            for shard_idx, shard_file in enumerate(shard_files):
                shard_path = os.path.join(ds_dir, shard_file)
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                embeddings = shard["embeddings"]
                # embeddings is a list of Tensor [T_i, 128] (ragged)
                if not isinstance(embeddings, list):
                    # Padded tensor case: split into list
                    embeddings = list(embeddings.unbind(0))
                self.emb_store[ds_idx][shard_idx] = embeddings
                loaded += 1
                if loaded % 100 == 0:
                    logger.info(f"  loaded {loaded}/{total_shards} shards...")

        elapsed = time.time() - t0
        logger.info(f"  All {loaded} shards loaded in {elapsed:.1f}s")

    @property
    def num_rows(self) -> int:
        return len(self.deck_locations)

    def get_deck_embeddings(self, row_index: int) -> list[torch.Tensor]:
        """Return 20 embedding tensors for the given row_index."""
        if row_index < 0 or row_index >= len(self.deck_locations):
            raise ValueError(f"row_index {row_index} out of range [0, {len(self.deck_locations)})")
        locs = self.deck_locations[row_index]
        embeddings = []
        for ds_idx, shard_idx, local_idx in locs:
            emb = self.emb_store[ds_idx][shard_idx][local_idx]
            embeddings.append(emb)
        return embeddings


# ══════════════════════════════════════════════════════════════════════
# QueryEncoder — ColQwen2 text → embedding on GPU
# ══════════════════════════════════════════════════════════════════════

class QueryEncoder:
    """Encodes text queries into multi-vector embeddings using ColQwen2."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda:0"):
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor
        from transformers.utils.import_utils import is_flash_attn_2_available

        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        logger.info(f"Loading ColQwen2 model: {model_name} on {device}")
        t0 = time.time()
        self.model = ColQwen2ForRetrieval.from_pretrained(
            model_name,
            dtype=self.dtype,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        ).to(self.device).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        logger.info(f"  Model loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def encode_batch(self, queries: list[str]) -> list[torch.Tensor]:
        """Encode a batch of text queries → list of [T_i, 128] tensors.

        IMPORTANT: ColQwen2 uses left-padding for batched queries.
        Padding token embeddings are NOT zero — they must be masked out
        using attention_mask to avoid inflating MaxSim scores.
        """
        if not queries:
            return []
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        # outputs.embeddings: [batch, max_tokens, 128]
        embeddings = outputs.embeddings
        attention_mask = inputs["attention_mask"]
        # Extract only non-padding embeddings per query (critical for correct MaxSim)
        result = []
        for i in range(embeddings.shape[0]):
            mask = attention_mask[i].bool()  # [max_tokens]
            result.append(embeddings[i][mask])  # [valid_tokens, 128]
        return result


# ══════════════════════════════════════════════════════════════════════
# MaxSimScorer — ColBERT-style late interaction
# ══════════════════════════════════════════════════════════════════════

class MaxSimScorer:
    """Computes ColBERT-style MaxSim between query and document embeddings."""

    def score(
        self, query_emb: torch.Tensor, doc_embs: list[torch.Tensor]
    ) -> list[tuple[int, float]]:
        """Score a single query against multiple documents.

        Args:
            query_emb: [T, 128] query embedding
            doc_embs: list of [P_i, 128] document embeddings

        Returns:
            List of (position, score) tuples sorted by score descending.
        """
        if not doc_embs:
            raise ValueError("No document embeddings provided")

        scores = []
        for pos, doc_emb in enumerate(doc_embs):
            # Ensure same device and dtype
            q = query_emb.to(doc_emb.device, dtype=doc_emb.dtype)
            # MaxSim: for each query token, max sim across doc tokens, then sum
            sim = q @ doc_emb.T  # [T, P]
            s = sim.max(dim=1).values.sum().item()
            scores.append((pos, s))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def score_batch(
        self,
        query_embs: list[torch.Tensor],
        doc_embs_list: list[list[torch.Tensor]],
    ) -> list[list[tuple[int, float]]]:
        """Score multiple queries against their respective document sets."""
        return [
            self.score(q, docs) for q, docs in zip(query_embs, doc_embs_list)
        ]


# ══════════════════════════════════════════════════════════════════════
# Payload parsing / response formatting
# ══════════════════════════════════════════════════════════════════════

def parse_search_payload(payload: list) -> list[dict]:
    """Parse the SearchTool payload format.

    Input:  [{"query": "...", "request_idx": 0, "row_index": 42}, ...]
    Output: list of dicts with query, request_idx, row_index
    """
    items = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        items.append({
            "query": item.get("query"),
            "request_idx": item.get("request_idx", 0),
            "row_index": item.get("row_index"),
        })
    return items


def format_search_response(
    scored_results: list[list[tuple[int, float]]],
) -> list[dict]:
    """Format scored results into SearchTool response format.

    Input:  list of [list of (position, score)] per request
    Output: [{"results": [{"position": int, "score": float}, ...]}, ...]
    """
    response = []
    for results in scored_results:
        response.append({
            "results": [
                {"position": int(pos), "score": float(score)}
                for pos, score in results
            ]
        })
    return response


# ══════════════════════════════════════════════════════════════════════
# MicroBatcher — collects requests, batches GPU encoding
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _BatchItem:
    query: str
    row_index: int
    future: asyncio.Future


class MicroBatcher:
    """Collects individual search requests and processes them in GPU batches.

    - Requests enter via submit() and get an asyncio.Future
    - Background worker collects items from queue
    - When max_batch items collected OR max_wait_ms elapsed, fires a batch
    - Query encoding is batched on GPU; MaxSim is per-query
    """

    def __init__(
        self,
        encoder: QueryEncoder,
        scorer: MaxSimScorer,
        store: EmbeddingStore,
        max_batch: int = 64,
        max_wait_ms: float = 20.0,
    ):
        self.encoder = encoder
        self.scorer = scorer
        self.store = store
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the background batch worker."""
        self._worker_task = loop.create_task(self._batch_worker())

    async def submit(self, query: str, row_index: int) -> list[tuple[int, float]]:
        """Submit a single search request. Returns scored results."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_BatchItem(query=query, row_index=row_index, future=future))
        return await future

    async def _batch_worker(self):
        """Infinite loop: collect items → batch encode → score → distribute results."""
        logger.info("MicroBatcher worker started")
        while True:
            batch: list[_BatchItem] = []
            try:
                # Wait for the first item
                item = await self._queue.get()
                batch.append(item)

                # Collect more items up to max_batch or max_wait_ms
                deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000.0
                while len(batch) < self.max_batch:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            except Exception as e:
                logger.error(f"Batch collection error: {e}")
                continue

            if not batch:
                continue

            # Process the batch
            try:
                await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def _process_batch(self, batch: list[_BatchItem]):
        """Encode queries, score against docs, distribute results."""
        queries = [item.query for item in batch]
        row_indices = [item.row_index for item in batch]

        # Batch query encoding on GPU (in thread to avoid blocking event loop)
        query_embs = await asyncio.to_thread(self.encoder.encode_batch, queries)

        # Gather deck embeddings for each query (CPU lookup, fast)
        doc_embs_list = []
        for row_idx in row_indices:
            doc_embs = self.store.get_deck_embeddings(row_idx)
            doc_embs_list.append(doc_embs)

        # Score each query against its 20 docs
        # Move query embeddings to CPU for scoring against CPU doc embeddings
        def _score_all():
            results = []
            for q_emb, doc_embs in zip(query_embs, doc_embs_list):
                # Move query to CPU for MaxSim with CPU doc embeddings
                q_cpu = q_emb.cpu()
                scored = self.scorer.score(q_cpu, doc_embs)
                results.append(scored)
            return results

        all_results = await asyncio.to_thread(_score_all)

        # Distribute results to futures
        for item, result in zip(batch, all_results):
            if not item.future.done():
                item.future.set_result(result)


# ══════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════

def create_app(
    deck_index_path: str = DEFAULT_DECK_INDEX,
    embedding_base: str = DEFAULT_EMBEDDING_BASE,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda:0",
    max_batch: int = 64,
    max_wait_ms: float = 20.0,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="ColQwen2 Retrieval Server")

    # These will be initialized in the startup event
    app.state.store = None
    app.state.encoder = None
    app.state.scorer = None
    app.state.batcher = None

    @app.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info("Starting ColQwen2 Retrieval Server")
        logger.info("=" * 60)

        # 1. Load embedding store (CPU)
        app.state.store = EmbeddingStore(deck_index_path, embedding_base)

        # 2. Load query encoder (GPU)
        app.state.encoder = QueryEncoder(model_name, device)

        # 3. Create scorer
        app.state.scorer = MaxSimScorer()

        # 4. Create and start micro-batcher
        app.state.batcher = MicroBatcher(
            encoder=app.state.encoder,
            scorer=app.state.scorer,
            store=app.state.store,
            max_batch=max_batch,
            max_wait_ms=max_wait_ms,
        )
        loop = asyncio.get_running_loop()
        app.state.batcher.start(loop)

        logger.info("Server ready!")
        logger.info(f"  Rows: {app.state.store.num_rows}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Max batch: {max_batch}, Max wait: {max_wait_ms}ms")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "rows": app.state.store.num_rows if app.state.store else 0,
        }

    @app.post("/search")
    async def search(request: Request):
        payload = await request.json()
        items = parse_search_payload(payload)

        if not items:
            return JSONResponse(content=[])

        # Submit all items to the batcher concurrently
        tasks = []
        for item in items:
            query = item.get("query", "")
            row_index = item.get("row_index")
            if row_index is None:
                # Return empty results for missing row_index
                tasks.append(None)
                continue
            tasks.append(app.state.batcher.submit(query, row_index))

        # Await all results
        scored_results = []
        for task in tasks:
            if task is None:
                scored_results.append([])
            else:
                result = await task
                scored_results.append(result)

        response = format_search_response(scored_results)
        return JSONResponse(content=response)

    return app


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="ColQwen2 Retrieval Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5002, help="Bind port")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ColQwen2 model name")
    parser.add_argument("--deck-index", default=DEFAULT_DECK_INDEX, help="Path to deck_index.pt")
    parser.add_argument("--embedding-base", default=DEFAULT_EMBEDDING_BASE, help="Embedding shards dir")
    parser.add_argument("--max-batch", type=int, default=64, help="Max batch size")
    parser.add_argument("--max-wait-ms", type=float, default=20.0, help="Max wait time (ms)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    app = create_app(
        deck_index_path=args.deck_index,
        embedding_base=args.embedding_base,
        model_name=args.model,
        device=args.device,
        max_batch=args.max_batch,
        max_wait_ms=args.max_wait_ms,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        loop="uvloop",
        log_level="info",
    )


if __name__ == "__main__":
    main()
