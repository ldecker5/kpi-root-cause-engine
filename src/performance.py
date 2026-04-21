"""
performance.py
--------------
MILESTONE 14: Performance & Cost Optimization

Provides three categories of optimization for the KPI Root Cause Engine:

    1. Inference speedups:
        - ResponseCache       content-hash LRU + disk cache for chat completions
        - EmbeddingCache      content-hash cache for query embeddings
        - BatchEmbedder       batch text → embeddings in a single API call
        - embedding_dimensions  cloud-safe "quantization" via Matryoshka dims

    2. Cost-aware routing (FrugalGPT-style):
        - ModelCascade        try a cheap model first, escalate only if needed
        - TASK_TIERS          preset (cheap / standard / premium) per task type

    3. Cost accounting:
        - CostTracker         token usage + $ per call, persisted to disk
        - PRICES              OpenAI $ per 1M tokens (as of 2026-04)
        - estimate_cost       helper for (model, input, output) → dollars

All classes are safe to import without an OpenAI key — only calls that actually
hit the API require one.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Pricing table — $ per 1M tokens. Keep in sync with OpenAI pricing page.
# ---------------------------------------------------------------------------
#   input  = prompt tokens
#   output = completion tokens
#   embed  = embedding tokens (no output column)
PRICES: Dict[str, Dict[str, float]] = {
    # Chat — GPT-4.1 family (2025)
    "gpt-4.1":         {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":    {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":    {"input": 0.10,  "output": 0.40},
    # Chat — GPT-4o family
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    # Embeddings
    "text-embedding-3-small": {"embed": 0.02},
    "text-embedding-3-large": {"embed": 0.13},
}


def estimate_cost(model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
                  embed_tokens: int = 0) -> float:
    """
    Return the dollar cost for a single API call.

    Unknown models fall back to gpt-4o-mini pricing and emit no error — this keeps
    cost tracking resilient when we try a new model.
    """
    p = PRICES.get(model) or PRICES["gpt-4o-mini"]
    cost = 0.0
    if "input" in p:
        cost += (prompt_tokens / 1_000_000) * p["input"]
    if "output" in p:
        cost += (completion_tokens / 1_000_000) * p["output"]
    if "embed" in p:
        cost += (embed_tokens / 1_000_000) * p["embed"]
    return cost


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    ts: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    cost_usd: float
    cached: bool
    tag: str = ""


class CostTracker:
    """
    Thread-safe token + cost accumulator.

    Usage:
        tracker = CostTracker.get()
        tracker.record_chat(model, resp.usage, tag="agent")
        print(tracker.summary())
    """

    _instance: "CostTracker | None" = None
    _lock = threading.Lock()

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        self.records: List[UsageRecord] = []
        self._lock = threading.Lock()

    @classmethod
    def get(cls, log_path: str | Path | None = None) -> "CostTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(log_path=log_path)
            return cls._instance

    # ------------------------------------------------------------------
    def _append(self, rec: UsageRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if self.log_path:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as f:
                        f.write(json.dumps(asdict(rec)) + "\n")
                except OSError:
                    # Never let telemetry break the main flow.
                    pass

    def record_chat(self, model: str, usage: Any, *, cached: bool = False,
                    tag: str = "") -> UsageRecord:
        """Record a chat completion. Accepts OpenAI usage object or dict."""
        p_tok = _getattr_or_item(usage, "prompt_tokens", 0) or 0
        c_tok = _getattr_or_item(usage, "completion_tokens", 0) or 0
        cost = 0.0 if cached else estimate_cost(model, prompt_tokens=p_tok,
                                                completion_tokens=c_tok)
        rec = UsageRecord(
            ts=time.time(), model=model,
            prompt_tokens=p_tok, completion_tokens=c_tok, embed_tokens=0,
            cost_usd=cost, cached=cached, tag=tag,
        )
        self._append(rec)
        return rec

    def record_embed(self, model: str, n_tokens: int, *, cached: bool = False,
                     tag: str = "") -> UsageRecord:
        cost = 0.0 if cached else estimate_cost(model, embed_tokens=n_tokens)
        rec = UsageRecord(
            ts=time.time(), model=model,
            prompt_tokens=0, completion_tokens=0, embed_tokens=n_tokens,
            cost_usd=cost, cached=cached, tag=tag,
        )
        self._append(rec)
        return rec

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(r.cost_usd for r in self.records)
            cached = sum(1 for r in self.records if r.cached)
            by_model: Dict[str, Dict[str, float]] = {}
            for r in self.records:
                m = by_model.setdefault(r.model, {
                    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "embed_tokens": 0, "cost_usd": 0.0,
                })
                m["calls"] += 1
                m["prompt_tokens"] += r.prompt_tokens
                m["completion_tokens"] += r.completion_tokens
                m["embed_tokens"] += r.embed_tokens
                m["cost_usd"] += r.cost_usd
            return {
                "total_calls": len(self.records),
                "cache_hits": cached,
                "total_cost_usd": round(total, 6),
                "by_model": by_model,
            }

    def reset(self) -> None:
        with self._lock:
            self.records.clear()


def _getattr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Response cache (chat completions)
# ---------------------------------------------------------------------------

def _stable_hash(obj: Any) -> str:
    """Hash a JSON-serializable structure deterministically."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class ResponseCache:
    """
    LRU cache for chat-completion responses keyed by (model, messages, tools,
    temperature, max_tokens). Optionally mirrors entries to disk as JSON so
    repeated runs across sessions reuse hits.

    We store only the minimal fields the agent reads back: the message (role,
    content, tool_calls) and the usage dict. That keeps cache entries small
    and decoupled from OpenAI SDK internals.
    """

    def __init__(self, capacity: int = 512, persist_dir: str | Path | None = None):
        self.capacity = capacity
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list, tools: Optional[list],
             temperature: float, max_tokens: int) -> str:
        # Normalize messages / tools so small SDK shape differences don't miss.
        norm_msgs = [
            {"role": m.get("role"), "content": m.get("content"),
             "tool_call_id": m.get("tool_call_id"),
             "tool_calls": _serialize_tool_calls(m.get("tool_calls"))}
            for m in messages
        ]
        return _stable_hash({
            "model": model, "messages": norm_msgs, "tools": tools,
            "temperature": temperature, "max_tokens": max_tokens,
        })

    def _disk_path(self, key: str) -> Optional[Path]:
        return self.persist_dir / f"{key}.json" if self.persist_dir else None

    def get(self, model: str, messages: list, tools: Optional[list],
            temperature: float, max_tokens: int) -> Optional[Dict[str, Any]]:
        key = self._key(model, messages, tools, temperature, max_tokens)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        path = self._disk_path(key)
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                with self._lock:
                    self._store[key] = data
                    self._store.move_to_end(key)
                    self._evict_if_needed()
                return data
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def put(self, model: str, messages: list, tools: Optional[list],
            temperature: float, max_tokens: int,
            entry: Dict[str, Any]) -> None:
        key = self._key(model, messages, tools, temperature, max_tokens)
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            self._evict_if_needed()
        path = self._disk_path(key)
        if path:
            try:
                path.write_text(json.dumps(entry, default=str))
            except OSError:
                pass

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._store), "capacity": self.capacity}


def _serialize_tool_calls(tcs: Any) -> Any:
    """OpenAI tool_calls may be pydantic objects; normalize for hashing."""
    if tcs is None:
        return None
    out = []
    for tc in tcs:
        if isinstance(tc, dict):
            out.append({
                "id": tc.get("id"),
                "type": tc.get("type"),
                "function": tc.get("function"),
            })
        else:
            fn = getattr(tc, "function", None)
            out.append({
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None),
                "function": {
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None),
                } if fn else None,
            })
    return out


# ---------------------------------------------------------------------------
# Embedding cache & batched embedder
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """In-memory + optional disk cache keyed by (model, dims, text hash)."""

    def __init__(self, persist_dir: str | Path | None = None,
                 capacity: int = 10_000):
        self.capacity = capacity
        self._store: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, dims: Optional[int], text: str) -> str:
        return _stable_hash({"m": model, "d": dims, "t": text})

    def get(self, model: str, dims: Optional[int], text: str) -> Optional[List[float]]:
        key = self._key(model, dims, text)
        with self._lock:
            vec = self._store.get(key)
            if vec is not None:
                self._store.move_to_end(key)
                return vec
        if self.persist_dir:
            path = self.persist_dir / f"{key}.json"
            if path.exists():
                try:
                    vec = json.loads(path.read_text())
                    with self._lock:
                        self._store[key] = vec
                        self._store.move_to_end(key)
                        self._evict()
                    return vec
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    def put(self, model: str, dims: Optional[int], text: str,
            vec: List[float]) -> None:
        key = self._key(model, dims, text)
        with self._lock:
            self._store[key] = vec
            self._store.move_to_end(key)
            self._evict()
        if self.persist_dir:
            path = self.persist_dir / f"{key}.json"
            try:
                path.write_text(json.dumps(vec))
            except OSError:
                pass

    def _evict(self) -> None:
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)


class BatchEmbedder:
    """
    Batch text → embeddings in one OpenAI call, with per-item cache checks.

    OpenAI's embedding endpoint accepts an array of inputs, billed as the sum of
    tokens. Sending chunks in batches of ~96 dramatically cuts wall time vs. a
    per-chunk loop.
    """

    def __init__(self, client, model: str, *,
                 dimensions: Optional[int] = None,
                 cache: Optional[EmbeddingCache] = None,
                 tracker: Optional[CostTracker] = None,
                 batch_size: int = 96):
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.cache = cache
        self.tracker = tracker
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str], *, tag: str = "") -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        pending: List[Tuple[int, str]] = []

        if self.cache:
            for i, t in enumerate(texts):
                hit = self.cache.get(self.model, self.dimensions, t)
                if hit is not None:
                    out[i] = hit
                else:
                    pending.append((i, t))
        else:
            pending = list(enumerate(texts))

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": [t for _, t in chunk],
            }
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)

            used = getattr(resp, "usage", None)
            tok = _getattr_or_item(used, "prompt_tokens", 0) or \
                  _getattr_or_item(used, "total_tokens", 0) or 0
            if self.tracker:
                self.tracker.record_embed(self.model, tok, tag=tag)

            for (idx, text), item in zip(chunk, resp.data):
                vec = list(item.embedding)
                out[idx] = vec
                if self.cache:
                    self.cache.put(self.model, self.dimensions, text, vec)

        return [v if v is not None else [] for v in out]


# ---------------------------------------------------------------------------
# Model cascade (FrugalGPT-style)
# ---------------------------------------------------------------------------

# Ordered from cheapest to most capable. Use these names as defaults but allow
# per-task overrides via TASK_TIERS.
CASCADE_DEFAULT = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o"]

# Preset routing per task type: use the cheapest model that historically
# handles the task, escalate on signal (low confidence, parse failure, etc.).
TASK_TIERS: Dict[str, List[str]] = {
    "plan":        ["gpt-4.1-nano", "gpt-4o-mini"],   # short planning thoughts
    "classify":    ["gpt-4.1-nano", "gpt-4o-mini"],   # simple category calls
    "tool_call":   ["gpt-4o-mini", "gpt-4o"],         # agent loop — needs tools
    "rag_answer":  ["gpt-4o-mini", "gpt-4o"],         # summarization quality matters
    "vision":      ["gpt-4o-mini", "gpt-4o"],         # needs vision capability
}


@dataclass
class CascadeVerdict:
    accept: bool
    reason: str = ""


class ModelCascade:
    """
    FrugalGPT-style cascade: run the cheap model, score the answer, and only
    escalate to a bigger model if the cheap answer fails a verifier.

    `verifier(response) -> CascadeVerdict` is user-supplied. A good verifier for
    the KPI agent is "did it parse as valid tool call or structured output,
    and does the response length exceed a minimum threshold?"

    For the planning step the default verifier accepts non-empty content.
    """

    def __init__(self, tiers: Sequence[str],
                 verifier: Optional[Callable[[Any], CascadeVerdict]] = None):
        self.tiers = list(tiers)
        self.verifier = verifier or default_verifier

    def run(self, call_fn: Callable[[str], Any]) -> Tuple[Any, str]:
        """
        Try each tier until the verifier accepts. Returns (response, model_used).
        Raises the last exception if every tier errors.
        """
        last_exc: Optional[Exception] = None
        for model in self.tiers:
            try:
                resp = call_fn(model)
            except Exception as e:                      # noqa: BLE001
                last_exc = e
                continue
            verdict = self.verifier(resp)
            if verdict.accept:
                return resp, model
        if last_exc:
            raise last_exc
        # All tiers rejected — return the last response anyway; caller decides.
        return resp, self.tiers[-1]


def default_verifier(response: Any) -> CascadeVerdict:
    """Accept any non-empty text or any response containing tool calls."""
    try:
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            return CascadeVerdict(True, "tool_call_present")
        txt = (msg.content or "").strip()
        if len(txt) >= 8:
            return CascadeVerdict(True, "non_empty_text")
        return CascadeVerdict(False, "empty_or_too_short")
    except Exception as e:                              # noqa: BLE001
        return CascadeVerdict(False, f"parse_error: {e}")


# ---------------------------------------------------------------------------
# Module-wide singletons — convenient defaults for the app
# ---------------------------------------------------------------------------

_PERF_DIR = Path(__file__).parent.parent / ".perf_cache"
RESPONSE_CACHE = ResponseCache(capacity=512, persist_dir=_PERF_DIR / "responses")
EMBEDDING_CACHE = EmbeddingCache(capacity=10_000, persist_dir=_PERF_DIR / "embeddings")
COST_TRACKER = CostTracker.get(log_path=_PERF_DIR / "usage.jsonl")
