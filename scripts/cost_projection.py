"""
cost_projection.py
------------------
MILESTONE 14: Cost Projection Simulator

Estimates monthly cost for the KPI Root Cause Engine under different load
profiles, cache-hit rates, and cascade split ratios.

Covers three cost categories:

    1. API usage  — OpenAI chat + embedding tokens
    2. Compute    — Streamlit host (flat monthly or per-hour)
    3. Storage    — Chroma DB on disk + embedding cache on disk

Usage:
    python scripts/cost_projection.py
    python scripts/cost_projection.py --profile heavy
    python scripts/cost_projection.py --custom 500 0.4 0.7

This script is read-only — it only does math. It never hits the OpenAI API.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Make src importable when running from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.performance import estimate_cost, PRICES   # noqa: E402


# ---------------------------------------------------------------------------
# Per-request token model (measured from live traces; tune as needed)
# ---------------------------------------------------------------------------

@dataclass
class TokenProfile:
    """
    Tokens consumed by one end-to-end investigation.

    Breakdown reflects the current ReAct loop:
      - 1 planning call per round (nano-tier, small)
      - 1 tool-call round per round (mini-tier, larger)
      - ~4 rounds average
      - 1 RAG generation per investigation (mini-tier)
      - ~8 query embeddings (nano-sized)
    """
    plan_calls:     int = 4
    plan_in_tok:    int = 400     # system + planning prompt + state summary
    plan_out_tok:   int = 80

    tool_calls:     int = 4
    tool_in_tok:    int = 1_800   # system + messages + tool results grow
    tool_out_tok:   int = 250

    rag_gen_calls:  int = 1
    rag_gen_in_tok: int = 1_400   # prompt + retrieved context
    rag_gen_out_tok:int = 400

    embed_queries:  int = 8
    embed_in_tok:   int = 120


@dataclass
class LoadProfile:
    name: str
    investigations_per_day: int
    cache_hit_rate: float       # 0..1  — fraction served from cache
    cascade_nano_rate: float    # 0..1  — fraction of cascade tiers that stop at nano


# Three preset load profiles covering the realistic deployment range.
PROFILES: Dict[str, LoadProfile] = {
    "light":  LoadProfile("light",  50,    0.30, 0.70),
    "medium": LoadProfile("medium", 500,   0.50, 0.75),
    "heavy":  LoadProfile("heavy",  5_000, 0.65, 0.80),
}


# ---------------------------------------------------------------------------
# Infrastructure cost assumptions (update quarterly)
# ---------------------------------------------------------------------------

@dataclass
class InfraCost:
    """Fixed monthly infrastructure costs."""
    streamlit_host_usd: float = 20.0       # Streamlit Cloud / small VM
    vector_store_gb_usd: float = 0.10      # GCS / S3 block storage per GB
    vector_store_gb: float = 0.15          # Chroma ~150MB at current KB size
    cache_storage_gb_usd: float = 0.10
    cache_storage_gb: float = 0.05

    def monthly(self) -> float:
        return (
            self.streamlit_host_usd
            + self.vector_store_gb * self.vector_store_gb_usd
            + self.cache_storage_gb * self.cache_storage_gb_usd
        )


# ---------------------------------------------------------------------------
# Projection logic
# ---------------------------------------------------------------------------

PLAN_MODEL_CHEAP = "gpt-4.1-nano"
PLAN_MODEL_EXPENSIVE = "gpt-4o-mini"
TOOL_MODEL = "gpt-4o-mini"
RAG_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


def project_api_cost_per_call(profile: LoadProfile, tp: TokenProfile) -> Dict[str, float]:
    """
    Per-investigation API cost in dollars, given cache + cascade assumptions.

    Cache-hit investigations cost $0 for API calls they hit on. We approximate
    this as: total_api_cost * (1 - cache_hit_rate).
    """
    # Planning calls — split between nano and mini based on cascade_nano_rate.
    plan_nano = tp.plan_calls * profile.cascade_nano_rate
    plan_mini = tp.plan_calls - plan_nano
    plan_cost = (
        plan_nano * estimate_cost(PLAN_MODEL_CHEAP,
                                  tp.plan_in_tok, tp.plan_out_tok)
        + plan_mini * estimate_cost(PLAN_MODEL_EXPENSIVE,
                                    tp.plan_in_tok, tp.plan_out_tok)
    )

    # Tool-call rounds stay on mini (tools require the mid-tier).
    tool_cost = tp.tool_calls * estimate_cost(
        TOOL_MODEL, tp.tool_in_tok, tp.tool_out_tok,
    )

    rag_cost = tp.rag_gen_calls * estimate_cost(
        RAG_MODEL, tp.rag_gen_in_tok, tp.rag_gen_out_tok,
    )

    embed_cost = tp.embed_queries * estimate_cost(
        EMBED_MODEL, embed_tokens=tp.embed_in_tok,
    )

    uncached = plan_cost + tool_cost + rag_cost + embed_cost
    effective = uncached * (1.0 - profile.cache_hit_rate)

    return {
        "plan": plan_cost,
        "tool": tool_cost,
        "rag": rag_cost,
        "embed": embed_cost,
        "uncached_per_call": uncached,
        "effective_per_call": effective,
    }


def project_monthly(profile: LoadProfile,
                    tp: TokenProfile | None = None,
                    infra: InfraCost | None = None) -> Dict[str, float]:
    tp = tp or TokenProfile()
    infra = infra or InfraCost()

    per_call = project_api_cost_per_call(profile, tp)
    monthly_calls = profile.investigations_per_day * 30
    api_monthly = per_call["effective_per_call"] * monthly_calls
    infra_monthly = infra.monthly()

    return {
        "profile": profile.name,
        "investigations_per_day": profile.investigations_per_day,
        "cache_hit_rate": profile.cache_hit_rate,
        "cascade_nano_rate": profile.cascade_nano_rate,
        "monthly_calls": monthly_calls,
        "per_call_uncached_usd": round(per_call["uncached_per_call"], 6),
        "per_call_effective_usd": round(per_call["effective_per_call"], 6),
        "api_monthly_usd": round(api_monthly, 2),
        "infra_monthly_usd": round(infra_monthly, 2),
        "total_monthly_usd": round(api_monthly + infra_monthly, 2),
        "breakdown_per_call": {k: round(v, 6)
                               for k, v in per_call.items() if "_per_call" not in k},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_projection(p: Dict[str, float]) -> None:
    print(f"\n=== Profile: {p['profile']} "
          f"({p['investigations_per_day']}/day, "
          f"cache={p['cache_hit_rate']:.0%}, "
          f"nano_route={p['cascade_nano_rate']:.0%}) ===")
    print(f"  Per-call uncached : ${p['per_call_uncached_usd']:.4f}")
    print(f"  Per-call effective: ${p['per_call_effective_usd']:.4f}")
    print(f"  Monthly API       : ${p['api_monthly_usd']:>8,.2f} "
          f"({p['monthly_calls']:,} calls)")
    print(f"  Monthly infra     : ${p['infra_monthly_usd']:>8,.2f}")
    print(f"  Monthly TOTAL     : ${p['total_monthly_usd']:>8,.2f}")
    print("  Per-call breakdown:")
    for k, v in p["breakdown_per_call"].items():
        print(f"     {k:>6}: ${v:.5f}")


def print_pricing_table() -> None:
    print("\n=== Model pricing ($ per 1M tokens) ===")
    print(f"  {'model':<24} {'input':>8} {'output':>8} {'embed':>8}")
    for model, p in PRICES.items():
        row = f"  {model:<24} "
        row += f"{p.get('input', 0):>8.2f} "
        row += f"{p.get('output', 0):>8.2f} "
        row += f"{p.get('embed', 0):>8.2f}"
        print(row)


def print_bottlenecks() -> None:
    print("\n=== Scaling bottlenecks (in order of impact) ===")
    print("  1. Streamlit single-process   — 1 worker serves all tabs; CPU-bound")
    print("                                   at ~5 concurrent investigations.")
    print("  2. Sequential tool calls      — agent waits on each tool; a parallel")
    print("                                   executor would cut median latency ~40%.")
    print("  3. OpenAI rate limits         — gpt-4o-mini TPM caps a single org")
    print("                                   at ~200 req/min without tier upgrade.")
    print("  4. Chroma in-process store    — fine to ~100k vectors; switch to a")
    print("                                   hosted vector DB beyond that.")
    print("  5. In-memory rate limiter     — resets on restart; use Redis for HA.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(PROFILES), default=None,
                    help="Preset load profile. If omitted, prints all three.")
    ap.add_argument("--custom", nargs=3, metavar=("PER_DAY", "CACHE", "NANO"),
                    help="Custom projection: investigations/day cache_hit nano_rate")
    ap.add_argument("--show-pricing", action="store_true")
    ap.add_argument("--show-bottlenecks", action="store_true")
    args = ap.parse_args()

    if args.show_pricing:
        print_pricing_table()

    profiles_to_run: List[LoadProfile] = []
    if args.custom:
        per_day, cache, nano = args.custom
        profiles_to_run.append(LoadProfile(
            name="custom",
            investigations_per_day=int(per_day),
            cache_hit_rate=float(cache),
            cascade_nano_rate=float(nano),
        ))
    elif args.profile:
        profiles_to_run.append(PROFILES[args.profile])
    else:
        profiles_to_run.extend(PROFILES.values())

    for prof in profiles_to_run:
        print_projection(project_monthly(prof))

    if args.show_bottlenecks:
        print_bottlenecks()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
