from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
import sys
import os
import weaviate
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.hand_input import collect_hand_config
from utils.preflop_input import collect_preflop, PreflopRecord
from utils.flop_input import collect_flop
from utils.turn_input import collect_turn
from utils.river_input import collect_river
from utils.game_flow import POS_9MAX
from langchain_openai import ChatOpenAI
from utils.rag_pipeline import (
    build_final_rag_schema,
    enrich_user_hand_with_llm,
    search_weaviate_hybrid,
    build_nl_summary,
)
#from test_fixtures import serialize_hand_fixture, save_fixture
from weaviate.classes.init import Auth
from utils.embedding import embed
from utils.coaching import generate_coaching_advice, build_evidence_context, evidence_card_from_obj

# ---------------------------
# Glue: who reached the flop?
# ---------------------------
def participants_to_flop(pre: PreflopRecord) -> List[str]:
    """
    Seats whose FINAL preflop state is call/raise (not fold).
    Return in table order.
    """
    final = {pre.opener: "raise"}
    for a in pre.trail_actions:
        final[a.position] = a.action
    involved = [p for p, act in final.items() if act in ("call", "raise")]
    return [p for p in POS_9MAX if p in involved]

# ---------------------------
# Flop board feature extractor (lightweight, RAG-friendly)
# ---------------------------
_RANK_VAL = {r: i for i, r in enumerate("..23456789TJQKA", start=0)}  # quick map

def analyze_flop(board: str) -> Dict[str, object]:
    """
    board: "AsTd2d"
    Returns very compact features for retrieval/reranking.
    """
    cards = [board[i:i+2] for i in range(0, len(board), 2)]
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    suit_counts = {s: suits.count(s) for s in set(suits)}
    flush_level = max(suit_counts.values())  # 1..3

    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    paired_level = max(rank_counts.values()) if rank_counts else 0
    paired_level = 0 if paired_level == 1 else paired_level

    vals = sorted({_RANK_VAL[r] for r in ranks})
    vals_wheel = {14 if v != 14 else 1 for v in vals}
    span = max(vals) - min(vals)
    span_wheel = max(vals_wheel) - min(vals_wheel)
    if len(vals) == 3 and (span <= 2 or span_wheel <= 2):
        straightness = 2
    elif len(vals) == 3 and (span <= 4 or span_wheel <= 4):
        straightness = 1
    else:
        straightness = 0

    hi = max(vals)
    high_card_bucket = "H" if hi >= _RANK_VAL["Q"] else ("M" if hi >= _RANK_VAL["9"] else "L")

    if flush_level == 3:
        texture_class = "monotone"
    elif paired_level >= 2:
        texture_class = "paired"
    elif straightness >= 1:
        texture_class = "draw-heavy"
    else:
        texture_class = "dry"

    return {
        "flop_board": board,
        "flush_level": flush_level,          # 1..3
        "paired_level": 0 if paired_level == 0 else (1 if paired_level == 2 else 3),
        "straightness": straightness,        # 0 low / 1 some / 2 strong
        "high_card_bucket": high_card_bucket,# H/M/L
        "texture_class": texture_class,      # dry/paired/monotone/draw-heavy
    }

# ---------------------------
# Build a compact RAG schema
# ---------------------------
def build_rag_schema(cfg, pre: PreflopRecord, flop_rec, flop_feats: Dict[str, object]) -> Dict[str, object]:
    pre_tokens = pre.preflop_tokens
    flop_tokens = flop_rec.tokens
    line_compact = f"{pre_tokens} || {flop_tokens}"

    schema_string = f"{cfg.schema_tokens()} | preflop={pre_tokens} | flop={flop_tokens}"

    facets = {
        "pot_type": pre.pot_type,
        "players_to_flop": pre.players_to_flop,
        "positions": participants_to_flop(pre),
        "line_compact": line_compact,
        **flop_feats,
    }

    embedding_text = (
        f"{cfg.schema_tokens()}. Preflop: {pre_tokens}. "
        f"Flop: {flop_tokens}. Board is {flop_feats['texture_class']} "
        f"(flush_level={flop_feats['flush_level']}, paired_level={flop_feats['paired_level']}, "
        f"straightness={flop_feats['straightness']}, high={flop_feats['high_card_bucket']})."
    )

    return {
        "title": "User-entered hand (preflop+flop)",
        "schema_string": schema_string,
        "embedding_text": embedding_text,
        "facets": facets,
        "tokens": {
            "preflop": pre_tokens,
            "flop": flop_tokens,
        },
    }

# ---------------------------
# CLI runner
# ---------------------------

def run_workflow_from_new_hand():
    print("\n=== Hand Config ===")
    cfg = collect_hand_config()

    print("\n=== Preflop ===")
    pre = collect_preflop(cfg)

    if pre.players_to_flop <= 1:
        print("\nHand ended preflop (no flop).")

    parts = participants_to_flop(pre)
    print("\n=== Flop ===")
    flop_result = collect_flop(parts)
    flop_rec = flop_result.record
    flop_feats = analyze_flop(flop_rec.board)
    turn_result = None
    river_result = None
    if len(flop_result.remaining_positions) > 1:
        print("\n=== Turn ===")
        turn_result = collect_turn(flop_result.remaining_positions)
        print(turn_result.record.pretty())
        tokens_turn = turn_result.record.tokens

    if len(turn_result.remaining_positions) > 1:
        print("\n=== River ===")
        river_result = collect_river(turn_result.remaining_positions)
        print(river_result.record.pretty())
        if river_result.showdown_raw:
            print("Showdown:", river_result.showdown_raw)

    raw_query = build_final_rag_schema(
        cfg=cfg,
        pre=pre,
        flop_rec=flop_result.record if flop_result else None,
        flop_feats=flop_feats,
        turn_rec=turn_result.record if turn_result else None,
        river_rec=river_result.record if river_result else None,
        showdown_raw=getattr(river_result, "showdown_raw", None) if river_result else None,
    )

    llm = ChatOpenAI(model="gpt-5-mini")
    enriched_query = enrich_user_hand_with_llm(llm, raw_query, cfg)

    nl_summary  = build_nl_summary(
        cfg=cfg,
        pre=pre,
        flop_rec=flop_result.record if flop_result else None,
        turn_rec=turn_result.record if turn_result else None,
        river_rec=river_result.record if river_result else None
    )

    # fixture = serialize_hand_fixture(
    #     cfg=cfg,
    #     pre=pre,
    #     flop_result=flop_result,
    #     turn_result=turn_result,
    #     river_result=river_result,
    #     nl_summary=nl_summary,
    #     raw_query=raw_query,
    # )
    # save_fixture(fixture) 

    headers = {"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers=headers,
        skip_init_checks=True,
    )
    print("\n\nconnected to client?: ", client.is_connected())

    #3) Weaviate hybrid search (BM25 + dense) with safe hard_filters
    objects, debug = search_weaviate_hybrid(
        client=client,                   # your weaviate client
        collection_name="PokerHand",
        enriched_query_doc=enriched_query,
        top_k=3,
        alpha=0.5,
        embed_fn=embed
    )

    evidence_context, evidence_json = build_evidence_context(objects, k=3)
    advice = generate_coaching_advice(nl_summary, evidence_context, evidence_json, model="gpt-5-mini")
    
    print("\n=== Coaching Advice ===\n")
    print(advice)


    
    client.close()

def main() -> None:
    print("nun")
    

def _emit(payload: dict, filename: str) -> None:
    print("\n--- RAG Schema (preflop+flop) ---")
    print(json.dumps(payload, indent=2))
    out = Path.cwd() / filename
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out}")