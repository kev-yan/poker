# utils/rag_builder.py
from __future__ import annotations

from typing import Dict, List, Optional

from utils.game_flow import POS_9MAX
from utils.preflop_input import PreflopRecord
# Street result/record types (import only for typing clarity)
from utils.flop_input import FlopRecord
from utils.turn_input import TurnRecord
from utils.river_input import RiverRecord

# Reuse your rank map for coarse straightness checks
_RANK_VAL = {r: i for i, r in enumerate("..23456789TJQKA", start=0)}

def build_final_rag_schema(
    cfg,                                  # HandConfig
    pre: PreflopRecord,
    flop_rec: Optional[FlopRecord] = None,
    flop_feats: Optional[Dict[str, object]] = None,
    turn_rec: Optional[TurnRecord] = None,
    river_rec: Optional[RiverRecord] = None,
    showdown_raw: Optional[str] = None,
) -> Dict[str, object]:
    """
    Combine all streets into a single Weaviate-ready RAG schema object.
    Safe to call with partial hands (e.g., preflop-only, flop-only, etc.).
    """

    # ---- Tokens (position-aware, amounts omitted) ----
    tokens = {"preflop": pre.preflop_tokens}
    street_tokens: List[str] = [pre.preflop_tokens]

    if flop_rec:
        tokens["flop"] = flop_rec.tokens
        street_tokens.append(flop_rec.tokens)
    if turn_rec:
        tokens["turn"] = turn_rec.tokens
        street_tokens.append(turn_rec.tokens)
    if river_rec:
        tokens["river"] = river_rec.tokens
        street_tokens.append(river_rec.tokens)

    line_compact = " || ".join(street_tokens)

    # ---- Boards / Features ----
    flop_board = getattr(flop_rec, "board", None)
    turn_card  = getattr(turn_rec, "board", None)
    river_card = getattr(river_rec, "board", None)

    # Minimal runout features across all visible board cards
    runout_feats = _analyze_runout(flop_board, turn_card, river_card)

    # If caller didn’t pass flop_feats, derive ultra-light from runout_feats (fallback)
    if flop_board and not flop_feats:
        flop_feats = {
            "flop_board": flop_board,
            "flush_level": runout_feats.get("flop_flush_level", 1),
            "paired_level": runout_feats.get("flop_paired_level", 0),
            "straightness": runout_feats.get("flop_straightness", 0),
            "high_card_bucket": runout_feats.get("flop_high_card_bucket", "M"),
            "texture_class": runout_feats.get("flop_texture_class", "dry"),
        }

    # ---- Facets you’ll filter/rerank on ----
    positions_flop = _participants_to_flop(pre)
    facets = {
        "pot_type": pre.pot_type,
        "players_to_flop": pre.players_to_flop,
        "heads_up": pre.players_to_flop == 2,
        "positions": positions_flop,
        "stack_depth": getattr(cfg, "stack_depth", None),   # "deep/medium/short" from HandConfig
        "line_compact": line_compact,

        # Boards
        "flop_board": flop_board,
        "turn_card": turn_card,
        "river_card": river_card,

        # Flop features (preferred – your analyzer)
        **(flop_feats or {}),

        # Runout features across 4–5 cards
        "runout_flush_max": runout_feats["runout_flush_max"],           # 1..5 max same suit
        "flush_by_river": runout_feats["flush_by_river"],               # bool
        "paired_by_river": runout_feats["paired_by_river"],             # 0 none / 2 pair / 3 trips / 4 quads
        "straightness_runout": runout_feats["straightness_runout"],     # 0/1/2 (coarse)
        "runout_texture_class": runout_feats["runout_texture_class"],   # dry/draw-heavy/paired/monotone
    }

    # ---- Schema + embedding text (concise, texture-focused) ----
    pre_tokens = tokens["preflop"]
    flop_tokens = tokens.get("flop")
    turn_tokens = tokens.get("turn")
    river_tokens = tokens.get("river")

    schema_string = (
        f"{_safe_schema_tokens(cfg)} "
        f"| preflop={pre_tokens}"
        + (f" | flop={flop_tokens}" if flop_tokens else "")
        + (f" | turn={turn_tokens}" if turn_tokens else "")
        + (f" | river={river_tokens}" if river_tokens else "")
    )

    # Build an NL embedding text that avoids exact ranks/suits; focuses on structure/texture
    desc_chunks = [
        _safe_schema_tokens(cfg),
        f"Preflop: {pre_tokens}."
    ]
    if flop_tokens:
        desc_chunks.append(f"Flop: {flop_tokens}. Board is {facets.get('texture_class', 'unknown')} "
                           f"(flush={facets.get('flush_level')}, paired={facets.get('paired_level')}, "
                           f"straightness={facets.get('straightness')}).")
    if turn_tokens:
        desc_chunks.append(f"Turn: {turn_tokens}.")
    if river_tokens:
        desc_chunks.append(f"River: {river_tokens}. Runout is {facets.get('runout_texture_class')} "
                           f"(flush_by_river={facets.get('flush_by_river')}, "
                           f"paired_by_river={facets.get('paired_by_river')}, "
                           f"straightness={facets.get('straightness_runout')}).")

    embedding_text = " ".join(desc_chunks)

    # ---- Hard/soft retrieval controls (tune as you go) ----
    hard_filters = {"pot_type": pre.pot_type}
    soft_signals = {
        "texture_class": facets.get("texture_class"),
        "runout_texture_class": facets.get("runout_texture_class"),
        "heads_up": facets["heads_up"],
        "stack_depth": facets.get("stack_depth"),
    }

    # Do not surface results in embeddings; keep raw showdown text out of embedding_text
    payload = {
        "title": "User-entered hand",
        "schema_string": schema_string,
        "embedding_text": embedding_text,
        "facets": facets,
        "tokens": tokens,
        "hard_filters": hard_filters,
        "soft_signals": soft_signals,
    }
    if showdown_raw:
        payload["showdown_raw"] = showdown_raw  # keep but don’t embed

    return payload


# --------------------------
# Helpers
# --------------------------

def _safe_schema_tokens(cfg) -> str:
    """
    Use HandConfig.schema_tokens() if available, otherwise fall back to a compact line.
    Avoid crashing if schema_tokens has been customized.
    """
    try:
        return cfg.schema_tokens()
    except Exception:
        # Minimal fallback
        s = getattr(cfg, "stakes_bb", None)
        pos = getattr(cfg, "hero_position", "?")
        hand = getattr(cfg, "hero_hand", "??")
        return f"stakes=? table=? pos={pos} hero={hand}"

def _participants_to_flop(pre: PreflopRecord) -> List[str]:
    """
    Seats whose *final* preflop state is call/raise (not fold), in table order.
    Mirrors your participants_to_flop helper.
    """
    final = {pre.opener: "raise"}
    for a in pre.trail_actions:
        final[a.position] = a.action
    involved = [p for p, act in final.items() if act in ("call", "raise")]
    return [p for p in POS_9MAX if p in involved]

def _analyze_runout(
    flop_board: Optional[str],
    turn_card: Optional[str],
    river_card: Optional[str],
) -> Dict[str, object]:
    """
    Coarse runout features across the visible board (3–5 cards):
    - max suit count (-> flush_by_river)
    - paired strength across ranks
    - straightness (coarse) across 4–5 cards
    Also emits *flop-only* features (for fallback when no dedicated flop analyzer is passed).
    """
    # Helper to parse 2-char cards
    def _split_cards(s: str) -> List[str]:
        return [s[i:i+2] for i in range(0, len(s), 2)] if s else []

    flop_cards = _split_cards(flop_board)        # len 3 or []
    all_cards = flop_cards + ([turn_card] if turn_card else []) + ([river_card] if river_card else [])

    # ---- Flop-only (fallbacks) ----
    flop_ranks = [c[0] for c in flop_cards]
    flop_suits = [c[1] for c in flop_cards]
    flop_suit_counts = {s: flop_suits.count(s) for s in set(flop_suits)} or {"x": 1}
    flop_flush_level = max(flop_suit_counts.values())
    flop_rank_counts = {r: flop_ranks.count(r) for r in set(flop_ranks)} or {}
    flop_paired_level = 0 if not flop_rank_counts else max(flop_rank_counts.values())
    flop_paired_level = 0 if flop_paired_level == 1 else flop_paired_level
    flop_vals = sorted({_RANK_VAL[r] for r in flop_ranks}) if flop_ranks else []
    straightness_flop = _coarse_straightness(vals=flop_vals)
    flop_hi = max(flop_vals) if flop_vals else _RANK_VAL["T"]  # bias medium/high if unknown
    flop_high_bucket = "H" if flop_hi >= _RANK_VAL["Q"] else ("M" if flop_hi >= _RANK_VAL["9"] else "L")
    if flop_flush_level == 3:
        flop_tex = "monotone"
    elif flop_paired_level >= 2:
        flop_tex = "paired"
    elif straightness_flop >= 1:
        flop_tex = "draw-heavy"
    else:
        flop_tex = "dry"

    # ---- Runout features (4–5 cards) ----
    ranks = [c[0] for c in all_cards]
    suits = [c[1] for c in all_cards]

    suit_counts = {s: suits.count(s) for s in set(suits)} or {"x": 1}
    runout_flush_max = max(suit_counts.values())

    rank_counts = {r: ranks.count(r) for r in set(ranks)} or {}
    # paired_by_river: 0 none / 2 pair / 3 trips / 4 quads (values echo your earlier style)
    paired_raw = 0 if not rank_counts else max(rank_counts.values())
    if paired_raw <= 1:
        paired_by_river = 0
    elif paired_raw == 2:
        paired_by_river = 2
    elif paired_raw == 3:
        paired_by_river = 3
    else:
        paired_by_river = 4

    vals = sorted({_RANK_VAL[r] for r in ranks}) if ranks else []
    straightness_runout = _coarse_straightness(vals=vals, allow_len=len(all_cards))

    if runout_flush_max >= 5:
        runout_tex = "monotone"
    elif paired_by_river >= 2:
        runout_tex = "paired"
    elif straightness_runout >= 1:
        runout_tex = "draw-heavy"
    else:
        runout_tex = "dry"

    return {
        # flop fallback
        "flop_flush_level": flop_flush_level,
        "flop_paired_level": 0 if flop_paired_level == 0 else (1 if flop_paired_level == 2 else 3),
        "flop_straightness": straightness_flop,
        "flop_high_card_bucket": flop_high_bucket,
        "flop_texture_class": flop_tex,

        # runout
        "runout_flush_max": runout_flush_max,
        "flush_by_river": runout_flush_max >= 5,
        "paired_by_river": paired_by_river,
        "straightness_runout": straightness_runout,
        "runout_texture_class": runout_tex,
    }

def _coarse_straightness(vals: List[int], allow_len: int = 3) -> int:
    """
    Coarse straightness 0/1/2:
    - 2: there exists a tight window (span <= 3) covering min(allow_len, 4) or more unique ranks
    - 1: there exists a looser window (span <= 5) with at least 4 unique ranks
    - 0: otherwise
    Treat A as both high and potential wheel (A=14 and also consider 1).
    """
    if not vals:
        return 0
    uniq = sorted(set(vals))
    # also consider wheel by mapping A->1
    uniq_wheel = sorted({1 if v == _RANK_VAL["A"] else v for v in uniq})

    def window_score(arr: List[int], need: int) -> int:
        n = len(arr)
        if n < need:
            return 0
        for i in range(n - need + 1):
            span = arr[i + need - 1] - arr[i]
            if need >= 4 and span <= 3:
                return 2
            if need >= 4 and span <= 5:
                return 1
        return 0

    need = 4 if allow_len >= 4 else min(allow_len, 3)
    return max(window_score(uniq, need), window_score(uniq_wheel, need))
