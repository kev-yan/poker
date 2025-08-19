# utils/rag_pipeline.py
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Callable
import json
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from weaviate.classes.query import MetadataQuery

# your modules
from utils.game_flow import POS_9MAX
from utils.preflop_input import PreflopRecord
from utils.flop_input import FlopRecord
from utils.turn_input import TurnRecord
from utils.river_input import RiverRecord

# --------------------------
# 1) Merge all streets -> query doc (tokens + features)
# --------------------------

_RANK_VAL = {r: i for i, r in enumerate("..23456789TJQKA", start=0)}

def build_final_rag_schema(
    cfg,                                  # HandConfig
    pre: PreflopRecord,
    flop_rec: Optional[FlopRecord] = None,
    flop_feats: Optional[Dict[str, Any]] = None,
    turn_rec: Optional[TurnRecord] = None,
    river_rec: Optional[RiverRecord] = None,
    showdown_raw: Optional[str] = None,
) -> Dict[str, Any]:
    tokens = {"preflop": pre.preflop_tokens}
    streets_ordered: List[str] = [pre.preflop_tokens]

    flop_board = getattr(flop_rec, "board", None)
    turn_card  = getattr(turn_rec, "board", None)
    river_card = getattr(river_rec, "board", None)

    if flop_rec:
        tokens["flop"] = flop_rec.tokens
        streets_ordered.append(flop_rec.tokens)
    if turn_rec:
        tokens["turn"] = turn_rec.tokens
        streets_ordered.append(turn_rec.tokens)
    if river_rec:
        tokens["river"] = river_rec.tokens
        streets_ordered.append(river_rec.tokens)

    line_compact = " || ".join(streets_ordered)

    # Runout features (coarse, good for retrieval)
    runout = _analyze_runout(flop_board, turn_card, river_card)

    # If no external flop_feats provided, fallback from runout
    if flop_board and not flop_feats:
        flop_feats = {
            "flop_board": flop_board,
            "flush_level": runout.get("flop_flush_level", 1),
            "paired_level": runout.get("flop_paired_level", 0),
            "straightness": runout.get("flop_straightness", 0),
            "high_card_bucket": runout.get("flop_high_card_bucket", "M"),
            "texture_class": runout.get("flop_texture_class", "dry"),
        }

    positions_flop = _participants_to_flop(pre)

    facets = {
        "pot_type": pre.pot_type,
        "players_to_flop": pre.players_to_flop,
        "heads_up": pre.players_to_flop == 2,
        "positions": positions_flop,
        "stack_depth": getattr(cfg, "stack_depth", None),
        "line_compact": line_compact,

        "flop_board": flop_board,
        "turn_card": turn_card,
        "river_card": river_card,

        **(flop_feats or {}),

        "runout_flush_max": runout["runout_flush_max"],
        "flush_by_river": runout["flush_by_river"],
        "paired_by_river": runout["paired_by_river"],
        "straightness_runout": runout["straightness_runout"],
        "runout_texture_class": runout["runout_texture_class"],
    }

    schema_string = (
        f"{_safe_schema_tokens(cfg)} "
        f"| preflop={tokens['preflop']}"
        + (f" | flop={tokens['flop']}" if 'flop' in tokens else "")
        + (f" | turn={tokens['turn']}" if 'turn' in tokens else "")
        + (f" | river={tokens['river']}" if 'river' in tokens else "")
    )

    # vector text: short, structure+texture focused (no exact ranks needed)
    pieces = [
        _safe_schema_tokens(cfg),
        f"Preflop: {tokens['preflop']}."
    ]
    if 'flop' in tokens:
        pieces.append(f"Flop: {tokens['flop']}. "
                      f"Board is {facets.get('texture_class','unknown')} "
                      f"(flush={facets.get('flush_level')}, paired={facets.get('paired_level')}, "
                      f"straightness={facets.get('straightness')}).")
    if 'turn' in tokens:
        pieces.append(f"Turn: {tokens['turn']}.")
    if 'river' in tokens:
        pieces.append(f"River: {tokens['river']}. Runout {facets.get('runout_texture_class')} "
                      f"(flush_by_river={facets.get('flush_by_river')}, "
                      f"paired_by_river={facets.get('paired_by_river')}, "
                      f"straightness={facets.get('straightness_runout')}).")

    embedding_text = " ".join(pieces)

    # use pot_type as a safe default hard gate; you can add street_focus when it’s clearly about a street
    hard_filters = {"pot_type": pre.pot_type}
    soft_signals = {
        "texture_class": facets.get("texture_class"),
        "runout_texture_class": facets.get("runout_texture_class"),
        "heads_up": facets["heads_up"],
        "stack_depth": facets.get("stack_depth"),
    }

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
        payload["showdown_raw"] = showdown_raw

    return payload


# --------------------------
# 2) LLM enrichment: map user doc -> corpus facet schema
# --------------------------

def enrich_user_hand_with_llm(
    llm,                   # e.g., langchain_openai.ChatOpenAI(model="gpt-4o-mini")
    raw_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask the LLM to produce the same facet keys your RAG entries use:
    - hero_hand_class, blocker_pattern
    - board texture class (already present but LLM can refine)
    - spr_bucket, stack_depth
    - street_events (e.g., flop c-bet, turn barrel, river jam), line tokens
    - pot_type (confirm), heads_up
    Returns a merged dict: raw_doc + {"enriched": {...}}.
    """
    system = (
        "You are a poker strategy normalizer. Convert a user-entered hand into a compact JSON facet schema "
        "that matches an existing RAG index of poker hands. Focus on structural features and standardized tags. "
        "Do NOT invent exact cards; use texture words (monotone, paired, draw-heavy, dry)."
    )
    # Keep the prompt tight: give tokens + coarse features; ask for the exact keys you index on.
    user = {
        "raw_schema_string": raw_doc["schema_string"],
        "tokens": raw_doc["tokens"],
        "facets_hint": {
            "pot_type": raw_doc["facets"].get("pot_type"),
            "heads_up": raw_doc["facets"].get("heads_up"),
            "stack_depth": raw_doc["facets"].get("stack_depth"),
            "texture_class": raw_doc["facets"].get("texture_class"),
            "runout_texture_class": raw_doc["facets"].get("runout_texture_class"),
        },
        "embedding_text": raw_doc["embedding_text"],
        "need_keys": [
            "hero_hand_class", "blocker_pattern",
            "spr_bucket", "stack_depth",
            "texture_class", "runout_texture_class",
            "street_events",           # list like ["flop c-bet","turn barrel","river jam"]
            "positions",               # confirm/normalize
            "line_compact",            # position-aware tokens joined
            "hard_filters",            # suggest safe gates like {"pot_type":"3-bet"}
            "soft_signals",            # texture class, heads_up, etc.
            "tags"                     # standardized tags for retrieval pivots
        ]
    }
    # Minimal structured output request
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "RagFacetSchema",
            "schema": {
                "type": "object",
                "properties": {
                    "hero_hand_class": {"type": "string"},
                    "blocker_pattern": {"type": ["string", "null"]},
                    "spr_bucket": {"type": "string"},
                    "stack_depth": {"type": "string"},
                    "texture_class": {"type": "string"},
                    "runout_texture_class": {"type": "string"},
                    "street_events": {"type": "array", "items": {"type": "string"}},
                    "positions": {"type": "array", "items": {"type": "string"}},
                    "line_compact": {"type": "string"},
                    "hard_filters": {"type": "object"},
                    "soft_signals": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["hero_hand_class","spr_bucket","texture_class","line_compact","tags"]
            }
        }
    }

    resp = llm.invoke(
        [{"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)}],
        response_format=schema,
    )

    enriched = _to_dict(resp)

    out = dict(raw_doc)
    out.setdefault("facets", {})
    out.setdefault("soft_signals", {})
    out.setdefault("hard_filters", {})
    #print("OUT:", out)

    # Optionally merge some fields upward for convenience
    out["facets"].update({
        "hero_hand_class": enriched.get("hero_hand_class"),
        "spr_bucket": enriched.get("spr_bucket"),
        "stack_depth": enriched.get("stack_depth", out["facets"].get("stack_depth")),
    })

    out["soft_signals"].update(enriched.get("soft_signals", {}))
    out["hard_filters"].update(enriched.get("hard_filters", {}))
    out["facets"]["tags"] = enriched.get("tags", [])
    out["facets"]["street_events"] = enriched.get("street_events", [])
    out["facets"]["line_compact"] = enriched.get("line_compact", out["facets"].get("line_compact"))

    return out



# --------------------------
# 3) Weaviate hybrid search helper
# --------------------------
from weaviate.classes.query import Filter
def search_weaviate_hybrid(
    client,
    collection_name: str,
    enriched_query_doc: Dict[str, Any],
    top_k: int = 8,
    alpha: float = 0.5,
    return_props: Optional[List[str]] = None,
    # NEW: inject the same embedding fn you used at ingest (384-D)
    embed_fn: Callable[[str], List[float]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    col = client.collections.get(collection_name)

    query_text = enriched_query_doc["embedding_text"]
    hf = enriched_query_doc.get("hard_filters") or {}

    # Build simple AND filter for equality keys
    flt = None
    if hf:
        for k, v in hf.items():
            f = Filter.by_property(k).equal(v)
            flt = f if flt is None else (flt & f)

    props = return_props or [
        "title", "schema_string", "embedding_text", "annotated_coaching_description",
        "pot_type", "texture_class", "tags", "line_compact"
    ]

    # 👇 Embed the query text with YOUR 384-D model and pass it explicitly
    vec = None
    if embed_fn is not None:
        vec = embed_fn(query_text)
        # Optional sanity check:
        if hasattr(vec, "__len__"):
            assert len(vec) == 384, f"Expected 384-D, got {len(vec)}"

    resp = col.query.hybrid(
        query=query_text,               # BM25 side
        vector=vec,                     # dense side (384-D)
        alpha=alpha,                    # blend
        limit=top_k,
        filters=flt,
        return_metadata=["score", "distance"],   # valid metadata keys
        return_properties=props,
    )
    return resp.objects, {"alpha": alpha, "hard_filters": hf, "query_text": query_text}


# --------------------------
# 4) Small NL summary for GPT
# --------------------------

def summarize_user_hand_nl(
    cfg, query_doc: Dict[str, Any]
) -> str:
    f = query_doc["facets"]
    t = query_doc["tokens"]
    parts = [
        f"Stakes {cfg.stakes_sb}/{cfg.stakes_bb}, {cfg.table_size}-max.",
        f"Hero {cfg.hero_position}, {cfg.hero_hand}.",
        f"Preflop: {t['preflop']}."
    ]
    if "flop" in t:
        parts.append(f"Flop: {t['flop']} (board texture={f.get('texture_class')}).")
    if "turn" in t:
        parts.append(f"Turn: {t['turn']}.")
    if "river" in t:
        parts.append(f"River: {t['river']} (runout={f.get('runout_texture_class')}).")
    return " ".join(parts)


# --------------------------
# Internals
# --------------------------

def _safe_schema_tokens(cfg) -> str:
    try:
        return cfg.schema_tokens()
    except Exception:
        pos = getattr(cfg, "hero_position", "?")
        hand = getattr(cfg, "hero_hand", "??")
        return f"pos={pos} hero={hand}"

def _participants_to_flop(pre: PreflopRecord) -> List[str]:
    final = {pre.opener: "raise"}
    for a in pre.trail_actions:
        final[a.position] = a.action
    involved = [p for p, act in final.items() if act in ("call", "raise")]
    return [p for p in POS_9MAX if p in involved]

def _analyze_runout(flop_board: Optional[str], turn_card: Optional[str], river_card: Optional[str]) -> Dict[str, Any]:
    def split2(s: str) -> List[str]:
        return [s[i:i+2] for i in range(0, len(s), 2)] if s else []

    flop_cards = split2(flop_board)
    all_cards = flop_cards + ([turn_card] if turn_card else []) + ([river_card] if river_card else [])

    # flop-only
    flop_r = [c[0] for c in flop_cards]
    flop_s = [c[1] for c in flop_cards]
    f_suit_ct = {s: flop_s.count(s) for s in set(flop_s)} or {"x": 1}
    f_flush = max(f_suit_ct.values())
    f_rank_ct = {r: flop_r.count(r) for r in set(flop_r)} or {}
    f_pair_raw = 0 if not f_rank_ct else max(f_rank_ct.values())
    f_pair = 0 if f_pair_raw == 1 else f_pair_raw
    f_vals = sorted({_RANK_VAL[r] for r in flop_r}) if flop_r else []
    f_straight = _coarse_straightness(f_vals)
    f_hi = max(f_vals) if f_vals else _RANK_VAL["T"]
    f_hi_bucket = "H" if f_hi >= _RANK_VAL["Q"] else ("M" if f_hi >= _RANK_VAL["9"] else "L")
    if f_flush == 3:
        f_tex = "monotone"
    elif f_pair >= 2:
        f_tex = "paired"
    elif f_straight >= 1:
        f_tex = "draw-heavy"
    else:
        f_tex = "dry"

    # runout
    rks = [c[0] for c in all_cards]
    sts = [c[1] for c in all_cards]
    suit_ct = {s: sts.count(s) for s in set(sts)} or {"x": 1}
    runout_flush_max = max(suit_ct.values())
    r_rank_ct = {r: rks.count(r) for r in set(rks)} or {}
    raw = 0 if not r_rank_ct else max(r_rank_ct.values())
    if raw <= 1: paired_by_river = 0
    elif raw == 2: paired_by_river = 2
    elif raw == 3: paired_by_river = 3
    else: paired_by_river = 4
    v = sorted({_RANK_VAL[r] for r in rks}) if rks else []
    straightness_runout = _coarse_straightness(v, allow_len=len(all_cards))
    if runout_flush_max >= 5:
        runout_tex = "monotone"
    elif paired_by_river >= 2:
        runout_tex = "paired"
    elif straightness_runout >= 1:
        runout_tex = "draw-heavy"
    else:
        runout_tex = "dry"

    return {
        "flop_flush_level": f_flush,
        "flop_paired_level": 0 if f_pair == 0 else (1 if f_pair == 2 else 3),
        "flop_straightness": f_straight,
        "flop_high_card_bucket": f_hi_bucket,
        "flop_texture_class": f_tex,
        "runout_flush_max": runout_flush_max,
        "flush_by_river": runout_flush_max >= 5,
        "paired_by_river": paired_by_river,
        "straightness_runout": straightness_runout,
        "runout_texture_class": runout_tex,
    }

def _coarse_straightness(vals: List[int], allow_len: int = 3) -> int:
    if not vals:
        return 0
    uniq = sorted(set(vals))
    uniq_wheel = sorted({1 if v == _RANK_VAL["A"] else v for v in uniq})

    def score(arr: List[int], need: int) -> int:
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
    return max(score(uniq, need), score(uniq_wheel, need))

def _to_dict(resp: Any) -> Dict[str, Any]:
    # already a dict
    if isinstance(resp, dict):
        return resp

    # Pydantic model -> dict
    if BaseModel is not None and isinstance(resp, BaseModel):
        return resp.model_dump()

    # LangChain AIMessage -> try parsed metadata, else JSON content
    if AIMessage is not None and isinstance(resp, AIMessage):
        # some LC providers stash parsed output here
        meta = getattr(resp, "response_metadata", {}) or {}
        if "parsed" in meta and isinstance(meta["parsed"], dict):
            return meta["parsed"]

        # some stash inside additional_kwargs
        ak = getattr(resp, "additional_kwargs", {}) or {}
        if "parsed" in ak and isinstance(ak["parsed"], dict):
            return ak["parsed"]

        # otherwise parse content
        content = getattr(resp, "content", "")
        if isinstance(content, str):
            return json.loads(content)

        # sometimes content is a list of parts with "text"
        if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
            return json.loads(content[0]["text"])

    # last-ditch: if it's a JSON string, parse it
    if isinstance(resp, str):
        return json.loads(resp)

    raise TypeError(f"Unsupported response type for structured output: {type(resp)}")