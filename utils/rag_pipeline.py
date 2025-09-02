# utils/rag_pipeline.py
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Callable
import json
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from weaviate.classes.query import MetadataQuery

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
    cfg,
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

    runout = _analyze_runout(flop_board, turn_card, river_card)

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
    schema_string = _safe_schema_tokens(cfg)

    pieces = [
        _safe_schema_tokens(cfg), #schema tokens may already include the betting line
        f"Preflop: {tokens['preflop']}."
    ]
    print("1234: ", _safe_schema_tokens(cfg))
    if 'flop' in tokens:
        pieces.append(f"Flop: {tokens['flop']}. "
                      f"Flop is {facets.get('texture_class','unknown')} "
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
    print("5678: ", embedding_text)

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


def enrich_user_hand_with_llm(llm, raw_doc: dict, cfg) -> dict:
    """
    Produce a RAG-ready JSON by calling an LLM with a compact system prompt
    + structured output (JSON Schema with enums). No post normalizer/validator here.
    """

    # -----------------------------
    # 0) Build inputs for the call
    # -----------------------------

    facets_in = raw_doc.get("facets", {}) or {}
    derived_fields = {
        "facets": {
            "positions":      facets_in.get("positions"),
            "line_compact":   facets_in.get("line_compact"),
            "pot_type":       facets_in.get("pot_type"),
            "heads_up":       facets_in.get("heads_up"),
            "spr_bucket":     facets_in.get("spr_bucket"),
            "stack_depth":    facets_in.get("stack_depth"),
            "texture_class":  facets_in.get("texture_class"),
            "flush_level":    facets_in.get("flush_level"),
            "paired_level":   facets_in.get("paired_level"),
            "straightness":   facets_in.get("straightness"),
            "high_card_bucket": facets_in.get("high_card_bucket"),
            "hero_position":  facets_in.get("hero_position"),
        }
    }

    private_hints = {
        "hero_hand":  getattr(cfg, "hero_hand", None),
        "flop_board": facets_in.get("flop_board"),
        "turn_card":  facets_in.get("turn_card"),
        "river_card": facets_in.get("river_card"),
    }

    raw_min = {
        "schema_string":  raw_doc.get("schema_string", ""),
        "embedding_text": raw_doc.get("embedding_text", ""),
        "facets":         facets_in,
        "tokens":         raw_doc.get("tokens", {}),
        "title":          raw_doc.get("title", "User-entered hand"),
    }

    # ---------------------------------
    # 1) Compact system rules (prompt)
    # ---------------------------------
    system = (
        "You are a poker strategy enricher. Output JSON only that conforms to the provided JSON Schema (PokerRAGEntry).\n"
        "INPUTS in user message:\n"
        "- DERIVED_FIELDS: objective truths computed upstream; copy verbatim and do NOT change.\n"
        "- RAW_JSON: canonical user hand context.\n"
        "- PRIVATE_HINTS: exact hole cards and board (for inference only; never reveal).\n\n"
        "TASK: Produce schema_string, facets, hard_filters vs soft_signals, tags, and embedding_text.\n\n"
        "GLOBAL RULES:\n"
        "- JSON only; no prose outside fields. If unknown, set null.\n"
        "- Never reveal exact cards/suits. Use concepts only.\n"
        "- Use PRIVATE_HINTS only to infer hero_hand_class and blocker_pattern.\n"
        "- Choose one street_focus (river > turn > flop unless the input says otherwise).\n"
        "- Don't duplicate the betting line: keep canonical line in facets.line_compact; embedding_text may include it once.\n"
        "- Keep values within schema enums; do not invent new tokens.\n\n"
        "HAND CLASS (on chosen street) priority:\n"
        "quads/full house/flush/straight > set > two-pair > overpair/underpair > top/second/weak-pair > draw > air.\n"
        "Definitions (strict): overpair = pocket pair higher than all board ranks; underpair = lower than the highest board rank and not on board; "
        "top/second/weak-pair = hole equals 1st/2nd/3rd+ highest distinct board rank; set = pocket pair matches a single board rank. "
        "If unsure between over/underpair, prefer underpair unless the overpair condition is strictly met.\n\n"
        "CONSTRUCTION:\n"
        "- schema_string: short, normalized tokens (pot/formation/pos/SPR/hero/board). Avoid per-street prose.\n"
        "- embedding_text: schema_string + one 'line=' from facets.line_compact + brief board features; no per-street narratives.\n"
        "- hard_filters: minimally pot_type and street_focus when known.\n"
        "- soft_signals: select 3–5 high-signal descriptors from the allowed improvement_flags list below; dedupe; also append to tags.\n\n"
        "ALLOWED improvement_flags (use these exact tokens or come up with your own flags (but don't deviate too far from the original intent)):\n"
        "A) dynamic_board, static_board, wet_board, dry_board, coordinated_high_connect, low_connect, monotone_pressure, two_tone_pressure, paired_board, paired_plus\n"
        "B) turn_completes_flush, turn_completes_straight, turn_pairs_board, straightening_turn, river_pairs_top, river_pairs_low, four_flush_river, backdoor_flush_river, backdoor_straight_river, scare_card_river, brick_river\n"
        "C) range_advantage_hero, nut_advantage_hero, range_advantage_villain, nut_advantage_villain, hero_IP, hero_OOP, multiway_pressure, blind_vs_blind\n"
        "D) turned_top_pair_value, underpair_showdown_bound, overpair_marginal_board, bluff_catcher_river, flush_blocker_pressure, straight_blocker_pressure, broadway_blocker_pressure\n"
        "E) small_cbet_range_node, polar_turn_barrel, delayed_cbet_viable, check_raise_dense, donk_represented, overbet_polar_node, thin_value_ok, cap_attack_spot, induce_vs_polar_line, minraise_strength_tell\n"
        "F) pool_low_bluff, pool_calls_wide, strong_player_present, weak_player_target, sticky_pool, nit_villain\n"
        "G) price_to_continue_good, fold_equity_low, deny_equity_priority, realization_penalty_OOP\n\n"
        "VALIDATION:\n"
        "- Do not alter DERIVED_FIELDS. Keep enums within schema vocab. If uncertain, set null."
    )

    # ------------------------------------
    # 2) JSON Schema with key enums (SO)
    # ------------------------------------
    POT_TYPES = ["limped", "single-raised", "3-bet", "4-bet", "5-bet", "6-bet", "7-bet", ">8-bet", None]
    POSITIONS = ["UTG", "EP", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB", "BvB", "straddle", None]
    TEXTURES  = ["dry", "paired", "monotone", "two-tone", "draw-heavy", "high-connect", "low-connect", "coordinated", "rainbow", None]
    SPR_BUCKETS  = ["SPR<1", "SPR~2", "SPR~3-5", "SPR>5", None]
    STACK_DEPTHS = ["short", "medium", "deep", None]
    HIGH_BUCKETS = ["H", "M", "L", None]
    STREET_FOCUS = ["flop", "turn", "river", "all", None]
    HERO_HAND_CLASSES = [
        "air","draw","weak-pair","second-pair","top-pair","overpair","underpair",
        "two-pair","set","straight","flush","full house","quads","straight-flush","bluff-catcher", None
    ]

    # exact improvement_flags list (verbatim)
    IMPROVEMENT_FLAGS = [
        # A) Board/texture nuance
        "dynamic_board","static_board","wet_board","dry_board","coordinated_high_connect","low_connect",
        "monotone_pressure","two_tone_pressure","paired_board","paired_plus",
        # B) Runout events
        "turn_completes_flush","turn_completes_straight","turn_pairs_board","straightening_turn",
        "river_pairs_top","river_pairs_low","four_flush_river","backdoor_flush_river",
        "backdoor_straight_river","scare_card_river","brick_river",
        # C) Advantage / formation
        "range_advantage_hero","nut_advantage_hero","range_advantage_villain","nut_advantage_villain",
        "hero_IP","hero_OOP","multiway_pressure","blind_vs_blind",
        # D) Hand-state & blockers
        "turned_top_pair_value","underpair_showdown_bound","overpair_marginal_board","bluff_catcher_river",
        "flush_blocker_pressure","straight_blocker_pressure","broadway_blocker_pressure",
        # E) Line & sizing semantics
        "small_cbet_range_node","polar_turn_barrel","delayed_cbet_viable","check_raise_dense",
        "donk_represented","overbet_polar_node","thin_value_ok","cap_attack_spot",
        "induce_vs_polar_line","minraise_strength_tell",
        # F) Pool/opp tendencies
        "pool_low_bluff","pool_calls_wide","strong_player_present","weak_player_target","sticky_pool","nit_villain",
        # G) Pricing / equity realization
        "price_to_continue_good","fold_equity_low","deny_equity_priority","realization_penalty_OOP"
    ]

    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PokerRAGEntry",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "schema_string": {"type": "string"},
                    "embedding_text": {"type": "string"},
                    "facets": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "pot_type":        {"type": ["string", "null"], "enum": POT_TYPES},
                            "hero_position":   {"type": ["string", "null"], "enum": POSITIONS},
                            "villain_position":{"type": ["string", "null"], "enum": POSITIONS},
                            "heads_up":        {"type": ["boolean", "null"]},
                            "players_to_flop": {"type": ["integer", "null"]},
                            "street_focus":    {"type": ["string", "null"], "enum": STREET_FOCUS},
                            "board_texture":   {"type": ["string", "null"], "enum": TEXTURES},
                            "flush_level":     {"type": ["integer", "null"]},
                            "paired_level":    {"type": ["integer", "null"]},
                            "straightness":    {"type": ["integer", "null"]},
                            "high_card_bucket":{"type": ["string", "null"], "enum": HIGH_BUCKETS},
                            "texture_class":   {"type": ["string", "null"], "enum": TEXTURES},
                            "hero_hand_class": {"type": ["string", "null"], "enum": HERO_HAND_CLASSES},
                            "blocker_pattern": {"type": ["string", "null"]},
                            "spr_bucket":      {"type": ["string", "null"], "enum": SPR_BUCKETS},
                            "stack_depth":     {"type": ["string", "null"], "enum": STACK_DEPTHS},
                            "line_compact":    {"type": ["string", "null"]},
                            "positions":       {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["positions", "line_compact"]
                    },
                    "hard_filters": {"type": "object"},
                    "soft_signals": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "texture_class":  {"type": ["string","null"], "enum": TEXTURES},
                            "hero_hand_class":{"type": ["string","null"], "enum": HERO_HAND_CLASSES},
                            "spr_bucket":     {"type": ["string","null"], "enum": SPR_BUCKETS},
                            "line_tokens":    {"type": "array", "items": {"type": "string"}},
                            "heads_up":       {"type": ["boolean","null"]},
                            # exact allowed improvement flags:
                            "improvement_flags": {"type": "array", "items": {"type": "string", "enum": IMPROVEMENT_FLAGS}}
                        }
                    },
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "title","schema_string","embedding_text","facets",
                    "hard_filters","soft_signals","tags"
                ]
            }
        }
    }

    # -----------------------------
    # 3) Build user message payload
    # -----------------------------
    user_payload = {
        "DERIVED_FIELDS": derived_fields,
        "RAW_JSON": raw_min,
        "PRIVATE_HINTS": private_hints,
    }

    # -----------------------------
    # 4) Invoke LLM (structured out)
    # -----------------------------
    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        response_format=response_schema,
    )

    # -----------------------------
    # 5) Parse & light backfill only
    # -----------------------------
    content = getattr(resp, "content", resp)
    try:
        enriched = json.loads(content) if isinstance(content, str) else (content if isinstance(content, dict) else {})
    except Exception:
        enriched = {}

    # Ensure top-level keys exist
    enriched.setdefault("title", raw_doc.get("title", "User-entered hand"))
    enriched.setdefault("schema_string", raw_doc.get("schema_string", ""))
    enriched.setdefault("embedding_text", raw_doc.get("embedding_text", ""))
    enriched.setdefault("facets", {})
    enriched.setdefault("hard_filters", {})
    enriched.setdefault("soft_signals", {})
    #enriched.setdefault("rule_features", {})
    enriched.setdefault("tags", [])

    # Backfill obvious facet fields from raw if model left them null
    f = enriched["facets"] or {}
    for k in [
        "positions","line_compact","pot_type","stack_depth","texture_class","board_texture",
        "flush_level","paired_level","straightness","high_card_bucket","heads_up",
        "flop_board","turn_card","river_card","spr_bucket","hero_position","villain_position"
    ]:
        if f.get(k) in (None, "", [], {}):
            val = facets_in.get(k)
            if val is not None:
                f[k] = val
    enriched["facets"] = f

    # Minimal hard_filters fill (no nulls)
    hf = enriched["hard_filters"] or {}
    if f.get("pot_type"):
        hf["pot_type"] = f["pot_type"]
    if f.get("street_focus") in ("flop","turn","river"):
        hf["street_focus"] = f["street_focus"]
    enriched["hard_filters"] = hf

    return enriched



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

    vec = None
    if embed_fn is not None:
        vec = embed_fn(query_text)
        # Optional sanity check:
        if hasattr(vec, "__len__"):
            assert len(vec) == 384, f"Expected 384-D, got {len(vec)}"

    hf = _sanitize_hard_filters(enriched_query_doc.get("hard_filters"))
    flt = None
    for k, v in hf.items():
        f = Filter.by_property(k).equal(v)
        flt = f if flt is None else (flt & f)

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

def build_nl_summary(cfg, pre, flop_rec=None, turn_rec=None, river_rec=None) -> str:
    """One-paragraph, human-readable summary for the final coaching LLM."""
    parts = [
        f"Stakes ${_trim_float(cfg.stakes_sb)}/${_trim_float(cfg.stakes_bb)}, {cfg.table_size}-max.",
        f"Hero {cfg.hero_position} with {cfg.hero_hand}.",
        f"Preflop: {_pretty_to_human_line(pre.pretty())}."
    ]

    if flop_rec:
        flop_board = _fmt_board(getattr(flop_rec, "board", None))
        parts.append(f"Flop {flop_board}: {_pretty_to_human_line(flop_rec.pretty())}.")

    if turn_rec:
        turn_card = getattr(turn_rec, "board", None)
        parts.append(f"Turn {turn_card}: {_pretty_to_human_line(turn_rec.pretty())}.")

    if river_rec:
        river_card = getattr(river_rec, "board", None)
        parts.append(f"River {river_card}: {_pretty_to_human_line(river_rec.pretty())}.")

    return " ".join(parts)


# --------------------------
# Internals
# --------------------------

def _sanitize_hard_filters(hard_filters: dict | None) -> dict:
    if not hard_filters:
        return {}
    return {k: v for k, v in hard_filters.items() if v not in (None, "", [], {})}


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
    if isinstance(resp, dict):
        return resp

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

def _pretty_to_human_line(pretty_text: str) -> str:
    # Drop the first header line like "Turn: Tc" or "River: 5h"
    lines = [ln.strip() for ln in pretty_text.splitlines() if ln.strip()]
    body = lines[1:] if len(lines) > 1 else lines
    return ", ".join(body).replace("  ", " ")

def _trim_float(x: float) -> str:
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")

def _fmt_board(board: str | None) -> str:
    if not board:
        return "?"
    return " ".join(board[i:i+2] for i in range(0, len(board), 2))