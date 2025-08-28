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

    # schema_string = (
    #     f"{_safe_schema_tokens(cfg)} "
    #     f"| preflop={tokens['preflop']}"
    #     + (f" | flop={tokens['flop']}" if 'flop' in tokens else "")
    #     + (f" | turn={tokens['turn']}" if 'turn' in tokens else "")
    #     + (f" | river={tokens['river']}" if 'river' in tokens else "")
    # )
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

ALLOWED_TOP_KEYS = {
    "title", "schema_string", "embedding_text",
    "facets", "hard_filters", "soft_signals",
    "rule_features", "tags"
}

def enrich_user_hand_with_llm(llm, raw_doc: dict, cfg) -> dict:
    """
    Produce a RAG-ready JSON (exact shape from your prompt).
    - Uses LLM to normalize & infer missing facets.
    - Merges in deterministic features from raw_doc.
    - Returns ONLY the keys defined by the prompt (no extras).
    """

    #REMOVED: Do not mention specific hole cards or suits when describing the hand unless the inclusion of specific cards will help convey the concept more evidently. Focus on the spot, not the result.
    # ----- 1) Your exact instructions as system prompt -----
    system = (
        """
        You are a Poker strategy assistant that converts short, summarized hand review segments from a hand into a RAG‑ready JSON. Output must be valid JSON only. Your job is to:
        Produce a compact, normalized schema string that captures the strategic spot.
        Emit facets for filtering and reranking, using the controlled vocabularies below.
        Generate a clean embedding_text that combines the schema string and the annotated coaching description.
        Add rule_features for a lightweight, poker‑aware reranker.
        Separate hard_filters from soft_signals so the retriever can do graduated backoff.
        CRITICAL (internal reasoning only): You are given the hero’s exact hand and the board cards in `private_hints`.
        Use them solely to infer `hero_hand_class` and `blocker_pattern`. Think step-by-step **internally**; do NOT reveal your reasoning or the exact cards. Output VALID JSON only.

        Street focus
        - Classify on ONE street: river if present, else turn, else flop, unless a specific street is explicitly called out.

        Internal decision cascade (stop at the first match on the chosen street)
        Assemble the best 5-card hand from hero’s 2 cards + the board.
        1) Straight-flush (royal flush is a special case of straight-flush) – five in sequence, all same suit. Label as `"straight-flush"`.
        2) Quads – four of a kind.
        3) Full house – any three-of-a-kind + any pair.
        4) Flush – five to the same suit.
        5) Straight – five in sequence; allow A-5 wheel.
        6) Set – pocket pair matching exactly one board rank (not trips from board alone).
        7) Two-pair – two distinct ranks paired (hole+board or board+board+hole).
        8) Overpair – pocket pair strictly higher than **all** board ranks.
        9) Underpair – pocket pair lower than the **highest** board rank and not present on the board.
        10) Top-pair – one hole equals the highest distinct board rank.
        11) Second-pair – one hole equals the second-highest distinct board rank.
        12) Weak-pair – one hole equals a lower distinct board rank (3rd+).
        13) Draw – ONLY if none above applies (e.g., 4-flush, OESD, gutshot)(MUST BE BEFORE THE RIVER).
        14) Air – none of the above.

        Disambiguation / fail-safes
        - Never assign **overpair** unless it strictly meets the definition. If unsure between overpair and under/second/weak-pair, choose the latter.
        - If both a made hand and a draw exist, choose the higher category from the cascade.
        - `"bluff-catcher"` is for river-only, marginal one-pair spots in clearly polar situations; otherwise use the structural class.

        Blockers
        - Monotone or 4-flush runouts + hero holds the Ace of that suit → `blocker_pattern="flush_blocker"`.
        - Highly connected/broadway boards with hero removing common straights → `blocker_pattern="broadway_blocker"`.
        - Hero blocks the nut straight with key ranks → `blocker_pattern="straight_blocker"`.
        - Else, null.

        Privacy
        - Do not echo ranks/suits in the output—only derived labels (e.g., `"overpair"`, `"second-pair"`, `"straight-flush"`, `"flush_blocker"`).

        Vocabulary alignment
        - `hero_hand_class` must be one of:
        ["air","draw","weak-pair","second-pair","top-pair","overpair","underpair",
        "two-pair","set","straight","flush","full house","quads","straight-flush","bluff-catcher"].

        Controlled vocabularies
        pot_type: single-raised, 3-bet, 4-bet, limped, multiway, heads-up,straddle
        positions (primary hero pos; villain optional): UTG, EP, MP, LJ, HJ, CO, BTN, SB, BB, straddle, BvB
        street_events tokens (use any that apply, order by time):
        open, call, 3bet, 4bet, c-bet, delayed-cbet, donk, check-raise, float, probe, barrel, overbet, jam, min-raise, x, bet-small, bet-mid, bet-big
        board_texture: dry, two-tone, monotone, paired, paired+, draw-heavy, high-connect, low-connect, coordinated, rainbow
        hero_hand_class: overpair, underpair, top-pair, second-pair, weak-pair, set, two-pair, trips, straight, flush, full house, quads, combo draw, flush draw, straight draw, gutshot, overcards, air, bluff-catcher
        stack_dynamics: short, medium, deep, plus SPR<1, SPR~2, SPR~3-5, SPR>5
        strategic_themes: thin value, exploit, GTO deviation, capped range attack, missed value, bluff catcher, indifferent, sizing tell, induce, deny equity, range advantage, story inconsistency, leveling,nit-villain, loose-villain,strong player, weak player, etc.
        Derived board features
        Compute coarse, integer features from the description. If unknown, infer from context or set null.
        flush_level: 0–4 where 0 none, 3 three to a suit, 4 four to a suit or completed flush environment.
        paired_level: 0 none, 1 one pair on board, 2 trips/paired+, 3 double-paired+.
        straightness: 0 low, 1 some connectivity, 2 high connectivity.
        high_card_bucket: H (A/K present), M (Q/J/T), L (<=9 focus).
        texture_class: one of board_texture above that best summarizes the runout.
        If the transcript indicates blockers, set blocker_pattern such as flush_blocker, straight_blocker, ace_blocker, broadway_blocker.
        Output shape
        Return one JSON object per video with exactly these keys:
        json
        Copy
        {
        "title": "string",
        "schema_string": "single line, normalized tokens (from inputted schema_string)",
        "embedding_text": "schema_string + hand recap (from inputted embedding_text)",
        "facets": {
            "pot_type": "single-raised|3-bet|4-bet|limped|multiway|heads-up",
            "hero_position": "UTG|EP|...|BB|BvB",
            "villain_position": "optional",
            "heads_up": true,
            "street_focus": "flop|turn|river|all",
            "board_texture": "see vocab",
            "flush_level": 0,
            "paired_level": 0,
            "straightness": 0,
            "high_card_bucket": "H|M|L",
            "texture_class": "see vocab",
            "hero_hand_class": "see vocab",
            "blocker_pattern": "string or null",
            "spr_bucket": "SPR<1|SPR~2|SPR~3-5|SPR>5",
            "stack_depth": "short|medium|deep",
            "line_compact": "tokenized street_events, e.g., 'open call | c-bet call | barrel jam'",
            "positions": ["hero", "villain", "others if relevant"]
        },
        "hard_filters": {
            "pot_type": "value",
            "street_focus": "value or null if not specific"
        },
        "soft_signals": {
            "texture_class": "value",
            "hero_hand_class": "value",
            "spr_bucket": "value",
            "line_tokens": ["c-bet","check-raise","barrel","jam"],
            "heads_up": true
        },
        "rule_features": {
            "board": { "texture_class": "value", "flush_level": 0, "paired_level": 0, "straightness": 0, "high_card_bucket": "H|M|L" },
            "hero": { "hand_class": "value", "blocker_pattern": "value or null" },
            "context": { "spr_bucket": "value", "position_primary": "hero_position", "line_compact": "same as facets.line_compact" }
        },
        "tags": ["comma style tags for UI and quick filters"],
        }

        Construction rules
        schema_string is short and machine‑friendly. Example pattern:
        pot=3-bet HU pos=BTN_vs_SB spr=~2 hero=overpair board=paired dry line=c-bet-call | barrel | river overbet
        embedding_text = schema_string + hand summary from the perspective of the hero (include hero's actions, villains actions and board textures)
        Prefer inference: if the board is monotone and the user contains the Ace of the suit on the board, set board_texture=monotone, flush_level>=3, blocker_pattern=flush_blocker, hero_hand_class=bluff-catcher if consistent. Do your best to infer these fields.
        If the street focus is clearly about one decision point, set street_focus to that street, else all.
        If info is missing, keep fields but set to "None" and avoid guessing wildly.
        """
    )


    facets = raw_doc.get("facets", {})
    fewshots = [
        {
            "input": {
                "schema_string": "pot=3-bet pos=BB spr=~3 board=draw-heavy",
                "embedding_text": "Preflop 3-bet, BB vs BTN. Flop is Q-high and connected. River checks through."
            },
            "private_hints": {
                "hero_hand": "JhJd", "flop_board": "Qs7c3h", "turn_card": "2d", "river_card": "9s"
            },
            "expect": {
                "facets.hero_hand_class": "underpair"
            }
        },
        {
            "input": {
                "schema_string": "pot=single-raised pos=HJ spr=>5 board=dry",
                "embedding_text": "Single-raised, HJ vs BB. Flop is T-high, dry."
            },
            "private_hints": {
                "hero_hand": "JdJc", "flop_board": "Td4s2c", "turn_card": "7h", "river_card": "Qh"
            },
            "expect": {
                "facets.hero_hand_class": "overpair"
            }
        },
        {
            "input": {"schema_string": "pot=single-raised pos=CO board=coordinated"},
            "private_hints": {"hero_hand": "Qs9s", "flop_board": "KhQd7c", "turn_card": "2h", "river_card": "2c"},
            "expect": {"facets.hero_hand_class": "second-pair"}
        },
        {
            "input": {"schema_string": "pot=single-raised pos=BTN board=monotone"},
            "private_hints": {"hero_hand": "AdKc", "flop_board": "5d9dQd", "turn_card": "2s", "river_card": "2c"},
            "expect": {"facets.blocker_pattern": "flush_blocker"}
        },
        {
            "input": {"schema_string": "pot=single-raised pos=BTN board=monotone"},
            "private_hints": {"hero_hand": "Ah4h", "flop_board": "QhJh9h", "turn_card": "8d", "river_card": "2c"},
            "expect": {"facets.blocker_pattern": "flush_blocker", "facets.hero_hand_class": "flush"}
        },
        {
            "input": {"schema_string": "pot=single-raised pos=BTN board=monotone"},
            "private_hints": {"hero_hand": "7h6d", "flop_board": "5h3d9s"},
            "expect": {"facets.blocker_pattern": "flush_blocker", "facets.hero_hand_class": "GS"}
        },
    ]

    user_payload = {
        "input": {
            "schema_string": raw_doc.get("schema_string", ""),
            "embedding_text": raw_doc.get("embedding_text", ""),
            "facets": facets,                    # includes texture metrics you already computed
            "tokens": raw_doc.get("tokens", {}), # preflop/flop/turn/river tokens if present
        },
        "private_hints": {
            "hero_hand": getattr(cfg, "hero_hand", None),         # e.g., "QhJh"
            "flop_board": facets.get("flop_board"),               # e.g., "9dQd2d"
            "turn_card":  facets.get("turn_card"),                # e.g., "Tc"
            "river_card": facets.get("river_card"),               # e.g., "9h"
        },
        "few_shots": fewshots,
    }
    # Constrain output with a JSON schema (enumerations keep it on-rails)
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PokerRAGEntry",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "schema_string": {"type": "string"},
                    "embedding_text": {"type": "string"},
                    "facets": {
                        "type": "object",
                        "properties": {
                            "pot_type": {"type": ["string", "null"]},
                            "hero_position": {"type": ["string", "null"]},
                            "villain_position": {"type": ["string", "null"]},
                            "heads_up": {"type": ["boolean", "null"]},
                            "street_focus": {"type": ["string", "null"], "enum": ["flop","turn","river","all", None]},
                            "board_texture": {"type": ["string", "null"]},
                            "flush_level": {"type": ["integer", "null"]},
                            "paired_level": {"type": ["integer", "null"]},
                            "straightness": {"type": ["integer", "null"]},
                            "high_card_bucket": {"type": ["string", "null"], "enum": ["H","M","L", None]},
                            "texture_class": {"type": ["string", "null"]},
                            "hero_hand_class": {
                                "type": ["string", "null"],
                                "enum": [
                                    "air","draw","weak-pair","second-pair","top-pair",
                                    "overpair","underpair","two-pair","set","straight",
                                    "flush","full house","quads","bluff-catcher", None
                                ]
                            },
                            "blocker_pattern": {"type": ["string", "null"]},
                            "spr_bucket": {"type": ["string", "null"]},
                            "stack_depth": {"type": ["string", "null"], "enum": ["short","medium","deep", None]},
                            "line_compact": {"type": ["string", "null"]},
                            "positions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["positions", "line_compact"]
                    },
                    "hard_filters": {"type": "object"},
                    "soft_signals": {"type": "object"},
                    "rule_features": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title","schema_string","embedding_text","facets","hard_filters","soft_signals","rule_features","tags"],
                "additionalProperties": False
            }
        }
    }

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        response_format=response_schema,
        #temperature=0,
    )

    # Parse & light normalization (no deterministic overrides)
    content = getattr(resp, "content", resp)
    try:
        enriched = json.loads(content) if isinstance(content, str) else json.loads(content or "{}")
    except Exception:
        enriched = {}

    enriched = {k: v for k, v in enriched.items() if k in ALLOWED_TOP_KEYS}
    enriched.setdefault("title", raw_doc.get("title", "User-entered hand"))
    enriched.setdefault("schema_string", raw_doc.get("schema_string", ""))
    enriched.setdefault("embedding_text", raw_doc.get("embedding_text", ""))
    enriched.setdefault("facets", {})
    enriched.setdefault("hard_filters", {})
    enriched.setdefault("soft_signals", {})
    enriched.setdefault("rule_features", {})
    enriched.setdefault("tags", [])

    # (Optional but harmless) backfill neutral fields from raw_doc if missing
    f = enriched["facets"]
    for k in ["positions","line_compact","pot_type","stack_depth","texture_class",
              "board_texture","flush_level","paired_level","straightness","high_card_bucket",
              "flop_board","turn_card","river_card"]:
        if f.get(k) in (None, "", [], {}):
            val = facets.get(k)
            if val is not None:
                f[k] = val

    # Hard filters: avoid nulls (this prevents your previous GRPC “unknown value type <nil>” error)
    hf = enriched["hard_filters"]
    if f.get("pot_type"):
        hf["pot_type"] = f["pot_type"]
    if f.get("street_focus") in ("flop","turn","river"):
        hf["street_focus"] = f["street_focus"]

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