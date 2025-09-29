from __future__ import annotations

import json
from typing import List, Dict, Any, Optional


from dotenv import load_dotenv
from pathlib import Path
import sys
import os
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_openai import ChatOpenAI

# ---------------------------
# System prompt for coaching
# ---------------------------
SYSTEM_PROMPT = (
    """
    You are a professional poker coach with 10+ years of live cash experience. Your job is to turn a single hand history into concise, actionable, street-by-street coaching a player can use to fix leaks and to improve their poker gameplay.
    Coaching stance:
    Prefer exploitative adjustments when pool reads suggest an edge (over-calling, under-bluffing, sizing tells).
    Otherwise anchor to a GTO-consistent baseline and explain why.
    Avoid results-oriented bias: evaluate decisions with the info available at the time.
    Cards & data:
    You may and should reference exact hole cards and board cards (ranks/suits) where relevant.
    Be numerically consistent with stacks, SPR, and pot sizes; if math matters, show the one-line calculation (e.g., “need 28% equity to call”).
    Spot definition & similarity (for retrieved coaching):
    A “spot” = formation (pot type, HU/MW, IP/OOP) + street + board state (texture class + notable runout event) + SPR bucket + line semantics (c-bet / probe / check-raise / barrel / overbet / jam).
    SIMILAR (use as primary evidence) when all hold:
    Street matches (river > turn > flop).
    Formation compatible (same pot type or same IP/OOP; HU vs MW should match unless advice is clearly formation-agnostic).
    Board state compatible (same/adjacent texture; for turn/river the same runout event such as turn completes flush or river pairs board).
    SPR close (same bucket or adjacent: SPR<1, ~2, ~3–5, >5).
    Line overlap (≥1 shared action token: c-bet, check-raise, probe, barrel, overbet, jam).
    If data is sparse, keep (1)+(2) and require at least one of {(3),(4),(5)}.


    NOT SIMILAR (discard) if any: street mismatch that the tip hinges on; formation conflict that changes incentives (e.g., limped vs 4-bet, HU vs 4-way) and advice is formation-sensitive; board/runout contradiction (e.g., monotone-specific guidance on rainbow/paired dry); SPR far off so commitment thresholds flip; or line/node mismatch (e.g., tip about facing a check-raise applied to bet-bet-jam).
    Tie-breakers (borderline cases): prioritize alignment on line semantics and runout event, then formation, then SPR, then texture; use coarse hand class (pair/draw/air/monster) only as a final tiebreaker.
    Evidence policy (precedence)
    You MUST integrate retrieved coaching when it is similar enough to the spot and let it guide your advice.
    Merge the 2–4 strongest ideas; if conflicts arise, prefer the retrieved guidance unless it is clearly inapplicable due to explicit hand details.
    Output style
    Be crisp and imperative; no narration of every action. Reference the canonical line_compact once if needed.
    Use sizing bands only when helpful (e.g., 25–33%, 60–75%, 110–150%).
    When multiple viable lines exist, present a default line + a brief exploitative deviation and when to use it.`
    Vocabulary (use consistently)
    SPR, range advantage, nut advantage, capped range, polarity, blockers, bluff-catcher, thin value, deny equity, realization, MDF, pot odds.
    Deliverable
    Produce a street-by-street breakdown with short bullets for what to do and why, optional sizes, and citations only if evidence was used. End with 1–2 quick takeaways that summarize the most important adjustments. 
    """
)

# ---------------------------
# Public API
# ---------------------------
def generate_coaching_advice(
    nl_summary: str,
    evidence_context_text: str,
    evidence_json_list: List[Dict[str, Any]],
    *,
    model: str = "gpt-5-mini",
    max_tokens: Optional[int] = None,
) -> str:
    """
    Produce the final coaching response using:
      - nl_summary: your hand summary from build_nl_summary(...)
      - evidence_context_text: human-readable context from build_evidence_context(objects, k=3)
      - evidence_json_list: the parallel machine-readable evidence list (same function returns this)
    Returns plain text suitable for console output.
    """
    llm = ChatOpenAI(model=model, max_tokens=max_tokens)

    user_payload = {
        "HAND_NL_SUMMARY": nl_summary,                  
        "EVIDENCE_TEXT": evidence_context_text,         
        "EVIDENCE_JSON": evidence_json_list,          
        "INSTRUCTIONS": {
            "goal": "Provide concise, actionable coaching per street; integrate strongest relevant ideas from EVIDENCE.",
            "must_use_evidence": True,
        },
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    resp = llm.invoke(messages)
    # langchain_openai returns an AIMessage with .content
    return getattr(resp, "content", str(resp))

def evidence_card_from_obj(o) -> dict:
    """
    Returns:
      {
        "text_block": "...semi-structured evidence card...",
        "json": { minimal, high-signal json for traceability }
      }
    """
    p = o.properties

    title = p.get("title") or "<untitled>"
    schema = (p.get("schema_string") or "").strip()
    line = (p.get("line_compact") or "").strip()

    # Prefer explicit texture_class; fallback to board_texture
    texture = p.get("texture_class") or p.get("board_texture")

    # Merge tags + improvement_flags and normalize
    tags = _join_tokens((p.get("tags") or []) + (p.get("improvement_flags") or []))

    # Coaching fields (support old/new)
    coaching_summary, coaching_actions = _coaching_fields(p)

    # short facets dict (only anchors)
    facets = _short_facets({
        "street_focus": p.get("street_focus"),
        "pot_type": p.get("pot_type"),
        "texture_class": texture,
        "spr_bucket": p.get("spr_bucket"),
        "hero_position": p.get("hero_position"),
        "villain_position": p.get("villain_position"),
        "heads_up": p.get("heads_up"),
        "players_to_flop": p.get("players_to_flop"),
    })

    block_lines = []
    block_lines.append(f"[EVIDENCE]")
    block_lines.append(f"title={title}")
    if schema:
        block_lines.append(f"schema={schema}")
    if line:
        block_lines.append(f"line={line}")
    block_lines.append("facets={" + ", ".join(f"{k}: {v}" for k, v in facets.items() if v not in (None, "")) + "}")
    if texture:
        block_lines.append(f"texture={str(texture).lower()}")
    if tags:
        block_lines.append(f"tags={tags}")
    if coaching_summary:
        block_lines.append(f"coaching_summary={coaching_summary}")
    if coaching_actions:
        block_lines.append("action_points=" + "; ".join(coaching_actions[:3]))  # keep tight

    text_block = "\n".join(block_lines)

    json_payload = {
        "title": title,
        "facets": facets,
        "schema_string": schema,
        "line_compact": line,
        "texture_class": texture,
        "tags": [t.strip() for t in tags.split(", ")] if tags else [],
        "coaching_summary": coaching_summary,
        "coaching_actions": coaching_actions[:5],
        "score": getattr(o.metadata, "score", None),
        "uuid": str(o.uuid),
    }

    return {"text_block": text_block, "json": json_payload}

def build_evidence_context(objects, k) -> tuple[str, list[dict]]:
    """
    Compose top-K evidence blocks for the LLM.
    Returns (context_str, json_list)
    """
    cards = [evidence_card_from_obj(o) for o in objects[:k]]
    context = "\n\n".join(c["text_block"] for c in cards)
    return context, [c["json"] for c in cards]

# -------------------------------
# Evidence block builders
# -------------------------------

def _as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]

def _join_tokens(tokens):
    toks = [t for t in _as_list(tokens) if t]
    # normalize to lowercase and dedupe in stable order
    seen, out = set(), []
    for t in toks:
        t2 = str(t).strip().lower()
        if t2 and t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return ", ".join(out)

def _short_facets(d):
    # Only the highest-signal anchors you want the model to key on
    return {
        "street_focus": d.get("street_focus"),
        "pot_type": d.get("pot_type"),
        "texture_class": d.get("texture_class") or d.get("board_texture"),
        "spr_bucket": d.get("spr_bucket"),
        "hero_position": d.get("hero_position"),
        "villain_position": d.get("villain_position"),
        "heads_up": d.get("heads_up"),
        "players_to_flop": d.get("players_to_flop"),
    }

def _coaching_fields(p):
    """
    Support both old and new schema names:
      - annotated_coaching_description  (old)
      - coaching_summary                (new)
      - action_points                   (old)
      - coaching_actions                (new)
    """
    summary = p.get("coaching_summary") or p.get("annotated_coaching_description")
    actions = p.get("coaching_actions") or p.get("action_points") or []
    return summary, [a for a in _as_list(actions) if a]


if __name__ == "__main__":

    import sys
    from pathlib import Path

    nl_path = Path("/tmp/nl_summary.txt")
    ev_path = Path("/tmp/evidence.json")

    if not nl_path.exists() or not ev_path.exists():
        print("Usage (example): write nl_summary to /tmp/nl_summary.txt and evidence to /tmp/evidence.json, then run:")
        print("  python llm_coach.py")
        sys.exit(0)

    nl_summary = nl_path.read_text()
    ev = json.loads(ev_path.read_text())
    evidence_context = ev.get("context") or ""
    evidence_json = ev.get("cards") or []

    out = generate_coaching_advice(nl_summary, evidence_context, evidence_json)
    print("\n=== Coaching Advice ===\n")
    print(out)
