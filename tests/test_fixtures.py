from pathlib import Path
import json
from typing import Any, Dict, List, Optional

def _actions_to_dicts(actions) -> List[Dict[str, Any]]:
    """Serialize a list of StreetAction/Preflop actions (position, action, amount)."""
    out = []
    for a in actions or []:
        out.append({
            "position": getattr(a, "position", None),
            "action": getattr(a, "action", None),
            "amount": getattr(a, "amount", None),
        })
    return out

def serialize_hand_fixture(
    cfg, pre,
    flop_result=None,
    turn_result=None,
    river_result=None,
    nl_summary: Optional[str] = None,
    raw_query: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pack everything you need to replay a test without re-entering input."""
    fixture = {
        "cfg": {
            "stakes_sb": cfg.stakes_sb,
            "stakes_bb": cfg.stakes_bb,
            "table_size": cfg.table_size,
            "hero_position": cfg.hero_position,
            "hero_hand": cfg.hero_hand,
            "effective_stack": cfg.effective_stack,
            "straddle": cfg.straddle,
            "ante": cfg.ante,
        },
        "pre": {
            "opener": pre.opener,
            "open_amount": getattr(pre, "open_amount", None),
            "trail_actions": _actions_to_dicts(getattr(pre, "trail_actions", [])),
            "preflop_tokens": pre.preflop_tokens,
            "players_to_flop": pre.players_to_flop,
            "pot_type": pre.pot_type,
            "raises": getattr(pre, "raises", None),
        },
        "flop": None,
        "turn": None,
        "river": None,
        "nl_summary": nl_summary,
        "raw_query": raw_query,
    }

    if flop_result:
        fr = flop_result.record
        fixture["flop"] = {
            "board": fr.board,
            "actions": _actions_to_dicts(fr.actions),
            "tokens": fr.tokens,
            "remaining_positions": flop_result.remaining_positions,
        }
    if turn_result:
        tr = turn_result.record
        fixture["turn"] = {
            "card": tr.board,
            "actions": _actions_to_dicts(tr.actions),
            "tokens": tr.tokens,
            "remaining_positions": turn_result.remaining_positions,
        }
    if river_result:
        rr = river_result.record
        fixture["river"] = {
            "card": rr.board,
            "actions": _actions_to_dicts(rr.actions),
            "tokens": rr.tokens,
            "remaining_positions": river_result.remaining_positions,
            "showdown_raw": getattr(river_result, "showdown_raw", None),
        }

    return fixture

def save_fixture(fixture: Dict[str, Any], filename: str = "last_hand_fixture.json") -> Path:
    """Write the fixture to data/test_fixtures/<filename> and return the path."""
    out_dir = Path.cwd() / "data" / "test_fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"\n[Test Fixture] Saved to {out_path}")
    return out_path
