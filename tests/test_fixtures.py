from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils.hand_input import HandConfig
from utils.preflop_input import PreflopRecord, PreflopAction
from utils.flop_input import FlopRecord, StreetAction
from utils.turn_input import TurnRecord
from utils.river_input import RiverRecord

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
            "open_to": pre.open_to,
            "trail_actions": _actions_to_dicts(getattr(pre, "trail_actions", [])),
            "preflop_tokens": pre.preflop_tokens,
            "players_to_flop": pre.players_to_flop,
            "pot_type": pre.pot_type,
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

@dataclass
class Result:
    record: Any
    remaining_positions: list[str]
    showdown_raw: Optional[str] = None

def load_fixture(path: str | Path) -> Tuple[HandConfig, PreflopRecord,
                                            Optional[Result], Optional[Result], Optional[Result],
                                            dict]:
    """
    Rebuild HandConfig, PreflopRecord, and street Results from a saved JSON fixture.
    Returns: (cfg, pre, flop_result, turn_result, river_result, full_fixture_dict)
    """
    p = Path(path)
    data = json.loads(p.read_text())

    # HandConfig
    cfg = HandConfig(
        stakes_sb=data["cfg"]["stakes_sb"],
        stakes_bb=data["cfg"]["stakes_bb"],
        table_size=data["cfg"]["table_size"],
        hero_position=data["cfg"]["hero_position"],
        hero_hand=data["cfg"]["hero_hand"],
        effective_stack=data["cfg"]["effective_stack"],
        straddle=data["cfg"].get("straddle"),
        ante=data["cfg"].get("ante"),
    )

    # Preflop
    pre_actions = [PreflopAction(**a) for a in data["pre"].get("trail_actions", [])]
    pre = PreflopRecord(
        opener=data["pre"]["opener"],
        open_to=data["pre"].get("open_to", data["pre"].get("open_amount")),
        trail_actions=pre_actions,
    )

    # Flop
    flop_result = None
    if data.get("flop"):
        fr = data["flop"]
        fr_actions = [StreetAction(**a) for a in fr.get("actions", [])]
        flop_rec = FlopRecord(board=fr["board"], actions=fr_actions)
        flop_result = Result(record=flop_rec, remaining_positions=fr.get("remaining_positions", []))

    # Turn
    turn_result = None
    if data.get("turn"):
        tr = data["turn"]
        tr_actions = [StreetAction(**a) for a in tr.get("actions", [])]
        turn_rec = TurnRecord(board=tr["card"], actions=tr_actions)
        turn_result = Result(record=turn_rec, remaining_positions=tr.get("remaining_positions", []))

    # River
    river_result = None
    if data.get("river"):
        rr = data["river"]
        rr_actions = [StreetAction(**a) for a in rr.get("actions", [])]
        river_rec = RiverRecord(board=rr["card"], actions=rr_actions)
        river_result = Result(
            record=river_rec,
            remaining_positions=rr.get("remaining_positions", []),
            showdown_raw=rr.get("showdown_raw"),
        )

    return cfg, pre, flop_result, turn_result, river_result, data