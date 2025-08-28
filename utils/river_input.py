# utils/river_input.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from utils.game_flow import POS_9MAX  # seating order ring

RANK = "23456789TJQKA"
SUIT = {"s": "s", "h": "h", "d": "d", "c": "c"}

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StreetAction:
    position: str                   # "CO", "BTN", ...
    action: str                     # "check" | "bet" | "call" | "raise" | "fold"
    amount: Optional[float] = None  # for bet/raise: "to" amount

@dataclass(frozen=True)
class RiverRecord:
    board: str                      # e.g., "4c"
    actions: List[StreetAction] = field(default_factory=list)

    @property
    def tokens(self) -> str:
        # Position-aware, amounts omitted for retrieval stability
        return " | ".join(f"{a.position}:{a.action}" for a in self.actions)

    def pretty(self) -> str:
        lines: List[str] = [f"River: {self.board}"]
        for a in self.actions:
            if a.action in ("bet", "raise"):
                lines.append(f"{a.position} {a.action}s to ${_trim(a.amount or 0.0)}")
            elif a.action == "check":
                lines.append(f"{a.position} checks")
            elif a.action == "call":
                lines.append(f"{a.position} calls")
            else:
                lines.append(f"{a.position} folds")
        return "\n".join(lines)

@dataclass(frozen=True)
class RiverResult:
    record: RiverRecord
    remaining_positions: List[str]          # in table order, showdown-eligible
    showdown_raw: Optional[str] = None      # collected only if 2+ remain

# -----------------------------------------------------------------------------
# Public intake
# -----------------------------------------------------------------------------

def collect_river(
    participants_to_river: List[str],           # positions that reached the river, in table order
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> RiverResult:
    """
    Ask for the single river card + betting, with consistent '> ' prompts for actions.
    Card prompt uses exact 'What was the river?:' and a single 'River: ' input line.
    If 2+ players remain at the end, ask 'What was the showdown?' and store raw text.
    """
    board = _ask_river_card(input_fn, print_fn)

    # Defensive: unique + preserve order
    seen = set()
    still_in = [p for p in participants_to_river if (p not in seen and not seen.add(p))]

    # Acting order: first seat left of BTN, then wrap, filtered by still_in
    order = _postflop_acting_order(still_in)

    log: List[StreetAction] = []
    current_to: float = 0.0
    aggressor: Optional[str] = None
    invested: set[str] = set()  # players who voluntarily put chips in ON THE RIVER

    def reset_to_act_from(start_pos: str) -> List[str]:
        """Everyone else (still in) after start_pos, wrapping, until back to start_pos."""
        n = len(still_in)
        si = still_in.index(start_pos)
        out = []
        k = (si + 1) % n
        while True:
            if still_in[k] != start_pos:
                out.append(still_in[k])
            k = (k + 1) % n
            if k == (si + 1) % n:
                break
        return out

    to_act = order.copy()

    while to_act:
        pos = to_act.pop(0)

        if pos not in still_in:
            continue

        if current_to == 0.0:
            # No bet yet: check / fold / bet amount
            action = _ask_river_no_bet_action(pos, input_fn, print_fn)
            if action == "check":
                log.append(StreetAction(pos, "check"))
            elif action == "fold":
                # Free-fold: silent (no prior river investment)
                still_in.remove(pos)
                to_act = [x for x in to_act if x != pos]
                if len(still_in) <= 1:
                    break
            else:
                amount = action
                current_to = amount
                aggressor = pos
                invested.add(pos)
                log.append(StreetAction(pos, "bet", amount))
                to_act = reset_to_act_from(aggressor)
        else:
            # Facing a bet: fold / call / raise-to
            action = _ask_river_vs_bet_action(pos, current_to, input_fn, print_fn)
            if action == "fold":
                if pos in invested:
                    log.append(StreetAction(pos, "fold"))
                if pos in still_in:
                    still_in.remove(pos)
                to_act = [x for x in to_act if x != pos]
                if len(still_in) <= 1:
                    to_act = []  # bet/raise wins the pot
            elif action == "call":
                invested.add(pos)
                log.append(StreetAction(pos, "call"))
            else:
                raise_to = action
                if raise_to <= current_to:
                    print_fn("Raise must be greater than the current bet. Try again.")
                    to_act.insert(0, pos)
                    continue
                invested.add(pos)
                current_to = raise_to
                aggressor = pos
                log.append(StreetAction(pos, "raise", raise_to))
                to_act = reset_to_act_from(aggressor)

    record = RiverRecord(board=board, actions=log)

    showdown_text: Optional[str] = None
    if len(still_in) >= 2:
        # Ask for showdown now; you can parse later
        print_fn("What was the showdown?")
        showdown_text = input_fn("> ").strip() or None

    return RiverResult(record=record, remaining_positions=still_in, showdown_raw=showdown_text)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ask_river_card(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    """
    EXACT prompt style requested:
      What was the river?:
      River: 4c
    """
    print_fn("What was the river?:")
    while True:
        raw = input_fn("River: ").strip()  # no '> ' prefix for the card prompt
        c = _normalize_card(raw)
        if c:
            return c
        print_fn("Please enter a valid card like 4c, As, Td (s/h/d/c suits).")

def _postflop_acting_order(participants: List[str]) -> List[str]:
    """
    First seat left of BTN, then wrap.
    participants must be in table order, subset of POS_9MAX.
    """
    if not participants:
        return []
    ring = POS_9MAX
    btn_idx = ring.index("BTN")
    order_ring = ring[btn_idx+1:] + ring[:btn_idx+1]  # SB, BB, UTG, ..., BTN
    return [p for p in order_ring if p in participants]

def _ask_river_no_bet_action(
    pos: str,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str | float:
    """
    Returns: 'check' | 'fold' | float (bet-to amount)
    Uses '> ' for user input (action prompts).
    """
    print_fn(f"What did {pos} do? (check, fold, or enter a bet amount)")
    while True:
        raw = input_fn("> ").strip().lower()
        if raw in ("check", "x"):
            return "check"
        if raw in ("fold", "f"):
            return "fold"
        try:
            return _parse_money(raw)  # treat numeric as BET
        except ValueError:
            print_fn("Enter 'check', 'fold', or a bet amount (e.g., 200 or $200).")

def _ask_river_vs_bet_action(
    pos: str,
    current_to: float,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str | float:
    """
    Returns: 'fold' | 'call' | float (raise-to amount > current_to)
    """
    print_fn(f"{pos} facing ${_trim(current_to)}. (fold, call, or enter a raise-to amount)")
    while True:
        raw = input_fn("> ").strip().lower()
        if raw in ("fold", "f"):
            return "fold"
        if raw in ("call", "c"):
            return "call"
        try:
            amt = _parse_money(raw)
            return amt
        except ValueError:
            print_fn("Enter 'fold', 'call', or a raise-to amount (e.g., 600 or $600).")

def _normalize_card(card: str) -> Optional[str]:
    if not card:
        return None
    s = card.strip().replace(" ", "")
    if len(s) != 2:
        return None
    rank, suit = s[0].upper(), s[1].lower()
    if rank not in RANK or suit not in SUIT:
        return None
    return f"{rank}{suit}"

def _parse_money(s: str) -> float:
    cleaned = s.replace("$", "").replace(" ", "")
    val = float(cleaned)
    if val <= 0:
        raise ValueError("amount must be positive")
    return val

def _trim(x: float) -> str:
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")
