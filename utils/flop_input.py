# utils/flop_input.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from utils.game_flow import POS_9MAX  # seating order ring

RANK = "23456789TJQKA"
SUIT = {"s": "s", "h": "h", "d": "d", "c": "c"}

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StreetAction:
    position: str                   # "UTG1", "BTN", ...
    action: str                     # "check" | "bet" | "call" | "raise" | "fold"
    amount: Optional[float] = None  # for bet/raise: "to" amount

@dataclass(frozen=True)
class FlopRecord:
    board: str                      # e.g., "AsTd2d"
    actions: List[StreetAction] = field(default_factory=list)

    @property
    def tokens(self) -> str:
        # Position-aware, amounts omitted for retrieval stability
        parts: List[str] = []
        for a in self.actions:
            parts.append(f"{a.position}:{a.action}")
        return " | ".join(parts)

    def pretty(self) -> str:
        lines: List[str] = [f"Flop: {self.board}"]
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
class FlopResult:
    record: FlopRecord
    remaining_positions: List[str]  # in table-order, for the next street

# -----------------------------------------------------------------------------
# Public intake
# -----------------------------------------------------------------------------

def collect_flop(
    participants_to_flop: List[str],           # positions that reached the flop, in table order
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> FlopResult:
    """
    Ask for flop board + betting, with consistent '> ' prompts for actions.
    Card prompts use exact 'Card X: <val>' with no '>' as requested.
    """
    board = _ask_flop_board(input_fn, print_fn)

    # Defensive: ensure list is unique and preserves table order
    seen = set()
    still_in = [p for p in participants_to_flop if (p not in seen and not seen.add(p))]

    # Determine flop acting order: first seat left of BTN, then wrap, filtered by still_in
    order = _flop_acting_order(still_in)

    # State for the betting round
    log: List[StreetAction] = []
    current_to: float = 0.0        # current "bet to" amount on the table (0 if no bet yet)
    aggressor: Optional[str] = None
    invested: set[str] = set()     # players who voluntarily put chips in ON THE FLOP (bet/raise/call)

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

    # Initialize "to_act" (no bet yet): everyone acts once in order.
    to_act = order.copy()

    while to_act:
        pos = to_act.pop(0)

        # If player has been removed mid-round (folded earlier), skip
        if pos not in still_in:
            continue

        if current_to == 0.0:
            # No bet yet: ask 'check' or a bet amount (allow explicit 'fold', rare on a free option)
            action = _ask_flop_no_bet_action(pos, input_fn, print_fn)
            if action == "check":
                log.append(StreetAction(pos, "check"))
                # continue; if everyone checks, the street ends when to_act empties
            elif action == "fold":
                # Free-fold: silent in transcript (no prior flop investment)
                still_in.remove(pos)
                to_act = [x for x in to_act if x != pos]
                if len(still_in) <= 1:
                    break
            else:
                # Numeric -> bet to amount
                amount = action
                current_to = amount
                aggressor = pos
                invested.add(pos)
                log.append(StreetAction(pos, "bet", amount))
                # Everyone else now owes action again from after the bettor
                to_act = reset_to_act_from(aggressor)
        else:
            # Facing a bet: ask fold / call / raise-to
            action = _ask_flop_vs_bet_action(pos, current_to, input_fn, print_fn)
            if action == "fold":
                # Visible fold only if they had invested on the flop already
                if pos in invested:
                    log.append(StreetAction(pos, "fold"))
                if pos in still_in:
                    still_in.remove(pos)
                to_act = [x for x in to_act if x != pos]
                if len(still_in) <= 1:
                    # Everyone else folded to the aggression—round ends
                    to_act = []
            elif action == "call":
                invested.add(pos)
                log.append(StreetAction(pos, "call"))
                # continue; round will close when to_act empties (no new raises)
            else:
                # Numeric -> raise-to amount
                raise_to = action
                if raise_to <= current_to:
                    print_fn("Raise must be greater than the current bet. Try again.")
                    to_act.insert(0, pos)
                    continue
                invested.add(pos)
                current_to = raise_to
                aggressor = pos
                log.append(StreetAction(pos, "raise", raise_to))
                # Re-open action: everyone else (still in) after the raiser owes action
                to_act = reset_to_act_from(aggressor)

    record = FlopRecord(board=board, actions=log)
    return FlopResult(record=record, remaining_positions=still_in)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ask_flop_board(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    """
    EXACT prompt style requested:
      What was the flop? ("As" for Ace of Spades)
      Card 1: Ts
      Card 2: Td
      Card 3: 9h
    """
    print_fn('What was the flop? ("As" for Ace of Spades)')
    c1 = _ask_card("Card 1: ", input_fn, print_fn)  # no '>' here
    c2 = _ask_card("Card 2: ", input_fn, print_fn)
    c3 = _ask_card("Card 3: ", input_fn, print_fn)
    # Ensure distinct
    while len({c1, c2, c3}) < 3:
        print_fn("Duplicate cards detected. Re-enter the flop.")
        c1 = _ask_card("Card 1: ", input_fn, print_fn)
        c2 = _ask_card("Card 2: ", input_fn, print_fn)
        c3 = _ask_card("Card 3: ", input_fn, print_fn)
    return f"{c1}{c2}{c3}"

def _ask_card(
    prompt: str,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    # IMPORTANT: no '>' prefix for these prompts
    while True:
        raw = input_fn(prompt).strip()
        c = _normalize_card(raw)
        if c:
            return c
        print_fn("Please enter a valid card like As, Td, 2d (s/h/d/c).")

def _flop_acting_order(participants: List[str]) -> List[str]:
    """
    Return acting order starting with first seat left of BTN, then wrap.
    participants must be in table order, subset of POS_9MAX.
    """
    if not participants:
        return []
    ring = POS_9MAX
    btn_idx = ring.index("BTN")
    order_ring = ring[btn_idx+1:] + ring[:btn_idx+1]  # SB, BB, UTG, ..., BTN
    return [p for p in order_ring if p in participants]

def _ask_flop_no_bet_action(
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
            print_fn("Enter 'check', 'fold', or a bet amount (e.g., 55 or $55).")

def _ask_flop_vs_bet_action(
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
            print_fn("Enter 'fold', 'call', or a raise-to amount (e.g., 150 or $150).")

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
