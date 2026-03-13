# utils/preflop_input.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from utils.game_flow import GameFlow, POS_9MAX
from utils.hand_input import HandConfig

# ---------- Data models ----------
@dataclass(frozen=True)
class PreflopAction:
    position: str                   # "UTG", "BTN", ...
    action: str                     # "raise" | "call" | "fold"
    amount: Optional[float] = None  # for raises: "raised to" amount

@dataclass(frozen=True)
class PreflopRecord:
    opener: str
    open_to: float
    trail_actions: List[PreflopAction] = field(default_factory=list)
    bb: float = 0.0
    straddle: float = 0.0

    @property
    def _first_to_call(self) -> float:
        # Price to call preflop is straddle (if present) else BB
        return self.straddle if self.straddle and self.straddle > 0 else self.bb

    @property
    def opener_raised(self) -> bool:
        # Limp if open_to == to_call; raise only if strictly greater
        return self.open_to > self._first_to_call

    @property
    def raises_count(self) -> int:
        return (1 if self.opener_raised else 0) + sum(
            1 for a in self.trail_actions if a.action == "raise"
        )

    @property
    def pot_type(self) -> str:
        rc = self.raises_count
        if rc == 0: return "limped"
        if rc == 1: return "single-raised"
        if rc == 2: return "3-bet"
        if rc == 3: return "4-bet"
        return f"{rc+1}-bet"

    @property
    def players_to_flop(self) -> int:
        """
        Count final participants who put money in (call/raise) and did not fold by end of round.
        Silent folds (never invested) are not counted.
        """
        final = {self.opener: "raise"}  # opener invested
        for a in self.trail_actions:
            final[a.position] = a.action
        return sum(1 for act in final.values() if act in ("call", "raise"))

    @property
    def preflop_tokens(self) -> str:
        """
        If opener limps, the first subsequent raise is the 'open' (not a 3-bet).
        """
        parts: List[str] = [f"{self.opener}:{'open' if self.opener_raised else 'limp'}"]
        raise_i = 0
        for a in self.trail_actions:
            if a.action == "raise":
                raise_i += 1
                if self.opener_raised:
                    # opener raised → first re-raise is 3-bet
                    lbl = "3bet" if raise_i == 1 else ("4bet" if raise_i == 2 else "xbet")
                else:
                    # opener limped → first raise is the 'open', then 3-bet, 4-bet, ...
                    lbl = ("open" if raise_i == 1 else
                           "3bet" if raise_i == 2 else
                           "4bet" if raise_i == 3 else
                           "xbet")
                parts.append(f"{a.position}:{lbl}")
            elif a.action == "call":
                parts.append(f"{a.position}:call")
            else:
                parts.append(f"{a.position}:fold")
        return " | ".join(parts)

    def pretty(self) -> str:
        lines: List[str] = []
        if self.opener_raised:
            lines.append(f"{self.opener} raises to ${_trim(self.open_to)}")
        else:
            lines.append(f"{self.opener} limps")
        for a in self.trail_actions:
            if a.action == "raise":
                lines.append(f"{a.position} raises to ${_trim(a.amount or 0.0)}")
            elif a.action == "call":
                lines.append(f"{a.position} calls")
            else:
                lines.append(f"{a.position} folds")
        lines.append(f"\n{self.players_to_flop} players to the flop")
        return "\n".join(lines)
# ---------- Round tracker ----------
class PreflopTracker:
    """
    Minimal betting-round logic:
      - Tracks who owes action (to_act) after each (re)raise.
      - Auto-folds skipped seats, but only logs a 'fold' if that seat had voluntarily invested.
      - Prevents finishing the round while players still owe action.
    """

    def __init__(self, gf: GameFlow, opener: str, cfg: HandConfig):
        self.gf = gf
        self.opener = opener
        self.last_actor = opener
        self.last_aggressor = opener
        self.sb = cfg.stakes_sb
        self.bb = cfg.stakes_bb
        self.straddle = cfg.straddle if hasattr(cfg, "straddle") else 0.0
        self.effective_stack = cfg.effective_stack if hasattr(cfg, "effective_stack") else 0.0

        # Only the opener has voluntarily invested initially (NOT the blinds).
        self.invested: set[str] = {opener}

        self.to_act = self._compute_to_act_from(self.last_aggressor)

    def _compute_to_act_from(self, start: str) -> List[str]:
        order = self.gf.clockwise_between(start, start)  # full ring once
        return [p for p in order if self.gf.is_active(p) and p != start]

    def _remove_from_to_act(self, pos: str) -> None:
        if pos in self.to_act:
            self.to_act.remove(pos)

    def eligible_positions(self) -> List[str]:
        return list(self.to_act)

    def auto_fold_skipped(self, next_pos: str, actions_log: List[PreflopAction]) -> None:
        """
        Fold any still-active seats we 'jumped over' (last_actor -> next_pos) within the current to_act list.
        Only append a fold action if that seat had voluntarily invested (called/raised).
        """
        if not self.to_act:
            return
        skipped = self.gf.clockwise_between(self.last_actor, next_pos)
        for p in skipped:
            if p in self.to_act and self.gf.is_active(p):
                self.gf.fold(p)
                self._remove_from_to_act(p)
                if p in self.invested:
                    actions_log.append(PreflopAction(position=p, action="fold"))

    def on_call(self, pos: str) -> None:
        self.invested.add(pos)      # calling invests
        self._remove_from_to_act(pos)
        self.last_actor = pos

    def on_fold(self, pos: str) -> None:
        self.gf.fold(pos)
        self._remove_from_to_act(pos)
        self.last_actor = pos
        # If they fold without prior investment, we don't add them to invested and we don't log.

    def on_raise(self, pos: str) -> None:
        self.invested.add(pos)      # raising invests
        self.last_actor = pos
        self.last_aggressor = pos
        # After a raise, everyone else owes action again (who's still active)
        self.to_act = self._compute_to_act_from(self.last_aggressor)

    def round_complete(self) -> bool:
        return len(self.to_act) == 0

# ---------- Public intake ----------
def collect_preflop(
    cfg: HandConfig,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> PreflopRecord:
    #tracker = PreflopTracker(cfg)
    gf = GameFlow()

    print_fn("--PreFlop--")

    opener = _ask_position("Where did action start?", gf, input_fn, print_fn)
    gf.set_action_start(opener)

    tracker = PreflopTracker(gf, opener, cfg)

    open_to = _ask_money("How much did they bet? (e.g. $10 or enter big blind amount if limped)", input_fn, print_fn)

    actions: List[PreflopAction] = []

    # Continue until no one owes action
    while True:
        if tracker.round_complete():
            break

        next_pos = _ask_position(
            "What position acts next?",
            gf, input_fn, print_fn,
            choices=tracker.eligible_positions(),
            allow_done=False,  # can't end early while players owe action
        )

        # Auto-fold skipped seats between last actor and chosen next actor
        tracker.auto_fold_skipped(next_pos, actions)

        act = _ask_action(input_fn, print_fn)  # 'call' | 'fold' | float raise-to
        if act == "fold":
            # Log fold only if they had already invested
            if next_pos in tracker.invested:
                actions.append(PreflopAction(position=next_pos, action="fold"))
            tracker.on_fold(next_pos)
        elif act == "call":
            actions.append(PreflopAction(position=next_pos, action="call"))
            tracker.on_call(next_pos)
        else:
            amount = act  # float
            # Treat as 'call' if raise amount equals the big blind (or straddle)
            if amount == tracker.bb or (amount == tracker.straddle and tracker.straddle > 0):
                actions.append(PreflopAction(position=next_pos, action="call"))
                tracker.on_call(next_pos)
            elif amount < tracker.bb:
                print_fn(f"Amount raised ({amount}) is less than the big blind ({tracker.bb}). Please try again.")
                continue
            else:
                actions.append(PreflopAction(position=next_pos, action="raise", amount=amount))
                tracker.on_raise(next_pos)

    record = PreflopRecord(opener=opener, open_to=open_to, trail_actions=actions, bb=tracker.bb, straddle=tracker.straddle)

    # Summary for CLI
    print_fn("")
    print_fn(record.pretty())
    print_fn(
        f"\nDerived: pot_type={record.pot_type}, raises={record.raises_count}, "
        f"preflop_tokens='{record.preflop_tokens}'"
    )
    return record

# ---------- Prompts & parsing (consistent '> ') ----------
def _ask_position(
    prompt: str,
    gf: GameFlow,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
    choices: Optional[List[str]] = None,
    allow_done: bool = False,
) -> Optional[str]:
    print_fn(f"{prompt}")
    opts = choices if choices is not None else gf.active_positions()
    for i, p in enumerate(opts, 1):
        print_fn(f"{i}. {p}")
    if allow_done:
        print_fn("0. Done")
    index = {str(i): p for i, p in enumerate(opts, 1)}
    if allow_done:
        index["0"] = None
    while True:
        raw = input_fn("> ").strip()
        if raw in index:
            return index[raw]
        print_fn("Please enter a valid number from the list.")

def _ask_money(
    prompt: str,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> float:
    print_fn(prompt)
    while True:
        raw = input_fn("> ").strip()
        try:
            return _parse_money(raw)
        except ValueError:
            print_fn("Please enter a positive number, e.g., 10 or $16.")

def _ask_action(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str | float:
    """
    Returns:
      - 'call' | 'fold'  or
      - float (raise-to amount)
    """
    print_fn("What was the action? (call, fold, or enter the raise-to amount)")
    while True:
        raw = input_fn("> ").strip().lower()
        if raw in ("call", "c"):
            return "call"
        if raw in ("fold", "f"):
            return "fold"
        try:
            return _parse_money(raw)  # treat numeric as raise-to
        except ValueError:
            print_fn("Enter 'call', 'fold', or a raise-to amount (e.g., 16 or $16).")

def _parse_money(s: str) -> float:
    cleaned = s.replace("$", "").replace(" ", "")
    val = float(cleaned)
    if val <= 0:
        raise ValueError("amount must be positive")
    return val

def _trim(x: float) -> str:
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")
