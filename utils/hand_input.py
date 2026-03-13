from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Callable, Tuple

# -------------------------------
# Constants / simple vocab
# -------------------------------
RANK = "23456789TJQKA"
SUIT = {"s": "s", "h": "h", "d": "d", "c": "c"}

POS_9MAX = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

# -------------------------------
# Class Definitions
# -------------------------------
@dataclass(frozen=True)
class HandConfig:
    stakes_sb: float
    stakes_bb: float
    table_size: int
    hero_position: str  
    hero_hand: str       
    effective_stack: float 
    straddle: Optional[float] = None
    ante: Optional[float] = None

    @property
    def stack_depth(self) -> str:
        """
        Returns relative stack depth label, derived on the fly from cash/bb.
        deep >= 150bb, medium >= 50bb, else short.
        """
        bb_count = self.effective_stack / self.stakes_bb
        if bb_count >= 150:
            return "deep"
        elif bb_count >= 50:
            return "medium"
        else:
            return "short"

    def schema_tokens(self) -> str:
        """
        Short machine-friendly line to seed LLM or queries.
        Uses cash for stacks since that's what you store.
        """
        stakes = f"${_trim_float(self.stakes_sb)}/${_trim_float(self.stakes_bb)}"
        stack_cash = f"${_trim_float(self.effective_stack)}"
        return (
            f"stakes={stakes} table={self.table_size}-max "
            f"pos={self.hero_position} stacks={stack_cash} hero={self.hero_hand} depth={self.stack_depth}"
        )

    def as_dict(self) -> dict:
        return asdict(self)

# -------------------------------
# Public API (interactive intake)
# -------------------------------
def collect_hand_config(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> HandConfig:
    print_fn("Poker Hand Setup:")

    sb, bb = _input_stakes(input_fn, print_fn)
    table_size = _input_table_size(input_fn, print_fn)
    hero_position = _input_position(input_fn, print_fn)
    hero_hand = _input_hand(input_fn, print_fn)
    effective_stack = _input_effective_stack(input_fn, print_fn)

    straddle = None
    ante = None

    cfg = HandConfig(
        stakes_sb=sb,
        stakes_bb=bb,
        table_size=table_size,
        hero_position=hero_position,
        hero_hand=hero_hand,
        effective_stack=effective_stack,
        straddle=straddle,
        ante=ante,
    )

    # Summary
    print_fn("\nSummary:")
    print_fn(f"  Stakes: ${_trim_float(sb)}/${_trim_float(bb)} | Table: {table_size}-max")
    print_fn(f"  Position: {hero_position} | Hand: {hero_hand}")
    print_fn(f"  Effective stack (cash): ${_trim_float(effective_stack)}  (~{_trim_float(effective_stack/bb)}bb)")
    print_fn(f"  Stack depth: {cfg.stack_depth}")
    print_fn("\nSchema tokens:")
    print_fn(f"  {cfg.schema_tokens()}")

    return cfg

# -------------------------------
# Input Helpers
# -------------------------------
def _trim_float(x: float) -> str:
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")

def _input_stakes(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Tuple[float, float]:
    """
    Input stakes in the format 'sb/bb' (e.g. '2/3').
    """
    while True:
        raw = input_fn("Enter stakes (e.g. 2/3 or 0.25/0.5): ").strip()
        cleaned = raw.replace("$", "").replace(" ", "")
        if "/" not in cleaned:
            print_fn("Invalid format. Please use 'sb/bb' (e.g. 2/3).")
            continue
        try:
            sb_s, bb_s = cleaned.split("/", 1)
            sb = float(sb_s)
            bb = float(bb_s)
            if sb <= 0 or bb <= 0:
                raise ValueError("Stakes must be positive.")
            if sb > bb:
                raise ValueError("SB must be <= BB.")
            return sb, bb
        except Exception:
            print_fn("Could not parse stakes. Please try again.")

def _input_table_size(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> int:
    """
    Input table size (2–9).
    """
    while True:
        raw = input_fn("How many handed? Enter a number between 2 and 9: ").strip()
        try:
            n = int(raw)
            if 2 <= n <= 9:
                return n
        except ValueError:
            pass
        print_fn("Please enter a whole number between 2 and 9.")

def _input_position(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    """
    Input hero position (from a 9-max list; you can tailor later).
    """
    positions = POS_9MAX
    print_fn("Please enter your position:")
    for i, pos in enumerate(positions, 1):
        print_fn(f"  {i}. {pos}")
    while True:
        raw = input_fn(f"Choose a position (1-{len(positions)}): ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(positions):
                return positions[choice - 1]
        except ValueError:
            pass
        print_fn("Invalid choice. Please enter a number from the list.")

def _input_hand(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    """
    Input hero hand (e.g. 'AsKh' for Ace of Spades, King of Hearts).
    """
    print_fn("Enter your two hole cards (use s/h/d/c for suits). Examples: As, Kd, Qh, Tc.")
    while True:
        c1 = _normalize_card(input_fn("  Card 1: "))
        c2 = _normalize_card(input_fn("  Card 2: "))
        if not c1 or not c2:
            print_fn("Could not parse cards. Try formats like As, Kd, Qh, Tc.")
            continue
        if c1 == c2:
            print_fn("You cannot have two of the same card. Try again.")
            continue
        # order by rank desc, keep suits
        r1 = RANK.index(c1[0])
        r2 = RANK.index(c2[0])
        if r1 < r2:
            c1, c2 = c2, c1
        return f"{c1}{c2}"

def _input_effective_stack(
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> float:
    """
    Input effective stack size in cash (e.g. $150 or 75).
    """
    while True:
        raw = input_fn("Enter effective stack (cash, e.g. $150 or 75): ").strip()
        cleaned = raw.replace("$", "").replace(" ", "")
        try:
            stack = float(cleaned)
            if stack <= 0:
                raise ValueError("Stack must be positive.")
            return stack
        except ValueError:
            print_fn("Invalid input. Please enter a valid cash amount (e.g. $150).")

def _normalize_card(card: str) -> Optional[str]:
    card = card.strip()
    if not card or len(card) < 2:
        return None
    rank = card[0].upper()
    suit = card[1].lower()
    if rank not in RANK or suit not in SUIT:
        return None
    return f"{rank}{suit}"
