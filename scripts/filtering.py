# scripts/deduplication_utils.py
from difflib import SequenceMatcher

#is_similar isn't utilized in the current codebase, but it can be useful for future deduplication tasks.
def is_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold

def is_redundant(new_text, recent_texts, min_overlap_ratio=0.8):
    for _, prev in recent_texts:
        a = new_text.strip().upper()
        b = prev.strip().upper()

        if a in b or b in a:
            return True

        tokens_a = set(a.split())
        tokens_b = set(b.split())
        overlap = tokens_a & tokens_b
        if len(overlap) / max(len(tokens_a), 1) > min_overlap_ratio:
            return True

    return False
