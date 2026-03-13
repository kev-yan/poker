# tests/test_loo_variants.py
import os, json, re
from pathlib import Path

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter

# repo-local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "output" / "rag_entries_v2.json"
COLLECTION = "PokerHand"
from utils.embedding import embed  # MUST be same model as ingest

ALPHA       = 0.45            # blend of vector/BM25; try 0.35/0.45/0.60 later
TOPK        = 10

def load_entries() -> list[dict]:
    data = json.loads(JSON_PATH.read_text())
    assert isinstance(data, list)
    return data

def self_uuid(entry: dict) -> str:
    # MUST match your ingest seed exactly
    return generate_uuid5(entry.get("title") or entry.get("schema_string"))

def mrr(rank: int) -> float:
    return 0.0 if rank is None else 1.0 / (rank + 1)  # rank is 0-based

def strip_line_segment(text: str) -> str:
    # remove "line=..." once; keep everything else
    return re.sub(r"line=[^|]*\|?\s*", "", text, count=1)

def build_variants(entry: dict, stored: dict) -> dict[str, str]:
    """
    Return dict of query_text variants.
    `stored` is the object fetched from Weaviate (source of truth).
    """
    embedding_text = (stored.get("embedding_text") or "").strip()
    schema_string  = (entry.get("schema_string") or "").strip()

    tags = [t.lower() for t in entry.get("tags", [])]
    flags = [t.lower() for t in (entry.get("soft_signals", {}) or {}).get("improvement_flags", [])]
    # optionally include a couple of light facets to help BM25 for tags_only
    facets = entry.get("facets", {}) or {}
    extras = []
    for k in ("street_focus", "hero_hand_class", "texture_class"):
        v = (facets.get(k) or "").lower()
        if v: extras.append(v)

    keywords_join = " ".join(sorted(set(tags + flags + extras))).strip()

    variants = {
        "baseline": embedding_text,
        "plus_keywords": f"{embedding_text}\nkeywords: {keywords_join}" if keywords_join else embedding_text,
        "schema_only": schema_string,
        "tags_only": f"keywords: {keywords_join}" if keywords_join else "",
        "no_line": strip_line_segment(embedding_text),
        "noise": f"{embedding_text}\nnoise: overpair top pair underpair flush draw straight draw",
    }
    return variants

def eval_variant(col, uid: str, qtext: str, use_filters: bool = False, use_weights: bool = False) -> int | None:
    """
    Run one query and return 0-based rank of uid, or None if not in top-K.
    """
    vec = embed(qtext) if qtext else embed(" ")  # avoid empty string to embedder

    kwargs = {
        "query": qtext,
        "vector": vec,
        "alpha": ALPHA,
        "limit": TOPK,
        "return_properties": ["title", "tags", "street_focus"],
    }

    # Optional: add a light filter (street_focus) to test stability under filtering
    if use_filters:
        # you can parameterize which value to filter on by passing it in; here we leave off for generality
        pass

    # Optional: add property-weight profile (tags-first)
    if use_weights:
        kwargs["query_properties"] = [
            "tags^6","improvement_flags^4","schema_string^3","texture_class^3",
            "line_compact^3","hero_hand_class^2","board_texture^2",
            "spr_bucket","stack_depth","positions","line_tokens","embedding_text^2",
        ]

    resp = col.query.hybrid(**kwargs)
    objs = resp.objects or []
    ids = [str(o.uuid) for o in objs]
    uid_str = str(uid)
    try:
        return ids.index(uid_str)
    except ValueError:
        return None

def summarize_metrics(all_ranks: list[int | None], label: str) -> None:
    n = len(all_ranks)
    self1 = sum(1 for r in all_ranks if r == 0) / n
    self3 = sum(1 for r in all_ranks if r is not None and r <= 2) / n
    self5 = sum(1 for r in all_ranks if r is not None and r <= 4) / n
    mrr10 = sum(mrr(r) for r in all_ranks) / n
    print(f"{label:>14} | Self@1 {self1:.3f}  Self@3 {self3:.3f}  Self@5 {self5:.3f}  MRR@10 {mrr10:.3f}")

def main():
    load_dotenv()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        skip_init_checks=True,
    )
    col = client.collections.get(COLLECTION)

    entries = load_entries()

    # Fetch stored text once per entry to avoid drift
    stored_by_uid = {}
    for e in entries:
        uid = self_uuid(e)
        obj = col.query.fetch_object_by_id(
            uid,
            return_properties=["title","embedding_text","street_focus","tags"]
        )
        if obj is None:
            print(f"[WARN] Missing in DB: {e.get('title')}")
            continue
        stored_by_uid[uid] = obj.properties

    # Variants to evaluate
    VARIANT_NAMES = ["baseline","plus_keywords","schema_only","tags_only","no_line","noise"]

    # Run evaluation
    print(f"Docs: {len(entries)} | alpha={ALPHA} | topK={TOPK}")
    results = {name: [] for name in VARIANT_NAMES}

    for e in entries:
        uid = self_uuid(e)
        stored = stored_by_uid.get(uid)
        if not stored:
            for name in VARIANT_NAMES:
                results[name].append(None)
            continue

        variants = build_variants(e, stored)

        for name in VARIANT_NAMES:
            qtext = variants[name]
            r = eval_variant(col, uid, qtext, use_filters=False, use_weights=False)
            results[name].append(r)

            # If you'd like to inspect misses:
            # if r is None:
            #     print(f"[MISS] {e.get('title')} | variant={name}")

    # Summaries
    print("\n=== LOO Robustness (no filters, no weights) ===")
    for name in VARIANT_NAMES:
        summarize_metrics(results[name], name)

    # (Optional) Re-run with weights to see if robustness improves
    print("\n=== LOO Robustness (+ property weights) ===")
    for name in VARIANT_NAMES:
        ranks_weighted = []
        for e in entries:
            uid = self_uuid(e)
            stored = stored_by_uid.get(uid)
            if not stored:
                ranks_weighted.append(None)
                continue
            qtext = build_variants(e, stored)[name]
            r = eval_variant(col, uid, qtext, use_filters=False, use_weights=True)
            ranks_weighted.append(r)
        summarize_metrics(ranks_weighted, name)

    client.close()

if __name__ == "__main__":
    main()
