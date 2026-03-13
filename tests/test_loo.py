# test_loo.py
import os, json, math, statistics
from pathlib import Path

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter
import sys
import os

# --- adjust these paths/names for your repo ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "output" / "rag_entries_v2.json"
COLLECTION = "PokerHand"

from utils.rag_pipeline import search_weaviate_hybrid  
from utils.embedding import embed


def load_entries(path: Path):
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    return data


def mrr(rank: int) -> float:
    return 0.0 if rank is None else 1.0 / (rank + 1)  # rank is 0-based


def self_uuid(entry: dict) -> str:
    # MUST match ingest seed exactly:
    return generate_uuid5(entry.get("title") or entry.get("schema_string"))

def main():
    load_dotenv()
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        skip_init_checks=True,
    )
    col = client.collections.get(COLLECTION)

    entries = json.loads((Path(__file__).parents[1]/"data"/"output"/"rag_entries_v2.json").read_text())

    ranks = []
    for e in entries:
        
        uid = self_uuid(e)
        print("\n\n---\nEntry:", e.get("title"), "UUID:", uid)

        obj = col.query.fetch_object_by_id(uid, return_properties=["title"])
        #print("UUID exists?", bool(obj), " title:", (obj.properties.get("title") if obj else None))

        # Build query text + vector
        qtext = (e.get("embedding_text") or e.get("schema_string") or "").strip()
        # Soft-boost tags/flags — normalize to lowercase to avoid hero_IP vs hero_ip
        boost = []
        boost += [t.lower() for t in e.get("tags", [])]
        soft = e.get("soft_signals") or {}
        boost += [t.lower() for t in soft.get("improvement_flags", [])]
        if boost:
            qtext = f"{qtext}\nkeywords: " + " ".join(sorted(set(boost)))

        print(qtext)
        vec = embed(qtext)  # <-- CRITICAL: provide vector (no server vectorizer)

        resp = col.query.hybrid(
            query=qtext,
            vector=vec,
            alpha=0.25,
            limit=5,
            # start with no filters; add {'street_focus': ...} after this passes
            return_properties=["title","street_focus","tags"],
        )

        objs = resp.objects or []
        ids = [str(o.uuid) for o in objs]   # <-- cast
        uid_str = str(uid)                  # <-- cast

        try:
            r = ids.index(uid_str)          # 0-based rank
        except ValueError:
            r = None

        if r is None:
            print(f"[MISS] {e.get('title')} not in top-{len(ids)} | top3={[o.properties.get('title') for o in objs[:3]]}")

        ranks.append(r)

    total = len(ranks)
    self1 = sum(1 for r in ranks if r == 0)/total
    self3 = sum(1 for r in ranks if r is not None and r <= 2)/total
    self5 = sum(1 for r in ranks if r is not None and r <= 4)/total
    mrr10 = sum(0 if r is None else 1/(r+1) for r in ranks)/total
    print(f"\nDocs: {total}\nSelf@1 {self1:.3f}  Self@3 {self3:.3f}  Self@5 {self5:.3f}  MRR@10 {mrr10:.3f}")

    client.close()

if __name__ == "__main__":
    main()

