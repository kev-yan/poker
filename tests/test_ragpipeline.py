import json
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from langchain_openai import ChatOpenAI
import weaviate
from utils.embedding import embed
from utils.rag_pipeline import enrich_user_hand_with_llm, search_weaviate_hybrid

FIXTURE = Path("data/test_fixtures/last_hand_fixture.json")

def test_reuse_fixture():
    print("entered")
    data = json.loads(FIXTURE.read_text())
    print("Here???")

    raw_query = data.get("raw_query")
    assert raw_query, "Fixture missing raw_query. Re-run CLI to generate it."
    print("HERE IS RAW QUERY:", raw_query)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    enriched_query = enrich_user_hand_with_llm(llm, raw_query)

    # Connect Weaviate
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        skip_init_checks=True,
    )
    try:
        objects, debug = search_weaviate_hybrid(
            client=client,
            collection_name="PokerHand",
            enriched_query_doc=enriched_query,  # or an enriched version
            top_k=5,
            alpha=0.5,
            embed_fn=embed,                # 384-D embed, same as ingestion
        )
        print("after hybrid search:")
        print("Hybrid debug:", debug)
        for o in objects:
            title = o.properties.get("title")
            score = getattr(o.metadata, "score", None)
            print(f"{score:.3f}  {title}")
        assert len(debug) > 0
    finally:
        client.close()

def main():
    test_reuse_fixture()

if __name__ == "__main__":
    main()