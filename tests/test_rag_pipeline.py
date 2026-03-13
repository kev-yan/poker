import json
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_openai import ChatOpenAI
import weaviate
from utils.embedding import embed
from test_fixtures import load_fixture
from utils.rag_pipeline import enrich_user_hand_with_llm, search_weaviate_hybrid, build_nl_summary
from utils.coaching import generate_coaching_advice, build_evidence_context, evidence_card_from_obj

FIXTURE = Path("data/test_fixtures/last_hand_fixture.json")

# def weaviate_obj_to_json(o):
#     """Your existing mapper (kept for completeness; used as raw dict source)."""
#     p = o.properties
#     return {
#         "title": p.get("title"),
#         "schema_string": p.get("schema_string"),
#         "embedding_text": p.get("embedding_text"),
#         "annotated_coaching_description": p.get("annotated_coaching_description"),

#         "pot_type": p.get("pot_type"),
#         "street_focus": p.get("street_focus"),
#         "hero_position": p.get("hero_position"),
#         "villain_position": p.get("villain_position"),
#         "heads_up": p.get("heads_up"),

#         "board_texture": p.get("board_texture"),
#         "texture_class": p.get("texture_class"),
#         "flush_level": p.get("flush_level"),
#         "paired_level": p.get("paired_level"),
#         "straightness": p.get("straightness"),
#         "high_card_bucket": p.get("high_card_bucket"),

#         "hero_hand_class": p.get("hero_hand_class"),
#         "blocker_pattern": p.get("blocker_pattern"),

#         "spr_bucket": p.get("spr_bucket"),
#         "stack_depth": p.get("stack_depth"),
#         "line_compact": p.get("line_compact"),

#         "tags": p.get("tags", []),
#         "players_to_flop": p.get("players_to_flop"),
#         "positions": p.get("positions", []),
#         "line_tokens": p.get("line_tokens", []),
#         "improvement_flags": p.get("improvement_flags", []),
#         "action_points": p.get("action_points", []),

#         # newer names if present
#         "coaching_summary": p.get("coaching_summary"),
#         "coaching_actions": p.get("coaching_actions", []),
#     }



def test_reuse_fixture():
    data = json.loads(FIXTURE.read_text())

    cfg, pre, flop, turn, river, data = load_fixture("data/test_fixtures/last_hand_fixture.json")

    nl_summary = build_nl_summary(
        cfg=cfg,
        pre=pre,
        flop_rec=flop.record,
        turn_rec=turn.record,
        river_rec=river.record,
    )

    print("nl_summary:", nl_summary)

    #return
    raw_query = data.get("raw_query")
    assert raw_query, "Fixture missing raw_query. Re-run CLI to generate it."
    llm = ChatOpenAI(model = "gpt-5-mini")
    enriched_query = enrich_user_hand_with_llm(llm, raw_query, cfg)
    print("\nENRICHED QUERY:", enriched_query, "\n\n\n")

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
            enriched_query_doc=enriched_query,
            top_k=3,                #top_k is the maximum number of results to return (not the top k)
            alpha=0.5,
            embed_fn=embed,
        )

        evidence_context, evidence_json = build_evidence_context(objects, k=3)

        advice = generate_coaching_advice(nl_summary, evidence_context, evidence_json, model="gpt-5-mini")
        print("\n=== Coaching Advice ===\n")
        print(advice)

    finally:
        client.close()

def main():
    test_reuse_fixture()

if __name__ == "__main__":
    main()