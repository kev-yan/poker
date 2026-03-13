# Poker Coach AI

An AI-powered poker coaching CLI. Input a hand you played street-by-street, and the system retrieves structurally similar hands from a curated coaching knowledge base, then generates personalized, street-by-street advice grounded in real poker theory.

---

## The Problem

Getting quality poker coaching is expensive and slow. This system approximates a professional coach by:
1. Finding structurally similar hands from a curated coaching corpus (not just asking an LLM to improvise)
2. Using those retrieved hands as grounded evidence for LLM-generated feedback — anchoring advice in real poker theory and preventing hallucination

---

## How It Works

```
User Input (CLI)
       │
       ▼
Hand Config + Street-by-Street Collection
(utils/hand_input.py, preflop_input.py, flop_input.py, turn_input.py, river_input.py)
       │
       ▼
RAG Schema Builder — structured facets + embedding text
(utils/rag_builder.py → utils/rag_pipeline.py::build_final_rag_schema)
       │
       ▼
LLM Enrichment — GPT maps hand → corpus-aligned schema with tags/flags
(utils/rag_pipeline.py::enrich_user_hand_with_llm)
       │
       ▼
Hybrid Weaviate Search — BM25 + dense vector, tag-weighted
(utils/rag_pipeline.py::search_weaviate_hybrid)
       │
       ▼
Coaching Generation — LLM synthesizes retrieved evidence + hand NL summary
(utils/coaching.py::generate_coaching_advice)
```

### Why Two LLM Calls?

The first call (`enrich_user_hand_with_llm`) maps your raw hand into the same structured vocabulary as the coaching corpus — same enums, same improvement flags, same schema tokens. This is what makes the vector search actually work; without it, user-entered hands and corpus hands would use different language for the same concepts.

The second call (`generate_coaching_advice`) does the actual coaching, but it's fed both your specific hand and the top-3 retrieved evidence cards as context. The model is instructed to integrate the retrieved advice and only deviate when the evidence is clearly inapplicable.

---

## Tech Stack

| Layer | Technology |
|---|---|
| CLI / Input | Python (`input()` loops, `dataclasses`) |
| Hand State Machine | Custom `GameFlow` + `PreflopTracker` |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers, runs locally) |
| Vector DB | Weaviate Cloud (hybrid BM25 + dense search) |
| LLM Enrichment | OpenAI GPT via LangChain (`ChatOpenAI`) with structured JSON output |
| Coaching LLM | OpenAI GPT (gpt-5-mini) |
| Data Ingestion | OpenCV + OCR pipeline for extracting hands from poker reels |

---

## Project Structure

```
poker/
├── cli/
│   ├── main.py              # Entry point, menu loop
│   └── workflows.py         # Full pipeline orchestration (run_workflow_from_new_hand)
│
├── utils/
│   ├── hand_input.py        # HandConfig: stakes, positions, stack depth
│   ├── preflop_input.py     # PreflopRecord, PreflopTracker state machine
│   ├── flop_input.py        # FlopRecord + street collection
│   ├── turn_input.py        # TurnRecord + street collection
│   ├── river_input.py       # RiverRecord + showdown collection
│   ├── game_flow.py         # GameFlow: seat ordering, clockwise iteration
│   ├── rag_builder.py       # build_final_rag_schema (raw schema, no LLM)
│   ├── rag_pipeline.py      # enrich_user_hand_with_llm + search_weaviate_hybrid
│   ├── embedding.py         # BAAI/bge-small-en embed() wrapper
│   └── coaching.py          # generate_coaching_advice + evidence card builders
│
├── llm/
│   ├── prompts.py           # Original OpenAI coaching prompt (v1)
│   ├── coach_api.py         # LLM dispatch stub
│   └── weaviate/
│       ├── create_collection.py  # Weaviate schema setup
│       └── ingest_data.py        # Batch embed + upsert corpus entries
│
├── scripts/
│   ├── extract_captions.py  # OpenCV frame loop → OCR text extraction
│   ├── ocr_utils.py         # Frame-level OCR helpers
│   ├── filtering.py         # Redundancy filtering for caption dedup
│   └── config.py            # Path config for video ingestion
│
├── tests/
│   ├── test_rag_pipeline.py # RAG pipeline integration tests
│   └── test_fixtures.py     # Hand fixture serialization helpers
│
└── data/
    ├── sample_hand.json     # Sample hand for CLI demo
    └── ig_format.json       # Instagram reel hand format spec
```

---

## Key Components

### Hand State Machine (`utils/preflop_input.py`)

`PreflopTracker` models the betting round as a state machine. It tracks which seats still owe action after each raise, auto-folds skipped seats, and only logs a fold if the seat had voluntarily invested. `PreflopRecord` derives `pot_type` (limped / single-raised / 3-bet / 4-bet / ...) and `preflop_tokens` (compact action string like `UTG:open | CO:call | BTN:3bet`) from the collected actions.

### Board Texture Analysis (`utils/rag_builder.py`)

`_analyze_runout` classifies both the flop and the full runout (through river) into structured features: flush level, paired level, coarse straightness (0/1/2), high card bucket (H/M/L), and texture class (dry / paired / monotone / draw-heavy). These become filterable facets in Weaviate.

### LLM Enrichment (`utils/rag_pipeline.py::enrich_user_hand_with_llm`)

Takes the raw hand schema and calls GPT with a structured output schema (`PokerRAGEntry`) to produce corpus-aligned tags and improvement flags. The 50+ improvement flags span 7 categories:

- **A** — Board/texture nuance (`dynamic_board`, `wet_board`, `monotone_pressure`, ...)
- **B** — Runout events (`turn_completes_flush`, `river_pairs_top`, `scare_card_river`, ...)
- **C** — Range/nut advantage and formation (`hero_IP`, `range_advantage_hero`, `blind_vs_blind`, ...)
- **D** — Hand-state & blockers (`bluff_catcher_river`, `flush_blocker_pressure`, ...)
- **E** — Line & sizing semantics (`overbet_polar_node`, `polar_turn_barrel`, `thin_value_ok`, ...)
- **F** — Pool/opponent tendencies (`pool_low_bluff`, `sticky_pool`, `nit_villain`, ...)
- **G** — Pricing & equity realization (`fold_equity_low`, `deny_equity_priority`, ...)

### Hybrid Search (`utils/rag_pipeline.py::search_weaviate_hybrid`)

Tags-first hybrid search (BM25 + dense vector, `alpha=0.45`). Tags and improvement flags carry the most BM25 weight (`^4`), followed by texture class and hero hand class (`^2`), with hard filters only on `pot_type` and `street_focus`. Boost tokens from `tags`, `improvement_flags`, and `line_tokens` are appended to the query text for soft keyword boosting.

### Coaching Generation (`utils/coaching.py`)

The coaching system prompt defines a formal similarity policy — when retrieved evidence should be used vs. discarded — based on street, formation, board state, SPR bucket, and line overlap. The LLM is instructed to merge the 2–4 strongest retrieved ideas and produce a crisp, street-by-street breakdown with sizing recommendations and GTO/exploitative framing.

### Video Ingestion Pipeline (`scripts/`)

OpenCV-based pipeline that processes `.mp4` poker reels frame-by-frame, runs OCR on each frame, deduplicates redundant captions, and writes timestamped output files. Used to bootstrap the coaching knowledge base from instructional video content.

---

## Setup

### Prerequisites

- Python 3.11+
- A [Weaviate Cloud](https://weaviate.io/developers/wcs) cluster
- OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-key
```

### Run the CLI

```bash
cd cli
python main.py
```

---

## Weaviate Collection Setup

To create the `PokerHand` collection and ingest corpus data:

```bash
# Create collection schema
python llm/weaviate/create_collection.py

# Ingest annotated hand entries
python llm/weaviate/ingest_data.py
```

Corpus entries are expected at `data/output/rag_entries_v2.json`. Each entry should include `title`, `schema_string`, `embedding_text`, `facets`, `soft_signals`, `tags`, and `annotated_coaching_description`.

---

## Running Tests

```bash
pytest tests/
```

---

## Design Decisions

**Why local embeddings?** `BAAI/bge-small-en-v1.5` runs locally (no API call per query), is fast, and produces high-quality embeddings for structured text. Consistency matters more than raw quality here — the same model embeds both corpus and query documents.

**Why two LLM calls instead of one?** The enrichment call normalizes user vocabulary to match the corpus before search. Without this step, a user typing "I was on the button in a 3-bet pot" would miss corpus entries tagged `BTN:call | CO:3bet` because the surface text differs. The enrichment bridges that gap.

**Why hard filters only on `pot_type` and `street_focus`?** Overfiltering a small corpus produces zero results. Most selectivity comes from the weighted BM25 and dense vector scores, not hard exclusion.
