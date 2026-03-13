# Poker Coach AI — Design Document

## Problem Statement

Quality poker coaching is expensive and inaccessible. An LLM alone produces generic, sometimes hallucinated advice — models don't have expert-level poker theory baked in at the depth required for experienced players.

The core insight: rather than asking an LLM to generate coaching from scratch, retrieve structurally similar annotated hands from a curated knowledge base and use them as grounded evidence. The LLM synthesizes; it doesn't improvise.

---

## System Architecture

```
User Input (CLI)
      │
      ▼
Hand State Machine — collects hand street by street
(utils/preflop_input.py, flop_input.py, turn_input.py, river_input.py)
      │
      ▼
Schema Builder — converts raw hand into structured facets + embedding text
(utils/rag_builder.py)
      │
      ▼
LLM Enrichment — normalizes user vocabulary to match corpus schema
(utils/rag_pipeline.py::enrich_user_hand_with_llm)
      │
      ▼
Hybrid Search — BM25 + dense vector in Weaviate, tag-weighted
(utils/rag_pipeline.py::search_weaviate_hybrid)
      │
      ▼
Coaching Generation — LLM synthesizes retrieved evidence + hand summary
(utils/coaching.py::generate_coaching_advice)
```

---

## Component Design & Rationale

### 1. Hand State Machine (`utils/preflop_input.py`)

**Decision:** Model input as a state machine rather than free-text entry.

**Why:** Raw free-text hand descriptions are inconsistent — users describe the same action in many different ways. A state machine enforces a structured, standardized format that produces reliable schema fields downstream. Without this, schema population would be unpredictable and retrieval quality would suffer.

**Output:** `PreflopRecord` — derives `pot_type` (limped / single-raised / 3-bet / 4-bet) and a compact action string like `UTG:open | CO:call | BTN:3bet`. These become hard filters and ranking signals in Weaviate.

---

### 2. Schema Builder (`utils/rag_builder.py`)

**Decision:** Extract structured facets from the hand before embedding, rather than embedding raw text.

**Why:** Raw action strings aren't meaningfully searchable. The schema extracts the structural features that actually matter for poker strategy (board texture, pot type, position) and discards surface noise.

**Key design — board texture analysis (`_analyze_runout`):**

| Facet | Values | Purpose |
|---|---|---|
| `flush_level` | 1 / 2 / 3 | How many cards share a suit |
| `paired_level` | 0 / 1 / 2 | Board pair / trips |
| `straightness` | 0 / 1 / 2 | Coarse connectedness |
| `texture_class` | dry / paired / monotone / draw-heavy | Top-level board category |

These become filterable and rankable facets in Weaviate.

---

### 3. LLM Enrichment (`utils/rag_pipeline.py::enrich_user_hand_with_llm`)

**Decision:** Add an LLM normalization step before retrieval.

**Problem it solves — vocabulary mismatch:** A user typing "I was on the button in a 3-bet pot" and a corpus entry labeled `BTN:call | CO:3bet, IP, 3-bet` describe the same situation with different surface text. Without normalization, vector search can't bridge that gap.

**How it's constrained:** The LLM is given a JSON Schema with strict enums — it can only output values from predefined lists (pot types, positions, texture classes, improvement flags). This is **constrained generation** / structured output. It prevents the model from inventing vocabulary that won't match the corpus.

**The 50+ improvement flags** span 7 categories:
- **A** — Board/texture nuance (`wet_board`, `dynamic_board`, `monotone_pressure`)
- **B** — Runout events (`turn_completes_flush`, `scare_card_river`, `brick_river`)
- **C** — Range/position advantage (`hero_IP`, `range_advantage_hero`, `blind_vs_blind`)
- **D** — Hand state & blockers (`bluff_catcher_river`, `flush_blocker_pressure`)
- **E** — Line & sizing semantics (`overbet_polar_node`, `polar_turn_barrel`, `thin_value_ok`)
- **F** — Opponent tendencies (`pool_low_bluff`, `sticky_pool`, `nit_villain`)
- **G** — Pricing & equity (`fold_equity_low`, `deny_equity_priority`)

---

### 4. Hybrid Search (`utils/rag_pipeline.py::search_weaviate_hybrid`)

**Decision:** Hybrid BM25 + dense vector search, with `alpha=0.45` and tag-field boosting.

**Why hybrid over pure vector search?**
- BM25 rewards exact token matches — critical for improvement flags that must appear verbatim
- Dense vector search rewards semantic similarity — good for conceptual overlap when wording differs
- Neither alone is sufficient; hybrid outperforms both

**Why `alpha=0.45` (slightly keyword-leaning)?**
Improvement flags are highly specific tokens designed to appear verbatim in both query and corpus. BM25 deserves a slight edge over pure semantic similarity for this schema.

**Why `tags^4` and `improvement_flags^4`?**
These fields are the most signal-dense — specifically designed to encode the strategic situation. Giving them 4x BM25 weight means ranking is driven by strategic similarity, not surface text.

**Hard filters vs. soft signals:**

| Filter type | Fields | Rationale |
|---|---|---|
| Hard (must match) | `pot_type`, `street_focus` | Fundamental structural constraint — a 3-bet pot hand and a limped pot hand are not comparable |
| Soft (influences ranking) | texture, stack depth, heads-up | Influence ranking without excluding results on a small corpus |

**Why only two hard filters?** Overfiltering a small corpus produces zero results. Selectivity comes primarily from weighted ranking, not exclusion.

---

### 5. Coaching Generation (`utils/coaching.py`)

**Decision:** Two LLM calls instead of one.

| Call | Purpose |
|---|---|
| Call 1 — enrichment | Normalizes vocabulary for search; structured output schema |
| Call 2 — coaching | Generates advice grounded in retrieved evidence; free-form reasoning |

Collapsing into one call would mean searching before vocabulary is normalized — retrieval would be worse — and the coaching LLM would have no real evidence to ground its advice.

**Similarity policy in the system prompt:**
The coaching prompt defines explicitly when retrieved evidence should be used vs. discarded — based on street match, formation match, board state, SPR bucket, and action line overlap. This prevents the model from blindly applying irrelevant advice.

**In-context learning:**
Retrieved evidence cards are fed as context to the coaching LLM at inference time. The model isn't fine-tuned — it reasons from examples provided in the prompt. This is what enables expert-level specificity without training.

---

### 6. Data Ingestion Pipeline (`scripts/`)

**Decision:** Build a custom OCR pipeline to extract hands from instructional video rather than using synthetic or scraped data.

**Why:** There's no off-the-shelf annotated poker hand dataset. This pipeline processes poker instructional video frame-by-frame using OpenCV, runs OCR on each frame, deduplicates redundant captions, and writes timestamped output used to bootstrap the knowledge base. This gives the corpus real poker theory from expert coaches, not hallucinated or synthetic data.

---

### 7. Embeddings (`utils/embedding.py`)

**Decision:** Local `BAAI/bge-small-en-v1.5` (sentence-transformers) rather than an API-hosted model.

**Why local?**
- Consistency — the same model must embed both corpus entries at ingest time and user queries at search time. Different models (or version changes) would put vectors in different spaces, breaking similarity scores.
- No API latency or cost per query.
- 384-dimensional vectors are compact and fast for nearest-neighbor search.

---

## Key Design Tradeoffs

### Schema coupling
The current schema is tightly coupled to fully-played annotated hands. Supporting isolated spot analyses or text-form coaching tips would require a more flexible schema and would dramatically expand the knowledge base. This was a conscious scope decision.

### Corpus size vs. filter strategy
With a small corpus, hard filters are a liability — they can produce zero results. The design compensates by pushing selectivity into the ranking layer (tag-weighted BM25 + dense vector) rather than the filter layer.

### Hallucination mitigation
LLMs misread cards and gave advice for hands that didn't exist when tested without RAG. The enrichment + retrieval pipeline grounds the coaching LLM in real annotated evidence before generation. The similarity policy in the coaching prompt further constrains when retrieved evidence applies.

---

## What I'd Do Differently

- **Format-agnostic schema from day one.** The current design couples tightly to fully-played hands. Supporting spot analyses or text-form tips would expand the corpus significantly.
- **GTO solver integration.** Currently can't distinguish between exploitative adjustments and theory-optimal baselines. A solver would let the system anchor advice on GTO first, then layer exploitative adjustments.
- **Evaluation harness.** No automated ground truth exists in poker — even GTO has limits in live cash. A proper eval would combine solver-verified spots with expert annotation. Currently tested via manually verified corpus-mirroring hands.
