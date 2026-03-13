# Poker Coach AI — Interview Reference

## The Problem
Getting quality poker coaching is expensive and slow. An LLM alone gives generic, sometimes hallucinated advice. The goal was to build a system that retrieves structurally similar hands from a curated knowledge base and uses them as grounded evidence for coaching — so the LLM synthesizes real poker theory rather than improvising.

---

## Architecture (Top to Bottom)

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

## Components & Design Choices

### 1. Hand State Machine (`utils/preflop_input.py`)
**What it does:** Walks the user through the hand street by street, tracking betting action, who folded, pot type, and positions.

**Why:** Raw free-text hand descriptions are inconsistent. A state machine enforces a structured, standardized input format that produces reliable schema fields downstream. Without this, the schema population would be unpredictable.

**Key output:** `PreflopRecord` — derives `pot_type` (limped / single-raised / 3-bet / ...) and a compact action string like `UTG:open | CO:call | BTN:3bet`.

---

### 2. Schema Builder (`utils/rag_builder.py`)
**What it does:** Takes the collected street records and converts them into a structured JSON object with facets (pot type, board texture, positions, stack depth), a compact action line, and an embedding text string.

**Why:** The vector search needs a consistent representation of each hand. Raw action strings aren't searchable in a meaningful way — the schema extracts the structural features that actually matter for poker (board texture, pot type, position) and discards noise.

**Key design — board texture analysis (`_analyze_runout`):**
Classifies the flop and full runout into structured features:
- `flush_level` — how many cards share a suit (1/2/3)
- `paired_level` — whether the board has a pair/trips
- `straightness` — coarse connectedness score (0/1/2)
- `texture_class` — dry / paired / monotone / draw-heavy

These become filterable and searchable facets in Weaviate.

**Key design — hard filters vs soft signals:**
- Hard filters: only `pot_type` (must match exactly)
- Soft signals: texture, stack depth, heads-up — influence ranking, not required

**Why only one hard filter?** Overfiltering a small corpus produces zero results. Most selectivity comes from the weighted search ranking, not exclusion.

---

### 3. LLM Enrichment (`utils/rag_pipeline.py::enrich_user_hand_with_llm`)
**What it does:** Takes the raw hand schema and calls GPT with a structured output schema (`PokerRAGEntry`) to produce corpus-aligned tags, improvement flags, and a normalized embedding text.

**Why this step exists — the vocabulary mismatch problem:**
A user typing their hand ("I was on the button in a 3-bet pot") and a corpus entry labeled by someone else ("BTN:call | CO:3bet, IP, 3-bet") use different words for the same concept. Without normalization, the vector search can't bridge that gap — similar hands won't rank near each other.

**How it's constrained:** The LLM is given a JSON Schema with strict enums — it can only output values from predefined lists (pot types, positions, texture classes, improvement flags). This is called **structured output** or **constrained generation**. It prevents the model from inventing new vocabulary that won't match the corpus.

**The 50+ improvement flags** span 7 categories:
- A — Board/texture nuance (wet_board, dynamic_board, monotone_pressure...)
- B — Runout events (turn_completes_flush, scare_card_river, brick_river...)
- C — Range/position advantage (hero_IP, range_advantage_hero, blind_vs_blind...)
- D — Hand state & blockers (bluff_catcher_river, flush_blocker_pressure...)
- E — Line & sizing semantics (overbet_polar_node, polar_turn_barrel, thin_value_ok...)
- F — Opponent tendencies (pool_low_bluff, sticky_pool, nit_villain...)
- G — Pricing & equity (fold_equity_low, deny_equity_priority...)

---

### 4. Hybrid Search (`utils/rag_pipeline.py::search_weaviate_hybrid`)
**What it does:** Searches the Weaviate corpus using a combination of BM25 (keyword) and dense vector (semantic) search.

**Why hybrid instead of just vector search?**
- BM25 rewards exact token matches — great for tags and improvement flags that must appear verbatim
- Dense vector search rewards semantic similarity — great for conceptual overlap when wording differs
- Combining them (`alpha=0.45`, slightly keyword-leaning) captures both

**Why tag-weighted BM25 (`tags^4`, `improvement_flags^4`)?**
Tags and improvement flags are the most signal-dense fields — they're specifically designed to encode the strategic situation. Giving them 4x weight means the ranking is driven by strategic similarity, not surface text.

**Why `alpha=0.45`?**
Slightly below 0.5 means the search leans marginally toward keyword matching. Given that the improvement flags are highly specific tokens that should appear verbatim in both query and corpus, BM25 deserves a slight edge.

---

### 5. Coaching Generation (`utils/coaching.py`)
**What it does:** Takes the top retrieved evidence cards and the user's hand summary, then calls GPT to produce street-by-street coaching.

**Why two LLM calls instead of one?**
- Call 1 (enrichment): normalizes vocabulary for search
- Call 2 (coaching): generates advice grounded in retrieved evidence

Collapsing into one call would mean searching before the vocabulary is normalized — retrieval would be worse — and the model would have no real evidence to ground its advice.

**The similarity policy in the system prompt:**
The coaching prompt defines explicitly when retrieved evidence should be used vs. discarded — based on street, formation, board state, SPR bucket, and action line overlap. This is what prevents the model from blindly applying irrelevant advice.

**In-context learning:**
Feeding the retrieved evidence cards into the LLM at inference time is called in-context learning. The model isn't fine-tuned — it's given examples and grounding at query time. This is what enables expert-level specificity without training.

---

### 6. Data Ingestion Pipeline (`scripts/`)
**What it does:** Processes poker instructional videos frame-by-frame using OpenCV, runs OCR on each frame, deduplicates redundant captions, and writes timestamped output used to bootstrap the knowledge base.

**Why:** There's no off-the-shelf annotated poker hand dataset. This pipeline extracts coaching content from publicly available instructional video, giving the knowledge base real poker theory rather than synthetic data.

---

## Key Concepts

### RAG (Retrieval-Augmented Generation)
Grounding LLM output in retrieved documents to reduce hallucination and improve specificity. Instead of asking the LLM to generate advice from scratch, you retrieve relevant evidence first and feed it as context. The model synthesizes rather than invents.

### Embeddings
A way of converting text into a fixed-length vector of numbers such that semantically similar text ends up with similar vectors. The model has learned from large amounts of data how to map meaning into numerical space.

- Why local (`BAAI/bge-small-en-v1.5`): consistency — the same model must embed both corpus entries at ingest time and user queries at search time. Different models (or version changes) would put vectors in different spaces, breaking similarity scores. Also: no API latency/cost per query.
- Output: a 384-dimensional vector of floats stored in Weaviate alongside the document.

### Hybrid Search
Combines BM25 (keyword/term-frequency matching) and dense vector search (semantic similarity). BM25 catches exact token matches; dense catches conceptual overlap when wording differs. Neither alone is sufficient — hybrid outperforms both.

### Structured Output / Constrained Generation
Forcing an LLM to output only values from a predefined schema (JSON Schema with enums). Used in the enrichment call to ensure the normalized hand uses corpus-aligned vocabulary and nothing else.

### In-Context Learning
Providing examples or grounding information to an LLM at inference time (in the prompt/context window) rather than through fine-tuning. The retrieved evidence cards are fed as context to the coaching LLM — the model reasons from them without any weight updates.

### Vector Database (Weaviate)
A database that stores vectors alongside metadata and supports nearest-neighbor search. Unlike a regular database that matches rows by exact field values, a vector DB finds the closest vectors by similarity score. Weaviate also supports hybrid search (BM25 + vector) natively.

---

## Interview Q&A Reference

**Hardest part:**
Getting retrieval to work — most poker hands share the same surface structure with different nuances. Had to design a schema capturing key structural features, solve a vocabulary mismatch problem between user input and corpus labels (solved with LLM enrichment), and calibrate hard filters vs. soft ranking to avoid zero-result cases on a small corpus.

**What I'd do differently:**
Design the schema to be format-agnostic from day one — the current design is tightly coupled to fully-played annotated hands. Supporting isolated spot analyses or text-form coaching tips would dramatically expand the knowledge base. Would also integrate a GTO solver to distinguish between exploitative adjustments and theory-optimal baselines.

**How I knew it was working:**
Two layers — first, test hands that closely mirrored corpus entries (verifiable: did retrieval pull the right hands and produce advice consistent with annotations?). Second, expert judgment on realistic hands. Honest limitation: no true ground truth in poker — even a GTO solver has limits in live cash because it doesn't account for opponent tendencies.

**Why not just prompt an LLM directly:**
Two problems. First, hallucinations — models misread cards and gave advice for a hand that didn't exist. Second, even when cards were correct, advice was surface-level — not useful for experienced players. LLMs don't have deep poker theory baked in at expert level. Grounding output with real annotated coaching evidence is what enables the depth and specificity.
