# searchmem — Build Specification (Gemini Edition)

## Project Pitch

An LLM iteratively improves a search scorer over 6 sessions under hard metrics (NDCG@10 target, p95 latency ceiling). A deterministic concept graph retrieves the right context. A session log records what was tried and what failed. When the LLM plans the next experiment, it sees its own history — so it skips what already failed without being told to. A memoryless baseline with the same LLM runs the same task and misses the target because it retries a failed approach.

**The one-line pitch for judges:**
"LLMs forget. We don't let them. One skipped experiment. One target hit."

**Hard rule:** The concept graph is deterministic. The LLM only plans experiments. No LLM is used for concept naming, retrieval, or evaluation.

---

## 1. Success Criteria

1. Run end-to-end with `python run.py`
2. Complete two runs: **memoryful** and **memoryless**
3. Produce `results/comparison.md` showing memoryful ≥ 0.740 and memoryless < 0.740
4. Produce `results/learning_curve.png` plotting NDCG per session for both runs
5. Dump the final concept graph to `results/graph.json`
6. Print the LLM's reasoning for every session to stdout (for the live demo)

Runtime for `python run.py` must be < 5 minutes (excluding LLM latency).

---

## 2. Directory Structure

```
searchmem/
├── run.py
├── requirements.txt
├── SPEC.md
├── README.md
├── .env                      # NOT committed — contains GEMINI_API_KEY
├── .env.example              # committed — shows required variable names
├── .gitignore
├── .cursor/
│   └── rules/project.mdc
├── .cursorignore
├── data/
│   ├── corpus.py
│   ├── queries.py
│   ├── scorer.py
│   └── golden.py
├── core/
│   ├── parser.py
│   ├── tokenizer.py
│   ├── graph.py
│   ├── cluster.py
│   ├── concepts.py
│   ├── memory.py
│   ├── retriever.py
│   └── reinforcement.py
├── agent/
│   ├── llm_planner.py
│   ├── strategy.py
│   ├── session.py
│   └── loop.py
├── eval/
│   ├── ndcg.py
│   ├── latency.py
│   └── metrics.py
├── baseline/
│   └── memoryless.py
├── results/
│   └── memory/
└── tests/
    ├── test_graph.py
    ├── test_concepts.py
    ├── test_metrics.py
    └── test_memory.py
```

---

## 3. The Toy Example

### 3.1 `data/scorer.py`

```python
import math


class Scorer:
    def score_keyword(self, query_terms, doc):
        """BM25 term overlap. Baseline: plain term frequency."""
        score = 0.0
        for term in query_terms:
            if term in doc.terms:
                score += 1.0
        return score

    def score_freshness(self, doc):
        """Exponential decay. Baseline: 30-day constant."""
        return math.exp(-doc.days_since_publish / 30.0)

    def score_document(self, query_terms, doc):
        """Combination. Baseline: 0.7 * keyword + 0.3 * freshness."""
        return 0.7 * self.score_keyword(query_terms, doc) \
             + 0.3 * self.score_freshness(doc)
```

### 3.2 Synthetic Data

`data/corpus.py` — Generate 500 synthetic docs with fields:
- `id` (str)
- `title` (str)
- `body` (str)
- `terms` (list[str]) — tokenized body + title
- `days_since_publish` (int, 0–365)
- `tags` (list[str])

Include **synonym structure**: "car", "automobile", "vehicle" appear in overlapping doc sets. This makes synonym expansion a viable-but-costly idea.

`data/queries.py` — Generate 200 queries with fields:
- `terms` (list[str])
- `relevant_doc_ids` (dict[str, int] — graded relevance 0–3)
- `query_type` (str: "exact" | "synonym" | "freshness")

Calibrate relevance labels so the **baseline NDCG@10 lands near 0.700**. This is the single most important calibration step — everything downstream depends on it.

### 3.3 Golden Set

`data/golden.py` — Provides:
- `load()` → returns the 200 queries with relevance labels
- `ranking_diff(baseline_rankings, current_rankings)` → number of position changes in top-10 per query, summed across queries

The golden set is fixed at generation time and never modified during the run.

---

## 4. Core Modules (Deterministic — No LLM)

### 4.1 `core/parser.py`

```python
from dataclasses import dataclass


@dataclass
class Method:
    fqn: str               # "Scorer.score_keyword"
    name: str              # "score_keyword"
    file: str              # "data/scorer.py"
    lines: tuple[int, int] # (start, end)
    docstring: str
    source: str            # full source of the method
    calls: list[str]       # FQNs of methods called within
    tokens: list[str]      # identifiers + params + locals + called names
```

Use Python's stdlib `ast` module. No tree-sitter. Walk the AST, find `ClassDef` and `FunctionDef` nodes, extract the required fields.

Provide `parse_file(path) -> list[Method]`.

### 4.2 `core/tokenizer.py`

Provide `tokenize_method(method_ast) -> list[str]`:
- Split method names on `_` and camelCase
- Include: method name tokens, parameter names, local variable names, called function names, string literals of length > 3
- Stopword filter: remove Python keywords, remove tokens that appear in > 30% of all methods in the file, remove single-character tokens

### 4.3 `core/graph.py`

Build two NetworkX graphs:

**Token graph (`G_tok`)**: nodes are tokens. Edges weighted by co-occurrence across methods. Two tokens get an edge if they appear in the same method. Weight = count of methods where both appear.

**CALLS graph (`G_calls`)**: nodes are method FQNs. Edges are `CALLS`.

Provide:
```python
def build_token_graph(methods: list[Method]) -> nx.Graph
def build_calls_graph(methods: list[Method]) -> nx.DiGraph
```

### 4.4 `core/cluster.py`

- Augment `G_calls` with token-similarity edges from `G_tok` (weight < 0.5 of CALLS edges, so CALLS dominates)
- Run `networkx.algorithms.community.louvain_communities` with `seed=42`
- Return: `dict[str, list[str]]` mapping cluster_id → method FQNs

For the toy Scorer, expect **exactly 3 clusters**:
- `{Scorer.score_keyword}`
- `{Scorer.score_freshness}`
- `{Scorer.score_document}`

Write `tests/test_graph.py` asserting exactly 3 clusters are returned.

### 4.5 `core/concepts.py`

```python
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Concept:
    id: str                    # "keyword_matching"
    name: str                  # TF-IDF-derived, e.g. "keyword_matching"
    member_methods: list[str]  # FQNs
    embedding: np.ndarray      # mean of member method token vectors
    metadata: dict = field(default_factory=dict)
```

Naming rule: for each cluster, pick the top-2 tokens by `tf_in_cluster * log(N_methods / df_token)`, join with `_`.

Embedding: mean of one-hot token vectors over the cluster's methods.

Provide:
```python
def build_concepts(methods: list[Method], clusters: dict) -> dict[str, Concept]
```

Expected concept names for the toy example:
- `keyword_matching`
- `freshness_decay`
- `score_combination`

Write `tests/test_concepts.py` asserting these three names appear.

### 4.6 `core/memory.py`

Three JSON-backed stores in `results/memory/`:

**`sessions.json`** — list of session records:
```json
[
  {
    "session_id": 1,
    "concepts_probed": ["keyword_matching", "freshness_decay", "score_combination"],
    "decision": "characterization: keyword_matching dominates",
    "outcome": {"ndcg": 0.700, "latency": 84.0, "sessions_used": 1}
  }
]
```

**`experiments.json`** — list of experiment records:
```json
[
  {
    "session_id": 2,
    "concept": "keyword_matching",
    "hypothesis": "synonym expansion improves recall",
    "action_id": "synonym_expansion",
    "result": {"ndcg_delta": 0.015, "latency_delta": 26.0},
    "verdict": "revert",
    "insight": "synonym expansion is NDCG-effective but latency-expensive",
    "generalizes_to": ["query_time_expansion"]
  }
]
```

**`concepts.json`** — concept metadata map:
```json
{
  "keyword_matching": {
    "headroom": 0.288,
    "sub_signals": [
      {"approach": "synonym_expansion", "cost": "high", "status": "rejected"},
      {"approach": "stemming", "cost": "low", "status": "kept"},
      {"approach": "n_gram_matching", "cost": "high", "status": "skipped"}
    ]
  },
  "query_time_expansion": {
    "created_by": "session_2",
    "reason": "cluster of approaches sharing high-latency cost profile"
  }
}
```

Provide a `Memory` class with methods:
- `log_session(record)`
- `log_experiment(record)`
- `log_skip(concept_id, action_id, reason)`
- `get_experiments_for_concept(concept_id)`
- `get_rejected_cost_profiles() -> set[str]`
- `experiments` property (returns list)

Provide a `NoOpMemory` class with the same interface but all methods are no-ops. `experiments` MUST return `[]`. `get_experiments_for_concept` and `get_rejected_cost_profiles` always return empty results. This is used by the memoryless baseline, and it's what makes the demo work — the LLM sees no ledger.

Write `tests/test_memory.py` with a round-trip test: log 3 experiments, save, reload, verify contents match.

### 4.7 `core/retriever.py`

```python
def retrieve(query: str, concepts: dict, memory, k: int = 3) -> list[Concept]:
    """
    Concept-first retrieval.
    1. Tokenize query into bag of tokens.
    2. Score each concept by cosine similarity of query vector to concept embedding.
    3. Return top-k concepts.
    """
```

Also provide:
```python
def get_session_memories(concepts: list[Concept], memory) -> list[dict]:
    """Return all session records linked to any of the given concepts."""
```

### 4.8 `core/reinforcement.py`

```python
def update_edges(experiment: dict, concepts: dict, edges: dict):
    """
    Reinforce concept-method edges based on outcome.
    - On "keep": +0.1 to the concept-method edge
    - On "revert": no edge change (concept is still correct)
    - If experiment has "generalizes_to": create a new concept node named
      by the first entry in generalizes_to, link the failed action_id to it
      with type "INSTANCE_OF", and add it to concepts.
    """
```

`edges` is a dict of `{(method_fqn, concept_id): float}`.

---

## 5. Agent Modules

### 5.1 `agent/strategy.py`

Hardcoded candidate action catalog. Each action has a `code_patch` function that takes the current scorer source as a string and returns a patched source.

```python
CANDIDATE_ACTIONS = {
    "keyword_matching": [
        {"id": "synonym_expansion", "hypothesis": "synonym expansion improves recall",
         "cost": "high", "expected_ndcg_gain": 0.015,
         "code_patch": patch_synonym,
         "generalizes_to": ["query_time_expansion"]},
        {"id": "stemming", "hypothesis": "stemming normalizes morphological variants",
         "cost": "low", "expected_ndcg_gain": 0.008,
         "code_patch": patch_stemming},
        {"id": "n_gram_matching", "hypothesis": "phrase matching via n-grams",
         "cost": "high", "expected_ndcg_gain": 0.014,
         "code_patch": patch_ngram,
         "generalizes_to": ["query_time_expansion"]},
        {"id": "field_weights", "hypothesis": "title matches matter more than body",
         "cost": "low", "expected_ndcg_gain": 0.012,
         "code_patch": patch_field_weights},
    ],
    "freshness_decay": [
        {"id": "sharpen_decay", "hypothesis": "sharper decay suits news corpus",
         "cost": "low", "expected_ndcg_gain": 0.009,
         "code_patch": patch_decay},
    ],
    "score_combination": [
        {"id": "tune_weights", "hypothesis": "BM25F changed score variance; weights need recalibration",
         "cost": "low", "expected_ndcg_gain": 0.012,
         "code_patch": patch_weights},
    ],
}
```

Provide `get_candidates(concept_id: str) -> list[dict]` — returns the unfiltered catalog for the concept. No filtering here; the LLM does the filtering.

The `patch_*` functions are string replacements on the scorer source. Each patch is a small, self-contained change. Example:

```python
def patch_stemming(src: str) -> str:
    return src.replace(
        "score += 1.0",
        "score += 1.0 if term in doc.terms else (0.5 if _stem(term) in doc.stems else 0.0)"
    )
```

Patches can assume fixed anchor strings in the original source. Do not make them robust to arbitrary input — they are only applied to the known baseline.

### 5.2 `agent/llm_planner.py` (the only LLM call — uses Gemini)

```python
import json
import os
from openai import OpenAI

# Gemini exposes an OpenAI-compatible endpoint. We use the standard
# `openai` Python client and point it at Google's base URL.
#
# Environment variables (set in .env):
#   GEMINI_API_KEY   — your key from Google AI Studio
#   GEMINI_MODEL     — optional; defaults to "gemini-2.5-flash"

_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def plan_experiment(concepts, memory, candidate_actions, goal) -> dict:
    """
    LLM-driven experiment selection. The ONLY LLM call in the system.

    Input:
      - concepts: top-k retrieved concepts
      - memory: Memory or NoOpMemory instance
      - candidate_actions: hardcoded catalog for these concepts
      - goal: {"target_ndcg": 0.740, "ceiling_ms": 100, "budget": 6}

    Output:
      {
        "chosen_action_id": str,
        "reasoning": str,
        "skipped_actions": [{"id": str, "reason": str}]
      }
    """
    prompt = f"""You are improving a search scorer.

GOAL: Raise NDCG@10 from 0.700 to >= 0.740 within 6 sessions.
      Latency ceiling: 100ms p95.

CURRENT CONCEPTS: {[c.name for c in concepts]}

CANDIDATE ACTIONS:
{_format_actions(candidate_actions)}

PRIOR EXPERIMENTS (chronological):
{_format_ledger(memory.experiments)}

Pick the next action. If an action shares a cost profile with a
previously rejected experiment, skip it and explain why.
Return JSON only:
{{
  "chosen_action_id": "...",
  "reasoning": "...",
  "skipped_actions": [{{"id": "...", "reason": "..."}}]
}}
"""
    response = _client.chat.completions.create(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _format_actions(actions):
    lines = []
    for a in actions:
        lines.append(f"- {a['id']} (cost={a['cost']}, "
                     f"expected_gain={a['expected_ndcg_gain']})")
    return "\n".join(lines)


def _format_ledger(experiments):
    if not experiments:
        return "(none)"
    lines = []
    for e in experiments:
        lines.append(
            f"[S{e['session_id']}] {e['action_id']} -> "
            f"{e['verdict'].upper()} (ndcg_delta={e['result']['ndcg_delta']:+.3f}, "
            f"latency_delta={e['result']['latency_delta']:+.1f}ms)"
        )
    return "\n".join(lines)
```

**Why this works with Gemini:** Google provides an OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/`. You use the standard `openai` Python client, change only the `base_url` and the API key, and the rest of the code is identical. The model name for the free tier is `gemini-2.5-flash`.

**Free tier note:** The Gemini free tier allows roughly 10–15 requests per minute for Flash-class models. Our system makes 6 LLM calls per run, 2 runs per demo — 12 calls total. Well within limits.

### 5.3 `agent/session.py`

```python
def run_session(session_id: int, state) -> state:
    # 1. Retrieve relevant concepts (deterministic)
    concepts = retriever.retrieve(state.goal, state.concepts, state.memory)

    # 2. Get candidate actions (deterministic, unfiltered)
    candidates = []
    for c in concepts:
        candidates.extend(strategy.get_candidates(c.id))

    if not candidates:
        return state

    # 3. LLM plans the experiment (the only LLM call)
    plan = llm_planner.plan_experiment(
        concepts=concepts,
        memory=state.memory,
        candidate_actions=candidates,
        goal=state.goal,
    )

    # 4. Print for the live demo
    print(f"\n=== SESSION {session_id} ===")
    print(f"Concepts: {[c.name for c in concepts]}")
    print(f"LLM reasoning: {plan['reasoning']}")
    for skip in plan.get("skipped_actions", []):
        print(f"  SKIPPED: {skip['id']} — {skip['reason']}")

    # 5. Log skips to memory
    for skip in plan.get("skipped_actions", []):
        state.memory.log_skip(skip["id"], skip["reason"])

    # 6. Find chosen action
    action = next(
        a for a in candidates if a["id"] == plan["chosen_action_id"]
    )

    # 7. Apply code patch
    patched_src = action["code_patch"](state.scorer_source)

    # 8. Evaluate (deterministic)
    ndcg = eval.ndcg.run(patched_src, state.golden)
    latency = eval.latency.run(patched_src)

    # 9. Verdict
    within_budget = latency <= 100.0
    improved = ndcg > state.current_ndcg
    verdict = "keep" if (improved and within_budget) else "revert"

    # 10. Update state
    old_ndcg = state.current_ndcg
    if verdict == "keep":
        state.scorer_source = patched_src
        state.current_ndcg = ndcg

    print(f"Action: {action['id']}")
    print(f"Result: ndcg {ndcg:.3f} ({ndcg - old_ndcg:+.3f}), "
          f"latency {latency:.1f}ms -> {verdict.upper()}")

    # 11. Log experiment
    state.memory.log_experiment({
        "session_id": session_id,
        "concept": action.get("concept", concepts[0].id),
        "hypothesis": action.get("hypothesis", ""),
        "action_id": action["id"],
        "result": {"ndcg_delta": ndcg - old_ndcg,
                   "latency_delta": latency - state.baseline_latency},
        "verdict": verdict,
        "insight": action.get("insight", ""),
        "generalizes_to": action.get("generalizes_to", []),
    })

    # 12. Reinforcement
    reinforcement.update_edges(
        state.memory.experiments[-1], state.concepts, state.edges
    )

    return state
```

### 5.4 `agent/loop.py`

```python
def run_memoryful() -> "RunResult":
    state = initialize(memory=Memory())
    for session in range(1, 7):
        state = run_session(session, state)
        state.record(session)
        if state.current_ndcg >= 0.740:
            break
    return state


def run_memoryless() -> "RunResult":
    state = initialize(memory=NoOpMemory())
    for session in range(1, 7):
        state = run_session(session, state)
        state.record(session)
    return state
```

`initialize()` sets up:
- `goal`: "Raise NDCG@10 from 0.700 to ≥0.740 within 6 sessions"
- `scorer_source`: initial content of `data/scorer.py`
- `concepts`: built from parse → cluster → concepts pipeline
- `edges`: initial concept-method edge weights
- `golden`: loaded golden set
- `baseline_ndcg`: 0.700
- `baseline_latency`: 84.0
- `current_ndcg`: 0.700
- `memory`: Memory or NoOpMemory
- `history`: list of per-session records

`RunResult` is a dataclass with a `summary()` method returning a dict.

Same Gemini model. Same prompt. Different memory. The `NoOpMemory.experiments` returns `[]`, so the LLM sees no ledger and has nothing to reason over.

---

## 6. Evaluation Modules

### 6.1 `eval/ndcg.py`

Standard NDCG@10:
```
DCG@10  = sum((2^rel_i - 1) / log2(i + 1) for i in 1..10)
NDCG@10 = DCG@10 / IDCG@10
```

Average over all 200 queries.

```python
def run(scorer_source: str, golden: list) -> float
```

exec the scorer source in a namespace, instantiate `Scorer`, evaluate each query. Return mean NDCG@10.

### 6.2 `eval/latency.py`

Time each query's `score_document` call, compute p95 across all 200 queries. Report in milliseconds.

```python
def run(scorer_source: str) -> float
```

Run each query 10 times, take median per query, then compute p95.

### 6.3 `eval/metrics.py`

Aggregate: `ndcg@10`, `p95_latency`, `sessions_used`, `experiments_run`, `experiments_kept`, `experiments_reverted`, `experiments_skipped`.

```python
def aggregate(run_result) -> dict
```

---

## 7. Run Script

### `run.py`

```python
def main():
    from dotenv import load_dotenv
    load_dotenv()  # loads GEMINI_API_KEY and GEMINI_MODEL from .env

    data.ensure_generated()
    golden = data.golden.load()

    print("Running memoryful agent...")
    mem_result = agent.loop.run_memoryful()

    print("\nRunning memoryless baseline...")
    base_result = agent.loop.run_memoryless()

    comparison = {
        "memoryful": mem_result.summary(),
        "memoryless": base_result.summary(),
    }

    write_json("results/comparison.json", comparison)
    write_markdown("results/comparison.md", comparison)
    plot_learning_curve("results/learning_curve.png", mem_result, base_result)
    write_json("results/graph.json", mem_result.concepts)
    write_json("results/memory/sessions.json", mem_result.memory.sessions)
    write_json("results/memory/experiments.json", mem_result.memory.experiments)


if __name__ == "__main__":
    main()
```

---

## 8. Expected Output

### Terminal (live demo)

```
=== SESSION 3 ===
Concepts: [keyword_matching, freshness_decay]
LLM reasoning: "Stemming is low-cost normalization. N-gram matching shares
the high-latency term-expansion profile with synonym_expansion, which was
rejected in S2 for exceeding the latency ceiling. Skipping n_gram_matching.
Selecting stemming."
  SKIPPED: n_gram_matching — shares cost profile with rejected synonym_expansion
Action: stemming
Result: ndcg 0.708 (+0.008), latency 87.0ms -> KEEP
```

### `results/comparison.md`

```
| Metric               | Memoryful | Memoryless |
|----------------------|-----------|------------|
| Final NDCG@10        | 0.741     | 0.729      |
| Target (>=0.740)     | HIT       | MISS       |
| Final p95 latency    | 94ms      | 96ms       |
| Sessions used        | 6         | 6          |
| Experiments kept     | 4         | 4          |
| Experiments reverted | 1         | 2          |
| Experiments skipped  | 1         | 0          |
```

### `results/graph.json`

Contains 4 concept nodes (3 initial + `query_time_expansion` from generalization) and edges linking `synonym_expansion` to `query_time_expansion` via `INSTANCE_OF`.

---

## 9. Guardrails

- The concept graph is deterministic. No LLM in parse, cluster, name, or retrieve.
- The LLM is called exactly once per session, in `llm_planner.plan_experiment`.
- `temperature=0` on every LLM call.
- The LLM can only choose from `candidate_actions`. It cannot invent new ones.
- Use `gemini-3.8-flash` as the default model (free tier compatible).
- Cache LLM responses in `results/llm_cache/` so the demo replays if the API is down.
- `--mock-llm` flag in `run.py` uses canned responses that produce the same memoryful/memoryless divergence.
- Keep total code under 1,800 lines excluding tests.
- Dependencies: `numpy`, `scikit-learn`, `networkx`, `matplotlib`, `pytest`, `openai`, `python-dotenv`.

---

## 10. Definition of Done

1. `python run.py` completes in < 5 minutes.
2. `results/comparison.md` shows memoryful ≥ 0.740 and memoryless < 0.740.
3. Terminal output shows the LLM's reasoning for every session.
4. `results/learning_curve.png` shows the divergence at Session 3.
5. `results/graph.json` contains 4 concept nodes.
6. `results/memory/experiments.json` contains the Session 2 revert record with `generalizes_to: ["query_time_expansion"]`.
7. `pytest tests/` passes.