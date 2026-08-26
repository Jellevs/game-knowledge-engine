# RS3 PvME RAG — Project Checklist

Status: [x] done · [~] partial · [ ] missing
Detail and results live in `ENGINEERING_LOG.md`.

---

## Headline result

| config | mrr@10 | hit@1 | hit@3 |
|---|---|---|---|
| raw markup, dense only | 0.493 | 0.371 | 0.557 |
| + preprocessing | 0.661 | 0.557 | 0.729 |
| + hybrid search (BM25 + RRF) | 0.763 | 0.657 | 0.857 |
| + enrage band tokens | **0.793** | **0.700** | **0.886** |

+0.300 MRR vs baseline, t=6.04, **p<0.0001**, n=70.

---

## 1. Foundations — DONE
- [x] Typed config (pydantic-settings), no hardcoded paths, env-overridable
- [x] Package layout, CLI with paired `--x/--no-x` flags, runs from project root
- [x] `.gitignore`, no committed artifacts
- [x] ruff + pytest, 40+ tests over the real corpus

## 2. Corpus — DONE
- [x] 3 guides (necromancy, melee/magic, melee/ranged)
- [x] Preprocessing: 30-39% Discord markup removed
- [x] All 116 emoji shortcodes mapped; 32 rendered as "Full Name (code)"
- [x] JSON embed parser (recovered content that was locked in raw JSON)
- [x] Video link extraction + enrichment ("Example kill video")
- [x] Duplicate-name collapsing
- [x] Enrage band tokens for BM25
- [x] `source` / `style` / `boss` metadata per chunk

## 3. Retrieval — DONE
- [x] Dense (all-MiniLM-L6-v2) + Qdrant
- [x] Hybrid: BM25 sparse with server-side IDF, fused with RRF
- [x] `HybridRetriever` wired into the app, not just the eval
- [ ] Cross-encoder reranker — deferred; targets `sequence` (0.557)
- [ ] Metadata filtering on `style` (stored, unused)

## 4. Retrieval evaluation — DONE
- [x] 72-question golden set, tagged by archetype
- [x] Relevance = source + section + phrase, phrases verified in every corpus mode
- [x] hit@k + MRR via ranx
- [x] Paired t-tests built into the harness
- [x] Per-tag diagnostic breakdown
- [x] Collection registry + embedding-model consistency guard
- [x] MLflow logging with build-time params

## 5. Generation evaluation — NOT STARTED
**The big remaining gap. Nothing about answer quality is measured.**
- [ ] Rebuild abstention slice (~8 genuinely unanswerable questions) ← BLOCKED ON JELLE
- [ ] RAGAS: faithfulness, answer relevancy, context precision
- [ ] Abstention / refusal rate
- [ ] Citations in answers
- [ ] Sequence-accuracy metric for rotations (order matters)

## 6. Serving — NOT STARTED
- [ ] FastAPI: `POST /query`, `GET /health`, SSE streaming
- [ ] Streamlit or Gradio UI + demo GIF
- [ ] Qdrant in server mode (embedded holds an exclusive lock)
- [ ] Plan the Ollama swap — llama3.2 is unusable on a small CPU container

## 7. Data & ops — NOT STARTED
- [ ] Postgres: query logs, feedback, eval history (SQLAlchemy + Alembic)
- [ ] Tracing (Langfuse), p50/p95/p99 latency, token accounting
- [ ] Semantic cache (needs the eval harness to tune the threshold safely)
- [ ] Dockerfile + compose
- [ ] CI: lint → test → **eval as a ship gate**
- [ ] Deploy (AWS ECS Fargate or Azure Container Apps)

## 8. Presentation — NOT STARTED
- [ ] README: problem, demo GIF, architecture diagram, results table
- [ ] Writeup on what the eval numbers taught (the wrong predictions are the good part)
- [ ] Meaningful commit history and PRs

---

## Blocked on game knowledge
- [ ] Confirm `cane` mapping (guessed "Cleave")
- [ ] Confirm `ezk` mapping (guessed "Igneous Kal-Zuk")
- [ ] Verify `anti` → "Anticipation" (vs Anti-fire)
- [ ] ~8 questions genuinely unanswerable by all three guides

## Next
1. Abstention slice (Jelle)
2. RAGAS
3. FastAPI + Streamlit + Docker
4. README
