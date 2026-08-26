# Engineering Log

A running record of what was tried, why, and what the numbers said. Entries are
append-only. Failed experiments stay in — they are the most useful part.

---

## 2026-08-12 — Session 1

### Starting state

Working end-to-end RAG: markdown-header + recursive chunking → all-MiniLM-L6-v2
embeddings → local Qdrant → llama3.2 via Ollama, wired with LangChain LCEL.
Single guide file (Arch-Glacor Necromancy, PvME). MLflow/DagsHub connected but
only a dummy run. No tests, no evaluation, one git commit.

---

## 1. Repository cleanup

**Problem.** Absolute Windows paths (`C:\Users\jelle\...`) hardcoded in
`search.py` and `main.py`; no `.gitignore`; `rag/` used bare imports
(`from utils import ...`) so it only ran with `cwd=rag/`; dead modules
(`embedder.py`, `utils.py`) superseded by `ingest.py`.

**Changes.**

- `.gitignore` covering `.venv`, `qdrant_db/`, `*.pkl`, `.env`, `mlruns/`, caches.
- `rag/config.py` — all paths derived from `Path(__file__).parent.parent`.
- `rag/` became a real package (`__init__.py`, absolute `from rag import ...`),
  declared in `pyproject.toml` via hatchling.
- `main.py` became a CLI (`--ingest`, question positional, `-k`).
- `rag/pipeline.py` extracted from the old `rag/main.py`.
- Deleted `embedder.py`, `utils.py`, `vector_cache.pkl`, stray empty `qdrant_db/`.
- Added ruff + pytest config.

**Why it mattered.** A cloned repo previously crashed on the first hardcoded path.

### Config: dotenv → pydantic-settings

Started with plain `python-dotenv`, switched to `pydantic-settings` (which uses
dotenv internally, so this is a layer, not a replacement).

**Why.** Type coercion and fail-fast validation. `TOP_K=5` arrives as `int`, not
`"5"`. A validator rejects `chunk_overlap >= chunk_size` at startup rather than
letting it fail deep inside the splitter.

**Payoff later.** Every experiment config is one `settings.model_dump()` away
from being MLflow params, and the same code reads a local `.env` or cloud-injected
env vars without change.

---

## 2. Corpus preprocessing

### The finding

Measured the raw guide: **37.6% of characters were Discord emoji markup**
(`<:bloat:1159433682403201044>`), 134 occurrences, 53 unique shortcodes.

A rotation line embedded as:

```
<:invokedeath:1137809121983336548> → <:splitsoul:1137809168368148490> → ...
```

The 19-digit IDs tokenize into meaningless fragments. At `chunk_size=500` only
~310 characters per chunk were actual content. No amount of question-writing
fixes this — it is a corpus problem.

### What was built

- `rag/emoji_map.py` — 53 shortcodes → readable RS3 names.
- `rag/preprocess.py` — expand emoji, flatten `.embed:json` blocks into markdown,
  strip PvME directives (`.tag:`, `.img:`, `.pin:`) and Discord refs, normalize
  bullets, tidy blank lines.
- `preprocess_guide` setting (default `True`) so the whole step is A/B-testable
  without a code edit.

**Order dependency (documented in the code):** emoji before embeds (emoji live
inside the JSON strings), embeds before directive-stripping (which would delete
the `.embed:json` marker the parser navigates by).

### Design decisions

**Kept the JSON embed parser** rather than hand-editing the three embeds.
Rationale: every PvME guide uses this format, so this repeats across future guides.

**Embed parsing approach changed** from reverse brace-matching to "walk upwards
from the marker adding lines until `json.loads` succeeds." Equivalent result,
far easier to read. Safe because any inner block still has trailing braces after
it, which makes it invalid JSON — so the first successful parse is always the
true outer block.

**Enrichment for videos.** The guide's example-kill section never contains the
word "video." Output lines were rewritten as
`- Example kill video, 3000% GM timer (5:25): <url>` so the content is
retrievable by the words a person would actually search with. Not inventing
facts — it genuinely is a video — just adding implied context. This turned out
to be the single clearest win of the session (see §5).

**Link handling.** `[Link](url)` → bare URL, so the model copies rather than
reconstructs markdown. Prompt updated to explicitly permit including URLs
"exactly as they appear," since a strict context-only instruction can otherwise
suppress them.

### Result

11,561 → 8,148 chars (29.5% removed). Chunk count 35 → 22.

---

## 3. Golden dataset

42 questions in `data/golden/arch_glacor.json`: 38 answerable, 4 unanswerable
(abstention slice).

**Question archetypes**, chosen because most of the guide is rotations:

| archetype | why |
|---|---|
| enrage-conditional pairs | same question at <2500% and 3000%; the two correct chunks are near-identical apart from a threshold. Hardest case for dense embeddings. |
| full ordered sequence | opening rotation, arms rotation |
| single-step lookup | "what do I cast first on minions?" |
| ordering constraint | "Spectral Scythe before or after Invoke Death?" |
| numeric threshold | 5 residual souls, 10 freezing blood stacks, ~27k Divert. Unambiguously scoreable, good hallucination detector. |
| why-questions | tests whether retrieval grabs justification, often a different sentence than the instruction |
| unanswerable | melee/magic rotations, 5000% enrage, a different boss — correct behaviour is refusal |

**Ambiguity caught during design.** "Which attack at the beginning of the fight?"
has two defensible answers (Bloat as first *attack*; the full Darkness → Invoke
Death → ... prep sequence). Split into two questions. Ambiguous questions are
unscoreable.

**Relevance labels derived from document structure** (`expected_section` matched
against chunk header metadata) rather than hand-labeling every chunk. Deliberate
cost tradeoff — see §4 for where it broke and §6 for the limit it hit.

---

## 4. Evaluation harness — three bugs, all silent

Built `evals/run_retrieval.py` using `ranx`. Retrieval metrics need **no LLM**,
which is the right tool here: preprocessing is a retrieval change, so measure
retrieval. RAGAS (LLM-judge, slow, noisy) is for the generation phase later.

**Every one of the three bugs produced plausible numbers rather than crashing.**
This is the characteristic failure mode of ML work and is the main lesson of the
session.

### Bug 1 — shared qrels across different corpora

`ranx.compare` assumes all runs search the same corpus with the same document
IDs. The two collections have independently generated UUIDs. Relevance labels
were built once from `arch_glacor_raw` and reused for both runs.

*Symptom:* clean scored exactly 0.000 on every metric — zero ID overlap.

*Fix:* score each collection against its own labels via `ranx.evaluate`. Dropped
cross-run significance testing, which was invalid anyway — a paired test assumes
a shared corpus.

### Bug 2 — CLI overrides after an early return

`--collection` and `--no-preprocess` were applied *after* the `if args.ingest:
... return` block, so they silently did nothing during ingest. Both ingests
wrote preprocessed data into the default collection `arch_glacor`, appending
22 + 22 = 44 chunks.

*Symptom:* script reported success while doing the wrong thing twice. Only
visible by reading the printed collection name.

*Fix:* move overrides directly after `parse_args()`. Delete and re-ingest —
`from_documents` appends, it does not replace.

### Bug 3 — relevance labels far too coarse

`MarkdownHeaderTextSplitter` carries parent headers forward. The guide has
`## __Presets and Relics__` and then no further `##` until `## __Example Kills__`,
so **every `###` chunk inherits "Presets and Relics" as its H2**. Joining all
headers meant the label `"Presets"` matched **29 of 35 chunks** in raw and 18 of
22 in clean.

*Symptom:* `hit@10 = 1.000`, `hit@3 = 0.895`. Would have gone into the README as
a great result. It was measuring nothing — retrieving almost anything counted.

*Fix:* match only the deepest header present.

*Relevant-set sizes after the fix:*

| section | raw | clean |
|---|---|---|
| Exposed Core | 9 | 5 |
| General Rotations | 8 | 6 |
| Clearing Glacyte Minions | 5 | 3 |
| Defensive | 4 | 3 |
| Introduction | 3 | 3 |
| Presets | 3 | 1 |
| Example Kills | 2 | 1 |

---

## 5. Result: preprocessing A/B

Two collections, same DB, built from the same source file:

| | chunks |
|---|---|
| `arch_glacor_raw` | 35 |
| `arch_glacor_clean` | 22 |

### Aggregate (38 answerable questions, section-level relevance)

| metric | raw | clean |
|---|---|---|
| hit_rate@1 | 0.658 | 0.658 |
| hit_rate@3 | 0.895 | 0.895 |
| hit_rate@5 | 0.974 | 0.974 |
| hit_rate@10 | 1.000 | 1.000 |
| mrr@10 | 0.795 | 0.795 |

Identical to three decimals. **Verified this was a coincidence, not a fourth
bug**, via a per-question rank comparison (`evals/compare_ranks.py`): similarity
scores differ on nearly every question and ranks differ on 12 of 38.

Exactly 6 improved and 6 degraded. Five questions moved into rank 1 and five
moved out, holding hit@1 at 25/38. The MRR reciprocal changes summed to +2.833
against −2.833.

**Gains:** video-4000 (4→1), minions-rotation-low (2→1), minions-rotation-high
(2→1), minions-vuln-bomb (2→1), defensive-divert (2→1), splitsoul-3500 (6→4)

**Losses:** intro-familiar (1→2), intro-enrage (1→2), rotation-upkeep (1→2),
stacks-for-core (1→2), defensive-pillars (1→3), arms-rotation-low (3→6)

### Conclusions

**Preprocessing had no net effect on section-level retrieval.** Contradicts the
expectation set by the 37.6%-markup finding. Recorded as-is.

**The video enrichment worked, unambiguously.** `video-4000`: rank 4 → 1, score
0.392 → 0.641. `video-gm-timer`: 0.400 → 0.629. Largest score movement in the
table, directly attributable to adding the word "video" to text that lacked it.
At `top_k=3` that question went from unanswerable to answered.

**Similarity scores rose broadly even where ranks held.** living-death-timing
0.495→0.619, living-death-arms 0.600→0.724, presets-sigil 0.342→0.450,
gear-flurry-autos 0.449→0.595. Rank is a coarse view of a continuous change.

**Hypothesis (untested):** losses cluster on rotation content, and
arms-rotation-low fell 3→6 with score 0.562→0.439. Expanding emoji may make
rotation chunks *more similar to each other* — they are now all sequences of the
same ability names, where numeric IDs previously acted as accidental
discriminators. Preprocessing may trade distinctiveness for readability.

---

## 6. The finding that matters most

Setting the A/B aside, the worst questions **in both collections**:

| question | raw | clean |
|---|---|---|
| rotation-splitsoul-3500 | 6 | 4 |
| arms-rotation-low-enrage | 3 | 6 |
| arms-rotation-high-enrage | 4 | 4 |
| arms-tank-swap-timing | 5 | 5 |

Three of four are **enrage-conditional**. The system cannot distinguish "below
2500%" from "at 3000%" — predicted when the golden set was designed, now
evidenced.

**Immediate practical consequence:** `top_k = 3`, so these four questions
retrieve **zero relevant context**. The model can only refuse or hallucinate.

**Actions this justifies (in order):**

1. Test `top_k=5` — one-line change, would fix three of the four.
2. Hybrid search (BM25 + dense). RS3 questions hinge on exact rare tokens —
   item names, ability names, enrage numbers — which dense embeddings blur.
3. Metadata filtering on enrage band (header metadata is already stored, unused).
4. Cross-encoder reranker.

---

## 7. Metric v2: phrase-level relevance

Section-level relevance hit its limit. Seven sections is too coarse to detect
within-section reranking, which is most of what preprocessing changed.

**Change.** Relevance now requires BOTH conditions:
- the chunk's deepest header matches `expected_section`
- `expected_phrase` appears in the chunk text

**Fairness constraint.** Phrases are drawn only from prose that is
byte-identical in the raw and preprocessed guide. A phrase like "Ripper Demon"
would be unusable — in raw it is still `<:ripperpouch:...>`, so the A/B would be
rigged. All 38 phrases were verified present in both versions before use.

**Why both conditions.** Phrase alone is not always unique (`0-2500%` appears in
both the arms and minion sections). Section alone is too coarse. Together they
pin the exact chunk — and this is precisely what makes the enrage pairs a real
test: `0-2500%` vs `2500%+` within the same section.

**Verification.** Relevant-set sizes dropped from 3–9 chunks to 1–2 (mean 1.1)
in both collections, with zero questions unmatched — so no phrase was split
across a chunk boundary.

**Bug found during refactor.** `compare_ranks.py` imported `relevant_ids` and
then redefined a function of the same name below it. The local definition
shadowed the import silently. Fixed by moving shared logic into
`evals/relevance.py` and deleting the duplicates — the drift between two copies
of the same logic was the root cause.

---

## 8. Result: preprocessing A/B, phrase-level metric

| metric | raw | clean | change |
|---|---|---|---|
| hit_rate@1 | 0.526 | 0.632 | +10.6 pp |
| hit_rate@3 | 0.737 | 0.816 | +7.9 pp |
| hit_rate@5 | 0.816 | 0.895 | +7.9 pp |
| hit_rate@10 | 0.921 | 0.974 | +5.3 pp |
| mrr@10 | 0.657 | 0.739 | +12.5% rel. |

38 questions scored. 17 ranked differently: **11 better, 6 worse**.

**The same experiment that showed no effect under section-level relevance shows
a +20% relative gain in hit@1 here. Only the measurement changed.**

### Statistical significance — NOT reached

Paired t-test over per-question reciprocal ranks: mean difference +0.082,
SD 0.317, **t ≈ 1.59, df 37, p ≈ 0.12**. Sign test on 11-vs-6: p ≈ 0.33.

Honest statement: *consistent improvement across all five metrics, not
statistically significant at n=38.* Do not claim a proven win. This is the
strongest argument for growing the golden set to ~100 questions.

### Biggest gains

| question | raw | clean |
|---|---|---|
| minions-scythe-order | 7 | 1 |
| minions-vuln-bomb | 6 | 1 |
| video-4000 | 4 | 1 |
| rotation-opening-sequence | not found | 2 |
| rotation-splitsoul-3500 | not found | 4 |

The two "not found" cases are the strongest evidence: in raw, the opening
sequence is a wall of numeric IDs with nothing for the embedding to match.

### Biggest regressions, and a mechanism

| question | raw | clean |
|---|---|---|
| defensive-divert-damage | 2 | 10 |
| defensive-pillars | 1 | 3 |
| rotation-stacks-for-core | 1 | 2 |

**Hypothesis: term collision.** In raw, "Divert" appeared as
`<:divert:787904334377648130>` almost everywhere, so the *word* was rare. After
preprocessing it is plain text across four sections, so the target chunk now
competes with several others carrying the same term. Preprocessing raised recall
but reduced distinctiveness — the markup noise was accidentally acting as a
discriminator. Testable; not yet tested.

---

## 9. Confirmed: the system cannot resolve enrage bands

Worst-ranked questions in **both** collections:

| question | raw | clean |
|---|---|---|
| arms-rotation-high-enrage | 8 | 7 |
| arms-rotation-low-enrage | 6 | 7 |
| arms-tank-swap-timing | 5 | 5 |
| rotation-splitsoul-3500 | not found | 4 |
| intro-enrage-range | 3 | 4 |

Preprocessing does not help and cannot: the problem is not readability but that
`0-2500%` and `2500%+` are near-identical in embedding space. Predicted when the
golden set was designed, now evidenced twice under two different metrics.

**Also:** `rotation-upkeep` is never retrieved in the top 10 in either
collection. Unexplained — investigate separately.

**Practical:** at the configured `top_k = 3`, 18% of questions retrieve zero
relevant context even on the clean collection.

### Actions this justifies, in order

1. Grow the golden set to ~100 questions — current n cannot resolve the effect.
2. Hybrid search (BM25 + dense). Exact rare tokens (enrage numbers, item names)
   are what dense embeddings blur. Directly targets the confirmed failure.
3. Metadata filtering on enrage band (header metadata already stored, unused).
4. Cross-encoder reranker — hit@10 is 0.974 while hit@1 is 0.632, so the right
   chunk is usually retrieved but poorly ranked. That gap is the reranker's case.
5. Test `top_k=5`.
6. Investigate `rotation-upkeep`.
7. Test the term-collision hypothesis.

---

## 2026-08-13 — Session 2: three guides

Added `arch_glacor_high_enrage_melee_magic.txt` and
`..._melee_ranged.txt`; the original guide was renamed to `..._necromancy.txt`.

### What adding two guides broke

**Silently, and worth noting for the writeup:**

- `guide_path` pointed at a renamed file — ingest and all tests failed immediately.
- **Two of four unanswerable questions became answerable.** `unanswerable-melee`
  and `unanswerable-magic-setup` are now covered by the new guides. Left in place
  they would have penalised the model for answering correctly, and looked like a
  model regression. Deleted; abstention slice is down to 2 and needs rebuilding.
- **Most questions became ambiguous.** "What is the arms rotation at 3000%?" now
  has three correct answers. 37 questions were style-qualified ("...with
  Necromancy").
- **Section labels stopped being unique.** Both hybrid guides contain a section
  literally called `Melee Phase`, so relevance needed `expected_source` as well.
- 62 of 116 emoji shortcodes were unmapped.

### Changes

- `guide_path` -> `guide_dir`; ingest walks every `.txt`.
- New `rag/sources.py`: derives `boss`/`style` from the filename pattern
  `<boss>_high_enrage_<style>.txt`. Every chunk stamped with `source`, `style`,
  `boss`. The `style` field is the hook for metadata filtering later.
- All 116 shortcodes mapped (2 flagged UNCERTAIN: `cane`, `ezk`).
- Relevance = source + section + phrase.
- `collapse_expansion_repeats`: the guides write a name in prose *and* place its
  emoji beside it, so expansion produced "Powder of Penance Powder of Penance".
  Only known mapped names are collapsed - a general repeated-phrase rule would
  corrupt rotations that legitimately cast an ability twice.
- Video embeds: handle multiple URLs per field and both `youtu.be/` and
  `youtube.com/watch?v=` forms.
- Paired t-test moved into the harness. Reciprocal ranks are computed locally
  rather than taken from ranx, because a paired test compares vectors
  position-by-position and silently returns nonsense if they are misaligned.

### Abbreviation aliases

32 shortcodes now render as "Full Name (code)" — `Greater Sonic Wave (gsonic)`.

Rationale: players ask "when do I use fsoa?". Expanding to
"Fractured Staff of Armadyl" alone **deletes the exact token the query needs**.
Also gives BM25 a rare high-signal term once hybrid search lands, and limits the
damage from a wrong mapping since the original code survives.

Costs ~9% more characters. Controlled by `emoji_aliases`, so it is measurable.

### Result: three-way A/B (38 questions, 3 guides)

| metric | raw | noalias | clean (aliases) |
|---|---|---|---|
| hit_rate@1 | 0.447 | **0.605** | 0.579 |
| hit_rate@3 | 0.632 | 0.737 | **0.763** |
| hit_rate@5 | 0.711 | 0.816 | **0.842** |
| hit_rate@10 | 0.763 | **0.947** | 0.921 |
| mrr@10 | 0.552 | **0.702** | 0.688 |

Paired t-test on reciprocal rank vs `raw` (n=38):

| comparison | mean diff | t | p |
|---|---|---|---|
| noalias | +0.151 | +2.08 | 0.0444 |
| clean | +0.137 | +1.92 | 0.0629 |

**Preprocessing's effect nearly doubled with the larger corpus** (+0.082 MRR on
one guide, +0.151 on three). Three times the chunks means three times the
distractors, so markup noise costs more. Absolute scores fell for every variant
(raw hit@1 0.526 -> 0.447): the task genuinely got harder.

### How to state this honestly

**Do not claim "preprocessing significant, aliases not."** The two effects are
+0.151 and +0.137 — nearly identical. The p-values straddle 0.05 by a hair, and
0.05 is arbitrary. Additionally these are **two tests against one baseline**;
under Bonferroni correction (p < 0.025) neither passes.

Correct statement: *preprocessing yields a ~25% relative MRR improvement;
at n=38 the evidence is borderline (p ~ 0.04-0.06). The golden set must grow
before significance can be claimed.*

### The alias experiment measured nothing

All 38 questions are phrased in full names. Nothing in the golden set types
`fsoa` or `gconc`, so aliases had no opportunity to help. Every alias-vs-noalias
difference is exactly **one question** out of 38, in both directions depending
on the metric.

Conclusion: aliases are **untested**, not useless. Testing them requires
jargon-phrased questions.

---

## Session 2b — golden set doubled, result now significant

### Golden set: 40 -> 72 questions

32 questions added covering the two hybrid guides. Every phrase verified present
in the raw guide **and** all three emoji modes before being written, so corpus
A/Bs stay fair.

| coverage | count |
|---|---|
| necromancy | 37 |
| melee_magic | 8 |
| melee_ranged | 7 |
| either hybrid (shared text) | 17 |
| any guide | 3 |
| unanswerable | 2 |

### Finding: the two hybrid guides are near-duplicates

The Opener, Melee Phase, Defensive Usage and General Rotations sections are
**word-for-word identical** between `melee_magic` and `melee_ranged`. They differ
only in the Magic/Ranged phase, the familiar, the preset link and the videos.

Consequence: `expected_source` had to accept a **list** — for those 17 questions
either guide is a correct answer. Tagged `shared-text` so the metric can be
sliced by it.

This is a realistic hard case. Near-duplicate documents are common in real
corpora, and two near-identical chunks split the similarity score for the same
query. Hypothesis: they push each other down the ranking. Now testable via the
per-tag breakdown added to the harness.

### Result: preprocessing A/B at n=70

| metric | raw | preprocessed (`both`) | relative |
|---|---|---|---|
| hit_rate@1 | 0.371 | **0.557** | +50% |
| hit_rate@3 | 0.557 | **0.729** | +31% |
| hit_rate@5 | 0.700 | **0.814** | +16% |
| hit_rate@10 | 0.771 | **0.900** | +17% |
| mrr@10 | 0.493 | **0.661** | +34% |

Paired t-test on reciprocal rank: mean diff **+0.168, t=3.25, p=0.0018, n=70**.

**Significant, and clears Bonferroni correction** — unlike the n=38 run
(p=0.044) where the caveat applied. Doubling the golden set is what settled it.

Publishable claim: *preprocessing PvME's Discord markup improved retrieval MRR
by 34% relative (0.493 -> 0.661, p=0.0018, n=70).*

Absolute scores fell again versus n=38 (raw hit@1 0.447 -> 0.371) because the
new hybrid questions are harder — 17 of them target text duplicated across two
files.

### Emoji mode: parked, not resolved

Extended `emoji_aliases` (bool) to `emoji_mode` = `full` | `alias` | `both`,
after considering whether to drop full names and keep only abbreviations.

Reasoning against alias-only, recorded for the writeup: dense embeddings work on
semantics, and `fsoa` has none — MiniLM never saw it, so the tokenizer produces
meaningless fragments. "Fractured Staff of Armadyl" contains real words with
real embeddings. This is the same mechanism that made the raw collection score
badly: the numeric IDs were not merely long, they were *semantically empty*.
Short codes also collide with ordinary English (`res`, `anti`, `prep`, `rend`).
And the generation step and human-readable citations both want real names.

The counter-argument stands for **BM25**, where `fsoa` is a rare high-signal
exact token — an argument for hybrid search, not for dropping full names.

`both` kept as the default: **a reasoned default, not a measured one.** Only 5
of 72 questions use jargon phrasing, so the experiment still cannot resolve it.
Revisit alongside hybrid search.

### Per-tag diagnostic (mean reciprocal rank)

Added a per-tag slice to the harness. This is where question tagging pays off:
the headline number says *whether* something works, the slice says *what to fix*.

| tag | n | raw | preprocessed | delta |
|---|---|---|---|---|
| from-embed | 5 | 0.369 | **1.000** | +0.63 |
| link | 4 | 0.271 | **0.875** | +0.60 |
| lookup | 20 | 0.445 | 0.710 | +0.27 |
| conditional | 8 | 0.598 | 0.806 | +0.21 |
| jargon | 5 | 0.420 | 0.600 | +0.18 |
| enrage-conditional | 8 | **0.167** | **0.339** | +0.17 |
| sequence | 14 | 0.395 | 0.524 | +0.13 |
| why | 8 | 0.719 | 0.812 | +0.09 |
| shared-text | 17 | 0.512 | 0.573 | +0.06 |
| negative-fact | 4 | 0.812 | 0.800 | -0.01 |
| numeric | 8 | 0.651 | **0.575** | **-0.08** |

**The embed parser is the single best-performing piece of work: 0.369 -> 1.000.**
Those five questions (Kwuarm stacks, Limitless Sigil, Powder of Penance, two
videos) were buried inside raw JSON blobs and are now always rank 1. This
validates the decision to write a real embed parser instead of hand-editing
three embeds — the parser generalises to every future PvME guide.

**Link retrieval 0.271 -> 0.875** — the "Example kill video" enrichment.

**`enrage-conditional` is the worst tag, 0.339 vs 0.661 overall.** Preprocessing
helps but leaves it at roughly half the average. MRR 0.339 means the correct
chunk sits near rank 3, so at `top_k=3` these are coin flips. Strongest possible
case for hybrid search: `0-2500%` and `2500%+` are lexically distinct and
semantically identical — precisely the gap BM25 fills.

**`shared-text` confirms the near-duplicate hypothesis.** Below average (0.573)
and — more tellingly — the *smallest gain of any tag* (+0.06). Preprocessing
cannot help, because the problem is two identical chunks competing rather than
markup. Fix is metadata filtering on `style` (already stored, unused), not
cleaning.

**Regression: `numeric` 0.651 -> 0.575.** n=8 so possibly noise, but consistent
with the term-collision hypothesis: preprocessing surrounds numbers with ability
names that now appear in many chunks, so "how many Bloodlust stacks?" competes
with every chunk mentioning Bloodlust. Investigate before dismissing.

### Priority order this implies

1. **Hybrid search (BM25 + dense)** — targets `enrage-conditional` (0.339), the
   worst tag, and `jargon` (0.600). Also finally makes the alias question
   testable.
2. **Metadata filtering on `style`** — targets `shared-text` (n=17, unresponsive
   to preprocessing).
3. **Cross-encoder reranker** — hit@10 0.900 vs hit@1 0.557 means the right
   chunk is usually retrieved and badly ranked.
4. Investigate the `numeric` regression.

## Session 3 — hybrid search (dense + BM25)

### Why

The per-tag diagnostic made the case: `enrage-conditional` sits at 0.339 against
0.661 overall. The failure is structural, not a bug.

`0-2500%` and `2500%+` are *semantically identical* — both are "Arch-Glacor arms
rotation, necromancy abilities, an enrage threshold". Their difference is a few
characters, averaged into 384 dimensions alongside everything else. Compression
is the point of embeddings, and small exact details are what gets compressed
away.

BM25 has the opposite profile: it understands nothing, but scores rare exact
tokens enormously. `0-2500%` appears in 2 chunks of ~70, so a query containing it
scores those two hugely and everything else zero.

Neither is sufficient alone — BM25 would destroy the `why`/`conditional`
questions currently at 0.81, which depend on paraphrase. Run both, merge.

### Implementation

- `rag/sparse.py` — BM25 vectors via fastembed `Qdrant/bm25`. Note `embed` vs
  `query_embed`: the query side skips term-frequency weighting, since a query
  mentioning a word twice does not make it twice as important.
- `rag/ingest.py::ingest_hybrid` — builds the collection with the **raw** Qdrant
  client rather than LangChain, so vector names and payload shape are explicit.
  The eval reads the same collection directly and the two must agree exactly.
  Payload shape kept identical to LangChain's (`{text, metadata}`) so
  `relevance.py` needs no branch.
- `models.Modifier.IDF` on the sparse config — IDF is applied **server-side**, so
  corpus statistics stay correct as documents are added.
- `rag/retrieve.py` — shared low-level search. `is_hybrid_collection` inspects
  the collection config, so one eval run can compare dense-only and hybrid
  collections side by side without a flag.
- `rag/search.py::HybridRetriever` — hand-written LangChain retriever. LangChain's
  own Qdrant hybrid support hides vector names and fusion settings; since the
  eval queries the same collection directly, they must match exactly.
- `--hybrid` CLI flag, `hybrid` setting, `hybrid_prefetch=40`.

### How RRF works, and its limitation

Fusion uses **Reciprocal Rank Fusion**: `score = Σ 1/(60 + rank)` across
retrievers. Positions are used rather than raw scores because a cosine
similarity and a BM25 score are not on comparable scales, and normalising them
is fiddly.

**RRF discards score magnitude.** Verified on a synthetic collection: with IDF
the sparse retriever scored the correct chunk 9.30 vs 0.47 — a 20x margin — but
under RRF that counts only as "rank 1". In the toy case dense happened to rank
the pair in the opposite order, so the two cancelled exactly and fused to a tie.

That is RRF working as designed: robust, no normalisation needed, but blind to
confidence. Qdrant also supports **DBSF** (distribution-based score fusion),
which does use scores — a follow-up experiment if RRF underdelivers.

### Verified in the sandbox

- Qdrant hybrid query path: named vectors, `Prefetch` x2, `FusionQuery(RRF)`.
- `Modifier.IDF` sparse scoring discriminates correctly (9.30 vs 0.47).
- `is_hybrid_collection` detection.
- Payload shape round-trips for `relevance.py`.

Not verified locally: the BM25 model download is blocked by the sandbox proxy,
so the fastembed call itself runs first on Jelle's machine.

### Predictions (written before running)

| tag | dense now | expectation |
|---|---|---|
| enrage-conditional | 0.339 | **large gain** — exact `0-2500%` match |
| jargon | 0.600 | large gain — `fsoa`, `grico` are rare tokens |
| numeric | 0.575 | gain — "27k", "4 stacks" are exact |
| sequence | 0.524 | modest gain |
| shared-text | 0.573 | **no change** — identical text scores identically under BM25 too |
| why / conditional | 0.81 | flat, possibly slightly worse |

If `shared-text` moves materially, the reasoning above is wrong and worth
digging into.

### Result

| metric | raw | dense | hybrid |
|---|---|---|---|
| hit_rate@1 | 0.371 | 0.557 | **0.657** |
| hit_rate@3 | 0.557 | 0.729 | **0.857** |
| hit_rate@5 | 0.700 | 0.814 | **0.914** |
| hit_rate@10 | 0.771 | 0.900 | **0.943** |
| mrr@10 | 0.493 | 0.661 | **0.763** |

Paired t-test vs raw: hybrid **+0.270, t=5.48, p<0.0001**.
(dense-only was +0.168, t=3.25, p=0.0018.)

At the configured `top_k=3`, 86% of questions now retrieve relevant context,
against 56% dense-only and 36% unprocessed.

### Per-tag, and the prediction scorecard: 2 of 6 correct

| tag | n | raw | dense | hybrid | prediction |
|---|---|---|---|---|---|
| conditional | 8 | 0.598 | 0.806 | **1.000** | WRONG - said flat/worse |
| enrage-conditional | 8 | 0.167 | 0.339 | 0.435 | WRONG - said large gain |
| from-embed | 5 | 0.369 | 1.000 | 1.000 | at ceiling |
| jargon | 5 | 0.420 | 0.600 | **0.900** | CORRECT |
| link | 4 | 0.271 | 0.875 | 0.875 | - |
| lookup | 20 | 0.445 | 0.710 | 0.787 | - |
| negative-fact | 4 | 0.812 | 0.800 | 0.875 | - |
| numeric | 8 | 0.651 | 0.575 | **0.729** | CORRECT (regression recovered) |
| sequence | 14 | 0.395 | 0.524 | 0.516 | WRONG - said modest gain |
| shared-text | 17 | 0.512 | 0.573 | **0.720** | WRONG - said no change |
| why | 8 | 0.719 | 0.812 | **0.938** | WRONG - said flat/worse |

### What the wrong predictions taught

**RRF is asymmetric.** The assumption that adding BM25 would drag down
paraphrase-heavy questions was wrong. Fusion considers the *union* of both
candidate lists, so a chunk dense ranked 1st still collects 1/61 from dense,
while BM25's noise scatters across different documents and rarely outranks it.
A weak second retriever is nearly free and can essentially only add recall.
Hence `conditional` reaching 1.000 and `why` 0.938.

**Duplicate text hurts less than claimed.** For `shared-text` questions the
relevance set accepts *either* guide's chunk - only one of the two needs to
appear. Dense was splitting confidence between two identical chunks and ranking
both mid-list; BM25 latched onto a rare term and pulled one up. Duplication is
only fatal when a *specific* copy is required.

**`enrage-conditional` moved least despite being the motivating case.**
Hypothesis: **BM25 tokenization destroys the distinction.** `0-2500%` likely
tokenizes to `0` + `2500`; `2500%+` to `2500`. Both chunks then contain `2500`,
and the discriminating characters (`0-` prefix, `+` suffix) are exactly what
tokenizers strip. BM25 cannot separate them either - for a completely different
reason than dense.

Cheap proposed fix: normalise enrage bands during preprocessing into single
distinctive tokens (`enrage_0_2500`, `enrage_2500_plus`), giving BM25 something
matchable. **Verify the tokenization claim before building this.**

**`sequence` stayed flat (0.516).** Rotation chunks are long lists of the same
ability names, so neither retriever has a discriminating signal. This is the
reranker's case, not hybrid's.

## Session 4 — enrage band tokens

### Hypothesis confirmed by measurement

`scripts/check_bm25_tokens.py` tested the claim from session 3:

```
'2500%+'   -> 1 term    (% and + stripped; the term is just '2500')
'0-2500%'  -> 2 terms   ('0' and '2500')
query vs LOW chunk: 1 term matched
query vs HIGH chunk: 1 term matched   <- identical, cannot discriminate
```

**BM25 preserves the number and destroys the direction.** Confirmed by contrast:
`'below 2500%'` vs `'3500%+ enrage'` share zero terms — different *numbers*
separate fine. Only above-versus-below is lost.

### Fix: `rag/enrage.py`

Every enrage expression is annotated with the 500-point bands it covers:

```
- **0-2500%:** [enrageband0 ... enrageband2500]
- **2500%+:**  [enrageband2500 ... enrageband5000]
```

**Band membership, not direction tokens** (`upto`/`from`), because a query can
name a value *inside* a range — "arms rotation at 3000%" must match the chunk
labelled `2500%+`. Bands handle both phrasings with one scheme.

**Appended, never substituted**, so `0-2500%` survives as literal text and the
golden-set phrases still match. Costs 8.5% more characters.

**Query and document processed identically** — annotation happens inside
`sparse.embed_query`, so both sides speak the same language. Analyzer symmetry;
forgetting it is a classic silent failure (documents full of tokens no query
ever produces).

**Sparse side only.** `enrageband3000` is meaningless to a sentence-transformer
and would only add noise to the dense vector. Applying different query
processing per retriever is a concrete advantage of hand-rolling fusion rather
than using a black-box hybrid helper.

**Bug found while testing:** `'below 2500%'` matched both the below-rule and the
at-rule, emitting a spurious `enrageband2500` that collided with the `2500%+`
chunk — recreating the exact confusion the module exists to remove. Fixed by
blanking each matched span before the next pattern runs.

**False-positive guard verified:** `'100% adrenaline'` and `'flick to Soul Split
for 1 tick to get to 100% adrenaline'` produce zero tokens, because the at-rule
requires the word "enrage" nearby.

### Result

| metric | raw | preprocessed | hybrid | + bands |
|---|---|---|---|---|
| hit_rate@1 | 0.371 | 0.557 | 0.657 | **0.700** |
| hit_rate@3 | 0.557 | 0.729 | 0.857 | **0.886** |
| hit_rate@5 | 0.700 | 0.814 | 0.914 | **0.929** |
| hit_rate@10 | 0.771 | 0.900 | 0.943 | **0.957** |
| mrr@10 | 0.493 | 0.661 | 0.763 | **0.793** |

Paired t-test vs raw, n=70:

| config | mean diff | t | p |
|---|---|---|---|
| preprocessed | +0.168 | 3.25 | 0.0018 |
| hybrid | +0.270 | 5.48 | <0.0001 |
| + bands | **+0.300** | **6.04** | **<0.0001** |

### Per tag — the targeted fix moved its target most

| tag | n | raw | preproc | hybrid | bands | hybrid->bands |
|---|---|---|---|---|---|---|
| enrage-conditional | 8 | 0.167 | 0.339 | 0.435 | **0.589** | **+0.154** |
| jargon | 5 | 0.420 | 0.600 | 0.900 | **1.000** | +0.100 |
| sequence | 14 | 0.395 | 0.524 | 0.516 | 0.557 | +0.041 |
| numeric | 8 | 0.651 | 0.575 | 0.729 | 0.771 | +0.042 |
| shared-text | 17 | 0.512 | 0.573 | 0.720 | 0.752 | +0.032 |
| lookup | 20 | 0.445 | 0.710 | 0.787 | 0.789 | +0.002 |
| conditional | 8 | 0.598 | 0.806 | 1.000 | 1.000 | at ceiling |
| from-embed | 5 | 0.369 | 1.000 | 1.000 | 1.000 | at ceiling |
| why | 8 | 0.719 | 0.812 | 0.938 | 0.938 | flat |

`enrage-conditional` gained the most of any tag, on exactly the change designed
for it, and is now 3.5x its baseline. Clean cause and effect: measure a specific
failure -> design a specific fix -> the targeted metric moves most.

### Retrieval work stops here

`enrage-conditional` (0.589) and `sequence` (0.557, n=14) remain the weakest,
but the whole pipeline is at hit@3 0.886 — 89% of questions retrieve relevant
context at the configured `top_k=3`, against 56% unprocessed. Further retrieval
tuning has clearly diminishing returns.

`sequence` is the reranker's case: rotation chunks are long lists of the same
ability names, so ranking needs to read query and chunk *together* rather than
compare vectors. Deferred - the generation half of the system is still entirely
unmeasured, which is a much bigger gap.

## Session 5 — experiment tracking

### The trap this avoids

Naive approach: log `settings.model_dump()` when the eval runs. That records the
config **you have**, not the config that **built the index**. They diverge the
moment a default changes — and MLflow would confidently display wrong params
with no error anywhere.

`rag/registry.py` writes `data/collections.json` at ingest time: embedding
model, chunk size/overlap, emoji mode, enrage bands, hybrid, chunk count,
guides, timestamp. The eval reads *that*. Collections built before the registry
existed log `build_config: "unknown"` rather than a plausible-looking lie.

This also closes an open item from session 1: `check_embedding_model` runs
before each collection is scored and raises if the model differs from the one
that built it. That mismatch never crashes on its own — both models emit vectors
of plausible shape in incompatible spaces — it just silently returns noise.
Guard verified to fire.

### `evals/tracking.py`

One MLflow run per collection: all params from the registry, all metrics, and
the per-tag breakdown as `tag_*` metrics. Plus a `comparison_vs_<baseline>` run
holding the paired t-tests. Local `./mlruns` by default; DagsHub via
`DAGSHUB_REPO_OWNER` + `DAGSHUB_REPO_NAME`.

**Bug:** MLflow rejects `@` in metric names, so `hit_rate@1` failed validation.
Sanitised to `hit_rate_at_1` / `mrr_at_10`. Hyphens are legal, so the tag
metrics keep their names.

### CLI flags fixed

Flipping `hybrid` and `enrage_bands` to `True` by default made their
`store_true` flags inert — a store_true can only turn something *on*. Converted
to `argparse.BooleanOptionalAction` with `default=None`, giving `--hybrid` /
`--no-hybrid` pairs.

`default=None` matters: it distinguishes "flag not given" from "given as false".
With `default=False` the CLI would silently override config defaults on every
run — the same class of bug as the `--collection` flag sitting after an early
`return` in session 2.

### Defaults now reflect measured results

`hybrid=True` (session 3), `enrage_bands=True` (session 4), each with a comment
citing the session that measured it.

### State

4 runs logged. Full progression visible in one place with the config that
produced each:

| collection | preprocess | hybrid | bands | mrr@10 |
|---|---|---|---|---|
| arch_glacor_raw | no | no | no | 0.493 |
| arch_glacor_both | yes | no | no | 0.661 |
| arch_glacor_hybrid | yes | yes | no | 0.763 |
| arch_glacor_bands | yes | yes | yes | **0.793** |

## Session 6 — generation evaluation

### RAGAS rejected, metrics written by hand

`ragas 0.4.3` imports `langchain_community.chat_models.vertexai`, which no
longer exists - langchain-community is being sunset in favour of per-vendor
packages (`langchain-ollama`, `langchain-qdrant`, ...). Pinning an old version
would fight the project's LangChain 1.x install.

Second reason, independent of the conflict: RAGAS prompts request structured
JSON and are tuned for frontier models. A 9B judge often returns something
unparseable and RAGAS records NaN for that row - a plausible-looking average
computed from partial data, with no warning.

Implemented four metrics directly, with prompts designed for a small judge:
one question per call, single word or number requested, never JSON, retry once,
and **parse failures counted as a metric** so a struggling judge is visible.

| metric | question it answers |
|---|---|
| faithfulness | is what the model said grounded in what it retrieved? |
| relevancy | does the answer address the question asked? |
| refusal_rate | on unanswerable questions, did it decline? |
| false_refusal_rate | on answerable questions, did it wrongly decline? |
| empty_answer_rate | did generation fail outright? |

`false_refusal_rate` exists because a model that refuses everything would score
a perfect `refusal_rate`. Both are needed to tell calibration from cowardice.

### Two-phase design (generate, then judge)

Forced by 8GB VRAM: answerer and judge would otherwise be swapped per question.
Turned out to matter more for iteration - prompts and metrics changed five times
today, and re-judging cached answers takes 4 minutes instead of 11.

### Four bugs, each found by reading a suspicious row

**1. Reasoning mode.** Qwen3.5 emits a `<think>` block by default. Measured:
98.5s per answer, 2 of 8 answers **empty** (whole output budget spent thinking),
2 more truncated mid-sentence. The tell was that the empty answers were the
*slowest*. Fixed in `rag/llm.py` with `reasoning=False` and `num_predict`.
**98.5s -> 3.5s per answer, a 28x speedup.**

**2. Sentence splitter broke on quotes.** `(?<=[.!?])\s+` misses a sentence
ending inside quotation marks - the character before the space is `"`, not `.`.
Whole answers became one compound claim, and a single unsupported fragment
scored the entire answer 0.0. Found by checking a 0.00 that should have been
1.00 (`intro-slayer-task`, which quoted the context verbatim). The headline
faithfulness was depressed ~16 points by this alone.

**3. Prompt caused meta-commentary.** The link instruction made the model add
"Additionally, there are no links provided in the context to include" to
answers. Prompt now says answer directly and do not describe the context.

**4. Retrieval-only tokens leaking to the LLM.** `[enrageband3000 ...]` was
reaching the model as literal text. Stripped in `format_docs`, not at ingest -
the index still needs them.

### The finding worth keeping

Prediction: false refusals would be retrieval failures.
**Wrong - 8 of 10 refusals had the correct chunk in context.**

Cause: `format_docs` concatenated `page_content` and discarded metadata. Session
2 added `source`/`style` to every chunk specifically so relevance could tell the
three guides apart - and then never showed it to the model. An example-kill line
reads "Example kill video, Arch-Glacor: ..." whether it came from the Necromancy
guide or a hybrid one, so "is there a video for the Melee/Magic hybrid?" was
unanswerable *even holding the right chunk*. The model refused, correctly.

Fix: each chunk is now prefixed `[From the Hybrid Melee/Magic guide]`.

**Principle: retrieval and generation want different views of the same chunk.**
Band tokens - index yes, model no. Source metadata - index yes, model also yes.

### Results (82 questions, arch_glacor_bands, top_k=3)

| metric | before labels | after labels |
|---|---|---|
| faithfulness | 0.950 | 0.919 |
| relevancy | 0.880 | 0.864 |
| **refusal_rate** | **1.000** | **1.000** |
| **false_refusal_rate** | 0.143 | **0.086** |
| empty_answer_rate | 0.000 | 0.000 |
| judge_parse_failure_rate | 0.000 | 0.000 |

**refusal_rate 1.000**: all 12 unanswerable questions declined, including
`unanswerable-pure-ranged`, where retrieval definitely returned plausible
Ranged Phase content. No hallucinations on the abstention slice.

**false_refusal_rate halved**, which was the target of the labelling fix.

**faithfulness and relevancy dipped, and that is NOT a regression.** It is a
composition effect: the 8 newly-answered questions were previously excluded
from faithfulness, because refusals are not scored. More questions attempted =
more opportunities to be imperfect.

**Metric design flaw, recorded rather than hidden: faithfulness averaged over
non-refusals is not comparable across runs with different refusal rates.**
Future runs should report the number of scored answers alongside it, or a
combined "answered and faithful" rate.

### Judge reliability - and a hazard

Spot-checking found the judge wrong at least three times: `intro-slayer-task`
(0.00, should be 1.00 - splitter bug), `video-gm-timer` and
`mm-minions-swap-early` (0.00 on answers that correctly paraphrase the context).
The last two share a cause: the judge read "supported by" as "appears verbatim".
Prompt now states explicitly that paraphrase counts.

**Hazard named: iterating the judge until the numbers look good is p-hacking.**
Clear defects were fixed (splitter, verbatim-strictness). Tuning stops here.
The honest next step is measuring judge-human agreement on a hand-labelled
sample and reporting it as a caveat, rather than continuing to adjust prompts.

Also outstanding: the judge is the same model as the answerer, so faithfulness
is optimistic (self-preference bias). `judge_model` is a separate setting, so
swapping in an independent judge is a one-line change.

---

## Open items

**Blocked on game knowledge (Jelle):**
- [ ] Confirm `cane` mapping — guessed "Cleave". Context: *"Don't overspend
      Bloodlust stacks before Glacyte Minions during Berserk, needed for `cane`
      cooldown refund"*
- [ ] Confirm `ezk` mapping — guessed "Igneous Kal-Zuk". Context: *"Have `ezk`
      equipped as often as possible while meleeing for Flamebound Rival damage
      reduction"*
- [ ] Verify `anti` → "Anticipation" (vs Anti-fire)
- [ ] Rebuild the abstention slice: ~8 questions genuinely outside all three guides

**Next work:**
- [ ] Port question archetypes to both hybrid guides -> ~100 questions.
      Fixes three things at once: statistical power, generalisation beyond one
      guide, and gives aliases something to prove. Include jargon phrasings
      ("when do I use fsoa?", "what's the gconc rotation?") — without them the
      alias experiment stays untestable. Target the `Melee Phase` sections,
      where the two hybrids are most confusable.
- [ ] Log eval runs to MLflow with `settings.model_dump()` as params
- [ ] Hybrid search (BM25 + dense) — the confirmed fix for enrage bands
- [ ] Cross-encoder reranker — hit@10 0.947 vs hit@1 0.605 is the case for it
- [ ] Metadata filtering on `style` (already stored, unused)
- [ ] Test `top_k=5`
- [ ] Embedding-model consistency guard: write model name + dim into collection
      metadata, assert at startup
- [ ] Abstention metric once the slice is rebuilt
- [ ] Investigate `rotation-upkeep` (never retrieved in top 10)
- [ ] Test the term-collision hypothesis
- [ ] Minor: `Kal'gerion Demon Kal'gerion Demon pouch` — partial-overlap
      duplicate that exact-match collapsing does not catch
