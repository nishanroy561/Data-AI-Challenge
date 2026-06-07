# Product Requirements Document (PRD)
## Intelligent Candidate Discovery & Ranking System — Redrob Data & AI Challenge

| | |
|---|---|
| **Document owner** | (your team) |
| **Status** | Draft v1 |
| **Last updated** | 2026-06-07 |
| **Challenge** | Redrob Hackathon v4 — *India Runs on Data & AI* |
| **Deliverable** | A reproducible ranking system that outputs the top-100 candidates for a given JD, with reasoning |

---

## 1. Background & Context

Redrob AI is a Series-A "AI-native talent intelligence platform." Its core product problem is **matching**: deciding which candidates a recruiter sees when they search, and which roles a candidate sees. Today that intelligence layer is "mostly BM25 + rule-based scoring — working but not great."

This challenge is a proxy for that real production problem. We are given:

- A pool of **100,000 synthetic candidate profiles** (`candidates.jsonl`, ~487 MB uncompressed, one JSON object per line).
- **One job description** to rank against: *Senior AI Engineer — Founding Team*.
- A hidden **ground-truth relevance labeling** of those candidates (relevance tiers 0–5, with ~80 forced-to-0 honeypots), against which our submission is scored.

Our job is to produce a CSV ranking the **top 100 best-fit candidates**, best-first, each with a short reasoning — under hard production-like compute constraints.

This is explicitly **not** a keyword-matching exercise. The challenge is engineered so that naïve approaches (embed the JD, embed the profile, sort by cosine) fail, because the dataset is salted with adversarial traps that those approaches walk straight into.

---

## 2. Problem Statement

> **Given 100,000 candidate profiles and one job description, identify and rank the 100 candidates who are genuinely the best hires — reasoning about what the JD *means*, not just what it *says* — while avoiding the deliberately planted traps, and do it in ≤5 minutes on a CPU with no network access.**

### 2.1 Why this is hard (the real problem)

The difficulty is not retrieval mechanics; it's **judgment under deception**. The dataset and JD are co-designed so that the easy signal (AI keywords in the skills list) is *anti-correlated* with true fit in several subpopulations. A correct system has to model the **gap between surface text and underlying suitability**:

| The naïve signal says… | …but the truth is |
|---|---|
| "This profile lists RAG, Pinecone, LangChain, LLMs as skills" | The person's title is *Marketing Manager* and their career history shows no ML production work → **keyword stuffer, not a fit** |
| "This profile never mentions RAG or vector DBs" | Their career history shows they *built and shipped a recommendation/search system at a product company* → **a 'Tier-5' fit hiding in plain language** |
| "This candidate is perfect on paper" | They haven't logged in for 6 months, 5% recruiter response rate, not open to work → **not actually hireable; down-weight** |
| "8 years of experience, expert in 10 skills" | The company they list was founded 3 years ago, and they claim expert proficiency with 0 months of usage → **honeypot / impossible profile** |

### 2.2 The job description being ranked against

**Role:** Senior AI Engineer — Founding Team (Series-A, Pune/Noida, hybrid, 5–9 yrs).

The JD is written adversarially and is explicit about intent. Distilled requirements:

**Must-have (high weight):**
- Production experience with **embeddings-based retrieval** deployed to real users (sentence-transformers, OpenAI embeddings, BGE, E5, etc. — *model-agnostic*).
- Production experience with **vector DBs / hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch).
- **Strong Python** / code quality.
- Hands-on **ranking-evaluation** design (NDCG, MRR, MAP, offline↔online correlation, A/B interpretation).
- Has **shipped at least one end-to-end ranking / search / recommendation system** to real users at meaningful scale.
- "Shipper" disposition over "researcher."

**Nice-to-have (low weight, non-disqualifying):**
- LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank (XGBoost/neural), HR-tech exposure, distributed systems, OSS contributions.

**Disqualifiers / strong negatives (must be modeled):**
- Pure-research careers with no production deployment.
- "AI experience" = <12 months of LangChain-calling-OpenAI with no pre-LLM ML production depth.
- "Architect/tech-lead" who hasn't written production code in 18+ months.
- **Title-chasers** — job-hopping every ~1.5 years optimizing for title.
- **Pure-services-firm careers** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) for their *entire* career.
- Primary expertise in **CV / speech / robotics** without NLP/IR.
- Entirely closed-source proprietary work for 5+ yrs with no external validation.

**Soft preferences:**
- 6–8 yrs total, 4–5 in applied ML at product (not services) companies.
- Located in / willing to relocate to Noida or Pune (Tier-1 Indian cities welcome).
- Sub-30-day notice preferred.
- **Active on the platform** (so they're reachable) — behavioral availability matters.

### 2.3 The traps (must be explicitly defended against)

The dataset deliberately contains:

1. **Keyword stuffers** — all the right AI skills, but title/career history reveal a non-fit (e.g., HR Manager, Graphic Designer with 9 "AI core skills"). The sample submission shows these ranked *high* precisely to illustrate the wrong approach.
2. **Plain-language Tier-5s** — true fits whose profiles describe real ranking/retrieval work without buzzwords.
3. **Behavioral twins** — near-identical profiles that differ only in behavioral signals; the available one should outrank the dormant one.
4. **~80 honeypots** — subtly *impossible* profiles (tenure exceeding company age, "expert" with 0 months used, internally inconsistent dates). **Forced to relevance tier 0.** Ranking >10% of them in the top 100 ⇒ **disqualification at Stage 3.**

---

## 3. Goals & Non-Goals

### 3.1 Goals
- **G1.** Maximize the composite ranking score (defined in §6) against the hidden ground truth.
- **G2.** Correctly down-rank keyword stuffers, honeypots, and behaviorally-unavailable candidates.
- **G3.** Correctly surface plain-language true fits that lack buzzwords.
- **G4.** Produce **honest, specific, non-hallucinated reasoning** for each of the 100 picks (Stage-4 manual review).
- **G5.** Run the ranking step within the hard compute budget (≤5 min, ≤16 GB, CPU-only, offline) and be **fully reproducible** from a clean repo (Stage-3).
- **G6.** Be **defensible in a live interview** (Stage-5) — the architecture and trade-offs must be genuinely understood and explainable.

### 3.2 Non-Goals
- **NG1.** Ranking candidates beyond the top 100 (only top 100 is submitted/scored).
- **NG2.** Calling hosted LLM APIs during the ranking step (forbidden).
- **NG3.** Building a UI beyond the required minimal sandbox demo (≤100-candidate sample).
- **NG4.** Special-casing individual honeypots by ID — the system should *naturally* avoid them via consistency checks, not hardcoded blocklists.
- **NG5.** Generalizing to arbitrary JDs — this challenge is scored on **one** JD (though a clean design should generalize).

---

## 4. Users & Stakeholders

| Stakeholder | What they need from the system |
|---|---|
| **Redrob recruiters (simulated end users)** | A ranked shortlist they can trust — high precision at the top, no obvious mis-hires or unavailable people. |
| **Hackathon evaluators (Stages 1–5)** | A valid CSV, reproducible code within constraints, honest reasoning, authentic git history, and an interviewable author. |
| **The hiring team (the JD's intent)** | Candidates who *match the meaning* of the JD, including the explicit disqualifiers and preferences. |
| **Us (the participating team)** | A maintainable, explainable pipeline we built and understand end-to-end. |

---

## 5. Functional Requirements

### 5.1 Input
- **FR1.** Ingest `candidates.jsonl` (or `.jsonl.gz`), 100,000 records, conforming to `candidate_schema.json`. Each record has: `candidate_id`, `profile`, `career_history[]`, `education[]`, `skills[]`, optional `certifications[]`/`languages[]`, and `redrob_signals` (23 behavioral signals).
- **FR2.** Ingest the job description as the ranking target.
- **FR3.** Tolerate missing/optional fields and malformed values without crashing (the dataset includes honeypots with internally inconsistent data).

### 5.2 Core ranking
- **FR4.** Compute, per candidate, a **fit score** combining at minimum:
  - **Role/title & career-trajectory fit** — is the trajectory an applied-ML-at-product-company arc? (decisive anti-keyword-stuffer signal)
  - **Semantic / skill match** to the JD's real requirements (retrieval, ranking, eval, vector search) — robust to plain-language phrasing.
  - **Experience fit** — years, recency of hands-on coding, ML-vs-services split.
  - **Disqualifier penalties** — services-only career, title-hopping cadence, CV/speech/robotics-only, research-only, stale coding.
  - **Behavioral availability modifier** — recency of activity, recruiter_response_rate, open_to_work, notice period, interview/offer history, verification flags.
  - **Honeypot / consistency penalty** — internal contradictions (tenure vs. company age, proficiency vs. duration_months, date inconsistencies).
- **FR5.** Produce a **strict total order** over the selected top 100 (unique ranks, deterministic tie-breaks).
- **FR6.** Generate a **1–2 sentence reasoning** per candidate that (a) cites specific profile facts, (b) connects to specific JD requirements, (c) acknowledges concerns where they exist, (d) hallucinates nothing, (e) varies across candidates, and (f) matches the rank's tone.

### 5.3 Output
- **FR7.** Emit a UTF-8 CSV named `<participant_id>.csv` with columns **exactly** `candidate_id,rank,score,reasoning`:
  - Exactly **100 data rows** + 1 header.
  - `rank` ∈ 1..100, each used exactly once.
  - `candidate_id` unique, matching the `CAND_XXXXXXX` pattern and existing in the pool.
  - `score` monotonically **non-increasing** as rank increases.
  - Equal scores broken by **`candidate_id` ascending**.
- **FR8.** Pass `validate_submission.py` with zero errors before any submission.

### 5.4 Reproducibility
- **FR9.** A single documented command reproduces the CSV from the candidates file:
  `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
- **FR10.** Any precomputed artifacts (embeddings, indexes) are either committed or regenerable by a documented script; pre-computation may exceed 5 min but the **ranking step must not**.

---

## 6. Evaluation & Success Metrics

### 6.1 Scoring formula (Stage 2 — automated, hidden ground truth)
```
composite = 0.50 · NDCG@10
          + 0.30 · NDCG@50
          + 0.15 · MAP
          + 0.05 · P@10
```
- **NDCG@10 (0.50)** — quality/ordering of the top 10. *Dominant term → top-10 precision and ordering matter most.*
- **NDCG@50 (0.30)** — quality of the top 50.
- **MAP (0.15)** — precision across all relevance levels.
- **P@10 (0.05)** — fraction of top-10 that are "relevant" (tier ≥ 3).
- **Tiebreaks between submissions:** higher P@5 → higher P@10 → earlier timestamp.

> **Design implication:** 80% of the score is concentrated in the top-50, half of it in the top-10. Getting the **head of the ranking exactly right** (and trap-free) dominates. A single honeypot or keyword-stuffer in the top 10 is very expensive.

### 6.2 Gating checks (pass/fail)
| Stage | Gate | Failure condition |
|---|---|---|
| 1. Format validation | `validate_submission.py` clean | Any spec violation in §5.3 |
| 2. Scoring | Composite ≥ advancement cutoff | Below cutoff |
| 3. Code reproduction + honeypot | Reproduces in sandbox; honeypot rate ≤ 10% in top 100 | Can't reproduce within limits; >10% honeypots; missing/fake repo |
| 4. Manual review | Reasoning quality (6 checks); methodology coherence; authentic git history; code quality | Failed reasoning checks; flat git history; LLM-only codebase |
| 5. Defend-your-work interview | Explain & defend architecture | Can't explain; contradicts code |

### 6.3 Our internal success criteria (proxy, since no live leaderboard)
- **0 honeypots** in our top 100 (target; hard ceiling is <10).
- **0 obvious keyword-stuffers** (wrong-title perfect-skill profiles) in our top 10.
- Manual spot-check: for any 10 random picks, reasoning is specific, honest, non-templated, and rank-consistent.
- Reproduction from a clean clone completes in **< 5 min on a 16 GB CPU machine**.

---

## 7. Constraints

### 7.1 Compute (enforced in a sandboxed Docker at Stage 3)
| Constraint | Limit |
|---|---|
| Ranking runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | **CPU only** (no GPU during ranking) |
| Network | **Off** — no hosted LLM / external API calls during ranking |
| Disk | ≤ 5 GB intermediate state |

**Implication:** an LLM call per candidate (100K calls) is infeasible and forbidden. The ranking must be a **lightweight scorer over precomputed features / compact local models / indexes**.

### 7.2 Process constraints
- **3 submissions max**; last valid submission counts. No live feedback. → Validate offline rigorously; avoid "submit and see."
- **AI tools allowed and not penalized**, but must be declared honestly; the pipeline (Stages 3–5) is designed to fail AI-only "paste-and-pray" work. Real human engineering must be evident.
- **Original work**, no collusion (per `submission_metadata.yaml` declarations).

---

## 8. Proposed Approach (reference architecture)

> This is the recommended direction; the architecture is the team's call and is what gets defended at Stage 5.

```
                    candidates.jsonl (100K)
                            │
              ┌─────────────┴─────────────┐
              │   PRECOMPUTE (offline OK)  │
              │  • parse + normalize       │
              │  • profile-text embeddings │   ← local sentence-transformer, CPU,
              │    (career descriptions)   │     precomputed & cached to disk
              │  • feature extraction      │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │  RANK STEP (≤5 min, CPU)   │
              │                            │
              │  1. Hard filters / penalties:
              │     - honeypot consistency checks (tenure vs company age,
              │       proficiency vs duration, date sanity)
              │     - services-only career, title-hop cadence,
              │       CV/speech/robotics-only, stale coding
              │  2. Component scores:
              │     - title & career-trajectory fit   (decisive)
              │     - semantic JD match (embeddings + lexical/BM25 hybrid)
              │     - experience fit (years, recency, product-vs-services)
              │     - skills trust (endorsements × duration, assessment scores)
              │  3. Behavioral availability multiplier
              │     (recency, response rate, open_to_work, notice, verifications)
              │  4. Combine → score; select top 100; deterministic tie-break
              │  5. Generate reasoning (templated-from-facts, varied, honest)
              └─────────────┬─────────────┘
                            │
                     submission.csv
```

**Key design principles:**
- **Title/career trajectory is the primary anti-keyword-stuffer signal** — a great skill list cannot rescue a "Marketing Manager" career arc.
- **Skills are *discounted by trust*** — endorsements and `duration_months` separate real expertise from claimed expertise; "expert + 0 months used" is a honeypot tell.
- **Behavioral signals are a *multiplier*, not an additive term** — an unavailable perfect-on-paper candidate is multiplied down.
- **Consistency checks are first-class**, not an afterthought — they protect the honeypot gate (a Stage-3 disqualifier).
- **Reasoning is generated from extracted facts** (no live LLM during ranking) so every claim is grounded and non-hallucinated by construction.

---

## 9. Deliverables (full submission package)

1. **Submission CSV** — `<participant_id>.csv`, top-100, validated.
2. **GitHub repository** containing:
   - `rank.py` (single-command reproduction) + full source.
   - Precompute scripts / committed artifacts (embeddings, indexes).
   - `requirements.txt` / `pyproject.toml` with pinned versions.
   - `README.md` with setup + exact reproduce command.
   - `submission_metadata.yaml` (mirrors portal metadata).
   - **Authentic git history** showing real iteration (not a single dump).
3. **Sandbox / demo link** — HuggingFace Spaces / Streamlit / Colab / Replit / Docker, runs the ranker on a ≤100-candidate sample within budget.
4. **Portal metadata** — team identity, contacts, compute summary, AI-tools declaration, ≤200-word methodology summary.

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Honeypots leak into top 100 (>10%) | **Disqualification (Stage 3)** | Dedicated consistency-check layer; assert 0 honeypots in top 100 in self-test. |
| Keyword stuffers dominate top 10 | Low NDCG@10 (half the score) | Title/career-trajectory as decisive component; skills discounted by trust. |
| Ranking step exceeds 5 min on CPU | Disqualification (Stage 3) | Precompute embeddings offline; vectorized/numpy scoring; profile runtime locally on 16 GB CPU. |
| Reasoning flagged as templated/hallucinated | Fails Stage 4 | Generate from extracted facts; enforce variation; acknowledge concerns; spot-check 10 random rows. |
| Flat git history / can't defend work | Fails Stage 4/5 | Commit incrementally with meaningful messages; ensure every author understands the design. |
| Over-fitting to assumed ground truth | Low real score | Ground scoring logic in JD text + data exploration, not guesses; sanity-check distributions. |
| Only 3 submissions, no feedback | Wasted attempts | Validate offline; treat submission 1 as a sound baseline, not a probe. |

---

## 11. Open Questions

- **OQ1.** Exact relevance-tier definition in the ground truth (0–5?) and how "tier 3+" relevance maps to JD fit — inferred from JD + signals doc, not given explicitly.
- **OQ2.** Precise honeypot construction rules — we infer them from the two stated examples (tenure vs. company age; expert proficiency vs. 0 months used) plus general internal-consistency checks.
- **OQ3.** Distribution of true fits in the pool — JD says "expect few matches in 100K; 10 great > 1000 maybes." Need data exploration to estimate how deep genuine relevance goes (affects ranks ~50–100).
- **OQ4.** Whether location/relocation should hard-gate or soft-penalize (JD is "flexible" but Pune/Noida-preferred).

---

## 12. Milestones (suggested)

| Phase | Output |
|---|---|
| **M0 — Understand** ✅ | Docs read, schema understood, this PRD. |
| **M1 — Explore** | Data-distribution analysis; identify/confirm trap subpopulations; spot example honeypots & stuffers. |
| **M2 — Baseline** | End-to-end `rank.py` producing a valid CSV (simple weighted score). Passes validator. |
| **M3 — Trap defense** | Consistency checks, title/career trajectory, behavioral multiplier. Honeypots driven out of top 100. |
| **M4 — Reasoning + tune** | Fact-grounded reasoning generator; weight tuning via held-out qualitative checks. |
| **M5 — Productionize** | Repo, README, requirements, git history, sandbox demo, metadata. Reproduce within budget. |
| **M6 — Submit** | Final validated CSV + package; submit (≤3 attempts). |

---

*Source documents: `job_description`, `submission_spec`, `redrob_signals_doc`, `candidate_schema.json`, `validate_submission.py`, `sample_submission.csv` (Redrob Hackathon participant bundle).*
