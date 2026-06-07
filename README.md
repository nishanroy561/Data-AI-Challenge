# Redrob Candidate Ranker — Senior AI Engineer JD

Ranks the top-100 candidates from a 100K-profile pool for the *Senior AI Engineer
— Founding Team* job description, under the hackathon's hard constraints
(CPU-only, offline, ≤5 min, ≤16 GB for the ranking step).

## Core idea

The dataset is adversarial: keyword-stuffers list every AI buzzword (RAG, Pinecone,
FAISS, LangChain) while their **title and career history** reveal a non-fit (e.g. a
Graphic Designer). So this ranker treats the **skills list as untrusted** and reads
the truth from **title + career-history descriptions**, with behavioral signals as a
multiplier and consistency checks to bury honeypots.

```
score = base_fit  ×  behavioral_multiplier  ×  honeypot_gate

base_fit = 0.35·title_career_fit      (decisive anti-stuffer signal)
         + 0.30·semantic_fit          (career-description embedding vs JD anchor)
         + 0.20·experience_fit        (6–8 yr ideal, recency)
         + 0.15·skills_trust          (AI skills discounted by endorsements×duration)
         − penalties                  (services-only, job-hopping, CV/speech-only, research-only)

behavioral_multiplier ∈ [0.30, 1.0]   (last-active, response rate, open-to-work, notice)
honeypot_gate         = 0.02 if any internal inconsistency, else 1.0
```

## Two-phase design (this is how we satisfy the constraints)

| Phase | File | Compute | Network | Time |
|---|---|---|---|---|
| **Precompute** | `precompute.py` | GPU OK | OK | no limit |
| **Rank step** | `rank.py` | **CPU only** | **OFF** | **≤5 min** |

The slow embedding work happens once in precompute and is saved to `artifacts/`.
`rank.py` loads those arrays and does fast vectorised CPU scoring — **no model,
no network**.

## Reproduce

```bash
# 1. (offline phase, GPU box e.g. Kaggle) generate embeddings
python precompute.py --candidates ./candidates.jsonl --out ./artifacts

# 2. (CPU, no network) produce the submission CSV  <-- the Stage-3 reproduce command
python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv

# 3. validate before uploading
python validate_submission.py submission.csv
```

> `rank.py` also runs **without** `artifacts/` using a lexical work-keyword
> fallback (a no-embedding baseline) — handy for quick local testing, but commit
> the embeddings for the real submission.

## Files

| File | Role |
|---|---|
| `jd_config.py` | JD encoded as signals: titles, keywords, disqualifiers, **weights** (tune here) |
| `features.py` | Model-free feature extraction + honeypot consistency checks |
| `precompute.py` | GPU embedding of candidate narratives + JD anchor → `artifacts/` |
| `rank.py` | CPU-only offline ranker → `submission.csv` |
| `reasoning.py` | Fact-grounded, non-hallucinated reasoning strings |
| `notebooks/kaggle_precompute.py` | Kaggle cell-by-cell precompute walkthrough |

## Performance (local CPU, lexical fallback)

100K candidates scored and ranked in **~23 s**, **0 honeypots** in top 100,
validator clean. (Embeddings improve top-of-ranking differentiation.)

## Status / TODO

- [ ] Run `precompute.py` on Kaggle GPU; commit `artifacts/`.
- [ ] Tune weights in `jd_config.py` against held-out qualitative checks (M4).
- [ ] Stand up the HuggingFace Spaces sandbox (≤100-candidate sample).
- [ ] Fill in `submission_metadata.yaml`.
