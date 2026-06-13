# Redrob Candidate Ranker — Senior AI Engineer JD

Ranks the top-100 candidates from a 100K-profile pool for the *Senior AI Engineer
— Founding Team* job description, under the hackathon's hard constraints
(CPU-only, offline, ≤5 min, ≤16 GB for the ranking step).

**Team Token Bandits** · Live sandbox: <https://huggingface.co/spaces/nishanroy561/redrob-ranker>

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

## Setup

- **Python 3.11+** (developed on 3.12).
- Install deps: `pip install -r requirements.txt` — the ranking step (`rank.py`)
  needs only **numpy**; `sentence-transformers`/`torch` are precompute-only.
- The embeddings in `artifacts/` are tracked with **git-lfs**. Run `git lfs install`
  once, then clone (or `git lfs pull` in an existing clone) so the `.npy` files
  download — otherwise you only get LFS pointer stubs.

## Reproduce

`candidates.jsonl` is the organizer-provided 100K pool (not committed, ~487 MB) —
place it at the repo root or pass its path. The `artifacts/` (precomputed
embeddings) **are** committed, so reproduction is a single offline command:

```bash
# Stage-3 reproduce command — CPU-only, no network, < 5 min.
# Reads the bundled artifacts/ and produces the exact submitted top-100.
python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv

# validate before uploading
python validate_submission.py submission.csv
```

Regenerating the embeddings from scratch is **optional** (GPU + network) — only
needed if you want to rebuild `artifacts/` yourself:

```bash
python precompute.py --candidates ./candidates.jsonl --out ./artifacts
```

> Without `artifacts/`, `rank.py` falls back to a lexical, no-embedding baseline —
> a *different* result, useful only for a quick local smoke test, not the submission.

## Files

| File | Role |
|---|---|
| `jd_config.py` | JD encoded as signals: titles, keywords, disqualifiers, **weights** (tune here) |
| `features.py` | Model-free feature extraction + honeypot consistency checks |
| `precompute.py` | GPU embedding of candidate narratives + JD anchor → `artifacts/` |
| `rank.py` | CPU-only offline ranker → `submission.csv` |
| `reasoning.py` | Fact-grounded, non-hallucinated reasoning strings |
| `validate_submission.py` | Official validator (schema, ranks, tie-breaks) |
| `notebooks/redrob-ai-challenge-precompute.ipynb` | Kaggle **GPU** notebook: embed 100K candidates → `artifacts/` |
| `notebooks/redrob-rank.ipynb` | Kaggle **CPU** notebook: offline rank + validator (Stage-3 proof) |
| `space/` | HuggingFace Spaces sandbox (Streamlit, ≤100-candidate sample) |

## Performance

Ranking step on **Kaggle CPU, offline: ~113 s** for the full 100K pool (under the
5-min cap), peak RAM **< 4 GB** (16 GB budget). Top-100 output: **0 honeypots**,
official validator clean, **93/100 unique scores**, **100/100 unique reasonings**.
A no-embedding lexical fallback is available for quick local testing (see Reproduce).

## Status

- [x] GPU precompute on Kaggle → `artifacts/` (candidate + JD embeddings)
- [x] Offline CPU rank reproduced on Kaggle with **Internet OFF** — validator clean
- [x] HuggingFace Spaces sandbox live (≤100-candidate sample)
- [x] `submission_metadata.yaml` complete
