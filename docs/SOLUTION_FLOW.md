# Solution Flow & Platform Plan
## Redrob Candidate Ranking Challenge — Cloud-First (no local setup)

> Goal: build, precompute, reproduce, and demo the ranker entirely in the cloud, while still satisfying the hard Stage-3 rule that the **ranking step** runs **CPU-only, offline, ≤5 min, ≤16 GB**.

---

## 0. The one constraint that drives every platform choice

The challenge splits your pipeline into **two phases with different rules**:

| Phase | What it does | Allowed to use | Time budget |
|---|---|---|---|
| **PRECOMPUTE** (offline) | Generate embeddings, build indexes, extract features | **GPU OK, internet OK, can take long** | No hard limit |
| **RANK STEP** (`rank.py`) | Load precomputed artifacts → score → write `submission.csv` | **CPU only, NO network, ≤16 GB** | **≤5 min** for 100K |

So the trick is: **do the heavy/slow stuff (embeddings) in precompute on a GPU box, save the result as a file, and make `rank.py` just load that file and do fast CPU math.** This is exactly what lets you use Kaggle's free GPU without violating the CPU-only ranking rule.

---

## 1. Recommended platform stack

| Job | Platform | Why |
|---|---|---|
| **Dev + Precompute (embeddings, features)** | **Kaggle Notebooks** ✅ | Free GPU (~30 hrs/week), 16–30 GB RAM, persistent **Datasets** to store the 487 MB pool + embedding artifacts, can toggle internet **off** to simulate the no-network rule. Best fit overall. |
| **Version control + reproduction repo** | **GitHub** | Required deliverable. Kaggle can push to it; holds `rank.py`, `requirements.txt`, README, committed artifacts. |
| **Sandbox / demo link** (required) | **Hugging Face Spaces** (Gradio or Streamlit) | Free **CPU** tier, persistent public URL, runs the ranker on a ≤100-candidate sample — exactly what Section 10.5 asks for. |
| **Stage-3 reproduction sanity check** | **Kaggle (internet OFF, CPU-only session)** | Lets you reproduce the 5-min/16 GB/offline run *before* submitting — the same thing organizers do in Docker. |

### Why Kaggle over Colab
- **Persistent Datasets** — upload `candidates.jsonl` once (private dataset); your embedding artifacts persist between sessions. Colab loses everything unless you wire up Drive.
- **Longer, more stable sessions** and a clean **internet on/off toggle** (Settings → Internet) — perfect for testing the offline constraint.
- **GitHub + Dataset versioning** built in.
- Colab is a fine *backup*; HF Spaces is **not** a dev environment, it's only the demo.

---

## 2. End-to-end flow

```
 ┌────────────────────────────────────────────────────────────────────┐
 │ STEP 1 — Upload data (once)                                         │
 │  Kaggle → Create Dataset → upload candidates.jsonl (+ JD, schema)   │
 │  → private dataset "redrob-candidates"                              │
 └───────────────┬────────────────────────────────────────────────────┘
                 │
 ┌───────────────▼────────────────────────────────────────────────────┐
 │ STEP 2 — PRECOMPUTE  (Kaggle GPU notebook, internet ON)            │
 │  • parse 100K profiles, normalize fields                           │
 │  • build profile text (summary + career descriptions + skills)     │
 │  • embed with a LOCAL sentence-transformer (e.g. bge-small / e5)   │
 │  • embed the JD once                                                │
 │  • save: embeddings.npy + candidate_index.parquet                  │
 │    → "Save Version" → output becomes a Kaggle Dataset artifact     │
 └───────────────┬────────────────────────────────────────────────────┘
                 │   (artifacts committed to GitHub too, or regen script)
 ┌───────────────▼────────────────────────────────────────────────────┐
 │ STEP 3 — RANK STEP  (Kaggle CPU notebook, internet OFF) = rank.py  │
 │  load embeddings.npy + features  →  vectorized scoring (numpy):    │
 │    title/career-trajectory fit · semantic match · experience fit   │
 │    · disqualifier penalties · behavioral multiplier · honeypot     │
 │    consistency checks  →  top 100  →  reasoning  →  submission.csv  │
 │  Must finish ≤5 min on CPU, no network.                            │
 └───────────────┬────────────────────────────────────────────────────┘
                 │
 ┌───────────────▼───────────────┐   ┌───────────────────────────────┐
 │ STEP 4 — Validate              │   │ STEP 5 — Demo sandbox         │
 │  python validate_submission.py │   │  HF Spaces (Gradio/Streamlit) │
 │  <id>.csv  → 0 errors          │   │  upload ≤100 sample → ranked  │
 └───────────────┬───────────────┘   │  CSV, CPU, <5 min, public URL │
                 │                    └───────────────────────────────┘
 ┌───────────────▼────────────────────────────────────────────────────┐
 │ STEP 6 — Package & submit                                          │
 │  GitHub repo (rank.py, requirements.txt, README, artifacts,        │
 │  submission_metadata.yaml, real git history)  +  CSV  +  sandbox   │
 │  link  +  portal metadata   → submit (≤3 attempts)                 │
 └────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-step (what to actually click/run)

### Step 1 — Put the data on Kaggle
1. kaggle.com → **Datasets → New Dataset**.
2. Upload `candidates.jsonl` (you can gzip it to ~52 MB first), plus `job_description`, `candidate_schema.json`. Keep it **private**.
3. Name it e.g. `redrob-candidates`.

### Step 2 — Precompute notebook (GPU, internet ON)
- New Notebook → **Add Data** → your dataset → **Accelerator: GPU** → **Internet: On** (only needed here, to download the embedding model weights once).
- Generate and **save** `embeddings.npy` + a `candidate_features.parquet` (years, title tokens, company-size, signals, consistency flags).
- **Save Version** → outputs become a reusable Kaggle Dataset. Also download artifacts to commit to GitHub.

### Step 3 — Rank notebook = `rank.py` (CPU, internet OFF)
- New Notebook → add **both** your candidates dataset and your precompute-output dataset → **Accelerator: None (CPU)** → **Internet: Off**.
- This is the real test: it must load artifacts and produce `submission.csv` in **≤5 min on CPU with no network**. If it passes here, it'll pass the organizers' Docker.
- Mirror this logic into a clean `rank.py` for the repo (`--candidates`, `--out` args).

### Step 4 — Validate
- Run `validate_submission.py <participant_id>.csv` in the same notebook. Fix any of the common rejections (row count, unique ranks, non-increasing score, tie-break by candidate_id).

### Step 5 — Sandbox demo (Hugging Face Spaces)
- Create a free **Space** → SDK **Gradio** (simplest) or **Streamlit** → CPU basic (free).
- App: upload a ≤100-candidate JSON → runs the *same* ranking code → returns ranked CSV / table. Commit `app.py` + `requirements.txt` + a tiny `bge`/`e5` model (or precomputed sample embeddings) so it stays within budget.
- The public URL is your required **sandbox link**.

### Step 6 — Repo + submit
- GitHub repo with: `rank.py`, precompute script, `requirements.txt` (pinned), `README.md` (the single reproduce command), committed/regenerable artifacts, `submission_metadata.yaml`.
- **Commit incrementally** (real git history matters at Stage 4).
- Submit CSV + GitHub URL + HF Spaces URL + portal metadata. ≤3 attempts — validate offline first.

---

## 4. Key gotchas for the cloud setup

- **GPU only in precompute.** The metadata field `uses_gpu_for_inference: false` must be true-to-reality — never let the *ranking step* depend on a GPU. Embeddings are precomputed; `rank.py` just reads `.npy`.
- **No network in `rank.py`.** Don't `from_pretrained()` a model inside the ranking step (that hits the network). Load only local files. Test with Kaggle Internet **off**.
- **Model weights must be local for the offline run.** Download the sentence-transformer once during precompute and cache/commit it (or precompute all embeddings so the rank step needs no model at all — cleanest).
- **5-minute budget is for 100K, on CPU.** Keep scoring vectorized (numpy/pandas), avoid per-candidate Python loops and avoid per-candidate model calls.
- **HF Spaces free tier is CPU** — good, it proves CPU-only reproducibility, but keep the demo to a small sample.
- **Set `has_network_during_ranking: false` and `pre_computation_required: true`** honestly in `submission_metadata.yaml`.

---

## 5. TL;DR recommendation

**Use Kaggle as your build + precompute + reproduction environment, GitHub for the repo, and Hugging Face Spaces for the required sandbox demo.** Put the slow GPU work (embeddings) in a Kaggle precompute notebook, save the artifacts, and make `rank.py` a fast CPU-only, offline loader+scorer — tested in a Kaggle CPU/internet-off session that mirrors the Stage-3 Docker exactly. Colab is an acceptable backup for dev, but Kaggle's persistent Datasets and internet toggle make it the better primary.
