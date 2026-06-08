"""
Redrob Candidate Ranker — HuggingFace Spaces sandbox (Streamlit)
===============================================================
Required Stage-1 deliverable: a hosted environment where the ranking system runs
on a SMALL candidate sample (<=100) and returns a ranked CSV.

An uploaded sample contains arbitrary candidates, so we embed it on the fly
(cheap for <=100). The SCORING is the exact same shared code as the competition
rank.py (features.combine_score), pulled from the GitHub repo at startup so the
demo can never drift from the real ranker.
"""
import json, os, subprocess, sys

import numpy as np
import pandas as pd
import streamlit as st

# ---- pull the real ranking modules from the repo (single source of truth) ----
REPO_URL = "https://github.com/nishanroy561/Data-AI-Challenge.git"
REPO_DIR = "Data-AI-Challenge"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import jd_config as J        # noqa: E402
import features as F         # noqa: E402
import reasoning as R        # noqa: E402


@st.cache_resource
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")   # CPU on the free tier


def load_candidates(uploaded, pasted):
    raw = None
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
    elif pasted and pasted.strip():
        raw = pasted
    else:
        sample = os.path.join(REPO_DIR, "sample_candidates.json")
        if os.path.exists(sample):
            raw = open(sample, "r", encoding="utf-8").read()
    if not raw:
        return []
    raw = raw.strip()
    cands = json.loads(raw) if raw.startswith("[") \
        else [json.loads(l) for l in raw.splitlines() if l.strip()]
    return cands[:100]


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯")
st.title("Redrob Candidate Ranker — Senior AI Engineer JD")
st.markdown(
    "Upload up to **100** candidate profiles (`.jsonl` or `.json` matching "
    "`candidate_schema.json`) and get the ranked shortlist with reasoning. "
    "Scoring is the **same code** as the competition `rank.py` "
    "(`features.combine_score`); embeddings are computed on the fly for the "
    "uploaded sample. Leave inputs empty to rank the bundled 50-candidate sample."
)

uploaded = st.file_uploader("Candidates (.jsonl / .json)", type=["jsonl", "json"])
pasted = st.text_area("…or paste JSON / JSONL", height=140,
                      placeholder='{"candidate_id": "CAND_0000001", "profile": {...}, ...}')
top_k = st.slider("Top-K to return", 1, 100, 20)

if st.button("Rank candidates", type="primary"):
    cands = load_candidates(uploaded, pasted)
    if not cands:
        st.error("No candidates provided. Upload a .jsonl/.json or paste JSON.")
        st.stop()

    with st.spinner(f"Embedding and scoring {len(cands)} candidates…"):
        model = get_model()
        texts = [F.profile_text(c) for c in cands]
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        jd = model.encode([J.JD_ANCHOR_TEXT], convert_to_numpy=True,
                          normalize_embeddings=True)[0]
        sem = ((emb @ jd) + 1.0) / 2.0                      # -> [0,1], same as rank.py

        scored = []
        for c, s in zip(cands, sem):
            score, comps = F.combine_score(c, float(s))
            scored.append((round(score, 4), c, comps))
        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        scored = scored[: int(top_k)]

        rows = [{
            "candidate_id": c["candidate_id"],
            "rank": i,
            "score": f"{sc:.4f}",
            "reasoning": R.make_reasoning(c, comps, i),
        } for i, (sc, c, comps) in enumerate(scored, start=1)]
        df = pd.DataFrame(rows)

    st.success(f"Ranked {len(df)} candidates.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download submission.csv",
                       df.to_csv(index=False).encode("utf-8"),
                       "submission.csv", "text/csv")
