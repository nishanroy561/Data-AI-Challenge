"""
Redrob Candidate Ranker — HuggingFace Spaces sandbox
====================================================
Required Stage-1 deliverable: a hosted environment where the ranking system runs
on a SMALL candidate sample (<=100) and returns a ranked CSV.

Because an uploaded sample contains arbitrary candidates, we cannot reuse the
precomputed 100K embeddings — we embed the sample on the fly (cheap for <=100).
The SCORING is the exact same shared code as the competition rank.py
(features.combine_score), pulled from the GitHub repo at startup so the demo can
never drift from the real ranker.
"""
import json, os, subprocess, sys, tempfile

import numpy as np
import pandas as pd

# --- Workaround for a gradio_client bug (TypeError: argument of type 'bool' is
# not iterable). When a component schema has `additionalProperties: true`, the
# schema value is a bool and json_schema_to_python_type / get_type crash on it.
# Patch both to tolerate booleans. Must run before gradio builds its API info. ---
import gradio_client.utils as _gcu

_orig_jstpt = _gcu._json_schema_to_python_type
def _safe_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jstpt(schema, defs)
_gcu._json_schema_to_python_type = _safe_jstpt

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

import gradio as gr

# ---- pull the real ranking modules from the repo (single source of truth) ----
REPO_URL = "https://github.com/nishanroy561/Data-AI-Challenge.git"
REPO_DIR = "Data-AI-Challenge"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)

import jd_config as J          # noqa: E402
import features as F           # noqa: E402
import reasoning as R          # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)   # CPU on the free Space tier
    return _model


def _load_candidates(file_obj, pasted_text):
    """Accept an uploaded .jsonl/.json OR pasted JSON; return list of candidates."""
    raw = None
    if file_obj is not None:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            raw = f.read()
    elif pasted_text and pasted_text.strip():
        raw = pasted_text
    else:
        # fall back to the sample bundled in the repo, if present
        sample = os.path.join(REPO_DIR, "sample_candidates.json")
        if os.path.exists(sample):
            raw = open(sample, "r", encoding="utf-8").read()
        else:
            return []

    raw = raw.strip()
    if raw.startswith("["):
        cands = json.loads(raw)                       # JSON array
    else:
        cands = [json.loads(l) for l in raw.splitlines() if l.strip()]  # JSONL
    return cands[:100]                                # sandbox cap


def rank(file_obj, pasted_text, top_k):
    cands = _load_candidates(file_obj, pasted_text)
    if not cands:
        return None, "No candidates provided. Upload a .jsonl/.json or paste JSON."

    # 1. embed the sample + JD anchor (on the fly; cheap for <=100)
    model = get_model()
    texts = [F.profile_text(c) for c in cands]
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    jd = model.encode([J.JD_ANCHOR_TEXT], convert_to_numpy=True,
                      normalize_embeddings=True)[0]
    sem = ((emb @ jd) + 1.0) / 2.0                    # -> [0,1], same as rank.py

    # 2. score with the SHARED competition scorer
    scored = []
    for c, s in zip(cands, sem):
        score, comps = F.combine_score(c, float(s))
        scored.append((round(score, 4), c, comps))

    # 3. sort (score desc, candidate_id asc) and build the ranked table
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    scored = scored[: int(top_k)]
    out_rows = []
    for rank_i, (score, c, comps) in enumerate(scored, start=1):
        out_rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank_i,
            "score": f"{score:.4f}",
            "reasoning": R.make_reasoning(c, comps, rank_i),
        })

    df = pd.DataFrame(out_rows)
    out_path = os.path.join(tempfile.gettempdir(), "submission.csv")
    df.to_csv(out_path, index=False)
    return df, out_path


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob Candidate Ranker — Senior AI Engineer JD\n"
        "Upload up to 100 candidate profiles (`.jsonl` or `.json`) and get the "
        "ranked shortlist with reasoning. Scoring is the **same code** as the "
        "competition `rank.py` (`features.combine_score`); embeddings are computed "
        "on the fly for the uploaded sample.\n\n"
        "Leave inputs empty to rank the bundled 50-candidate sample."
    )
    with gr.Row():
        file_in = gr.File(label="Candidates (.jsonl / .json)", file_types=[".jsonl", ".json"])
        text_in = gr.Textbox(label="…or paste JSON / JSONL", lines=6,
                             placeholder='{"candidate_id": "CAND_0000001", "profile": {...}, ...}')
    top_k = gr.Slider(1, 100, value=20, step=1, label="Top-K to return")
    btn = gr.Button("Rank candidates", variant="primary")
    table = gr.Dataframe(label="Ranked shortlist", wrap=True)
    csv_out = gr.File(label="Download submission.csv")
    btn.click(rank, inputs=[file_in, text_in, top_k], outputs=[table, csv_out])

if __name__ == "__main__":
    # show_api=False avoids the gradio_client schema-introspection bug
    demo.launch(show_api=False)
