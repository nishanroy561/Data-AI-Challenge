# ==========================================================================
# Kaggle notebook: PRECOMPUTE embeddings  (Accelerator = GPU, Internet = ON)
# ==========================================================================
# Paste each "# %% [cell]" block into its own Kaggle cell, OR upload this repo
# as a Kaggle Dataset and run `!python precompute.py ...`.
#
# Settings (right sidebar):
#   - Accelerator:  GPU T4 x2  (only needed here, for embedding speed)
#   - Internet:     ON         (only to download the model weights once)
#   - Add Data:     your "redrob-candidates" dataset
# --------------------------------------------------------------------------

# %% [cell 1] install (Kaggle usually has these; pin to be safe)
!pip install -q sentence-transformers==3.0.1

# %% [cell 2] make repo modules importable
import sys, os
# If you uploaded this repo as a dataset, point to it; else clone your GitHub:
# !git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
sys.path.append("/kaggle/working/redrob-ranker")   # adjust to where the .py files live
import jd_config as J
import features as F

CANDIDATES = "/kaggle/input/redrob-candidates/candidates.jsonl"   # adjust path
OUT = "/kaggle/working/artifacts"
os.makedirs(OUT, exist_ok=True)

# %% [cell 3] load + build truthful narrative text (summary + career, NO skills)
import json, numpy as np
ids, texts = [], []
with open(CANDIDATES, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            ids.append(c["candidate_id"])
            texts.append(F.profile_text(c))
print(len(ids), "candidates")
print("sample text:", texts[0][:200])

# %% [cell 4] embed on GPU
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")   # uses GPU automatically
emb = model.encode(texts, batch_size=512, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)
jd = model.encode([J.JD_ANCHOR_TEXT], normalize_embeddings=True,
                  convert_to_numpy=True)[0]
print("embeddings:", emb.shape)

# %% [cell 5] save artifacts -> "Save Version" turns /kaggle/working into a Dataset
np.save(f"{OUT}/candidate_ids.npy", np.array(ids))
np.save(f"{OUT}/cand_embeddings.npy", emb.astype(np.float32))
np.save(f"{OUT}/jd_embedding.npy", jd.astype(np.float32))
with open(f"{OUT}/model.txt", "w") as fh:
    fh.write("BAAI/bge-small-en-v1.5\n")
print("Saved. Now: Save Version (commit), or download artifacts/ into the GitHub repo.")

# --------------------------------------------------------------------------
# NEXT NOTEBOOK = RANK STEP  (Accelerator = None/CPU, Internet = OFF):
#     !python rank.py --candidates {CANDIDATES} --artifacts ./artifacts --out submission.csv
#     !python validate_submission.py submission.csv
# This second run MUST stay under 5 min on CPU with no network -- that is the
# real Stage-3 reproduction test.
# --------------------------------------------------------------------------
