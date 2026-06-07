"""
precompute.py  —  OFFLINE phase (GPU + internet allowed)
========================================================
Embeds every candidate's TRUTHFUL narrative (summary + career descriptions, NOT
the skills list) and the JD anchor, then saves them so rank.py needs no model
and no network.

Run this once on Kaggle with Accelerator=GPU, Internet=ON:

    python precompute.py --candidates ./candidates.jsonl --out ./artifacts

Outputs (commit these to the repo, or document this script as the regenerator):
    artifacts/candidate_ids.npy        (N,)   str   row order
    artifacts/cand_embeddings.npy      (N, D) float32  L2-normalised
    artifacts/jd_embedding.npy         (D,)   float32  L2-normalised
    artifacts/model.txt                model name used (for provenance)
"""
import argparse, gzip, json, os
import numpy as np

import features as F
import jd_config as J

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-d, strong + tiny; CPU-loadable too


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def load_candidates(path):
    ids, texts = [], []
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            ids.append(c["candidate_id"])
            texts.append(F.profile_text(c))
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="./artifacts")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sentence_transformers import SentenceTransformer  # precompute-only import
    model = SentenceTransformer(MODEL_NAME)   # auto-uses GPU on Kaggle if present

    print("Loading candidates...")
    ids, texts = load_candidates(args.candidates)
    print(f"  {len(ids)} candidates")

    print("Embedding candidates (this is the slow part; GPU recommended)...")
    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    jd = model.encode([J.JD_ANCHOR_TEXT], normalize_embeddings=True,
                      convert_to_numpy=True)[0]

    np.save(os.path.join(args.out, "candidate_ids.npy"), np.array(ids))
    np.save(os.path.join(args.out, "cand_embeddings.npy"), emb.astype(np.float32))
    np.save(os.path.join(args.out, "jd_embedding.npy"), jd.astype(np.float32))
    with open(os.path.join(args.out, "model.txt"), "w") as fh:
        fh.write(MODEL_NAME + "\n")
    print(f"Saved artifacts to {args.out}  (emb shape {emb.shape})")


if __name__ == "__main__":
    main()
