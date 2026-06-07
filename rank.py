"""
rank.py  —  RANKING phase (CPU-only, OFFLINE, <=5 min, <=16 GB)
===============================================================
Loads precomputed embeddings + reads candidates.jsonl, scores every candidate
with fast vectorised math, and writes the top-100 submission CSV.

NO model, NO network. Reproduce command (also goes in README):

    python rank.py --candidates ./candidates.jsonl \
                   --artifacts ./artifacts \
                   --out ./submission.csv

Final score = base_fit  x  behavioral_multiplier  x  honeypot_gate
  base_fit = W_TITLE_CAREER*title + W_SEMANTIC*sem + W_EXPERIENCE*exp
             + W_SKILLS_TRUST*skills  -  penalties   (clamped to [0,1])
"""
import argparse, csv, gzip, json, os, time
import numpy as np

import features as F
import jd_config as J
import reasoning as R


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def load_pool(path):
    cands = []
    with _open(path) as f:
        for line in f:
            if line.strip():
                cands.append(json.loads(line))
    return cands


def load_semantic(artifacts_dir, ids, pool):
    """Per-id semantic similarity to the JD (0..1), aligned to `ids`.

    Uses precomputed embeddings if present; otherwise falls back to a lexical
    work-keyword proxy so the pipeline runs end-to-end before precompute.py.
    """
    emb_path = os.path.join(artifacts_dir, "cand_embeddings.npy")
    if os.path.exists(emb_path):
        cand_ids = np.load(os.path.join(artifacts_dir, "candidate_ids.npy"))
        emb = np.load(emb_path)                                            # (N,D) normed
        jd = np.load(os.path.join(artifacts_dir, "jd_embedding.npy"))      # (D,) normed
        cos = ((emb @ jd) + 1.0) / 2.0                                     # -> [0,1]
        lookup = {cid: i for i, cid in enumerate(cand_ids)}
        return np.array([cos[lookup[i]] if i in lookup else 0.0 for i in ids])

    print("  [fallback] no embeddings found -> lexical work-keyword proxy")
    # Distinct-keyword count over a wider vocabulary -> more continuous than hits/5.
    work_kw = list(dict.fromkeys(J.WORK_RETRIEVAL + J.WORK_EVAL + J.WORK_ML_PROD))
    out = np.zeros(len(ids))
    for i, c in enumerate(pool):
        hits = F.kw_count(F.profile_text(c), work_kw)   # word-boundary, not substring
        out[i] = min(1.0, hits / 10.0)   # saturates less; embeddings differentiate better
    return out


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    print("Loading candidate pool...")
    pool = load_pool(args.candidates)
    ids = [c["candidate_id"] for c in pool]
    print(f"  {len(pool)} candidates ({time.time()-t0:.1f}s)")

    print("Loading semantic similarity...")
    sem = load_semantic(args.artifacts, ids, pool)

    print("Scoring...")
    rows = []
    for i, c in enumerate(pool):
        score, comps = F.combine_score(c, float(sem[i]))
        rows.append({
            "candidate_id": c["candidate_id"],
            # Round to the SAME precision we emit, so the sort tie-break matches
            # exactly what the validator sees in the CSV.
            "score": round(score, 4),
            "comps": comps,
            "_c": c,
        })

    # Sort: score desc, then candidate_id asc (matches validator tie-break rule).
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = rows[:args.top]

    print(f"Writing {len(top)} rows -> {args.out}")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            reason = R.make_reasoning(r["_c"], r["comps"], rank)
            w.writerow([r["candidate_id"], rank, f"{r['score']:.4f}", reason])

    # Quick self-checks (never hurts; cheap)
    hp_in_top = sum(1 for r in top if r["comps"]["honeypot_flags"])
    print(f"Done in {time.time()-t0:.1f}s. Honeypot-flagged in top {args.top}: {hp_in_top}")
    if hp_in_top:
        print("  WARNING: investigate flagged profiles before submitting.")


if __name__ == "__main__":
    main()
