# Submission Checklist — Redrob Candidate Ranking Challenge

Status legend: ✅ done · 🔄 in progress · ⬜ to do

---

## Phase 0 — Understand the challenge ✅
- [x] Read `job_description`, `submission_spec`, `redrob_signals_doc`, `candidate_schema.json`
- [x] Understand scoring: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`
- [x] Understand constraints: CPU-only, offline, ≤5 min, ≤16 GB for the **ranking step**
- [x] Understand traps: keyword-stuffers, plain-language Tier-5s, behavioral twins, ~80 honeypots
- [x] Note the gates: honeypot rate >10% in top-100 = DQ; 3 submissions max; no live leaderboard
- [x] Write `docs/PRD.md` (problem + design) and `docs/SOLUTION_FLOW.md` (platform plan)

## Phase 1 — Build the ranker ✅
- [x] `jd_config.py` — JD as signals (titles, work keywords, disqualifiers, weights)
- [x] `features.py` — model-free features + honeypot consistency checks
- [x] `precompute.py` — GPU embeddings of candidate narratives + JD anchor
- [x] `rank.py` — CPU-only, offline ranker → `submission.csv`
- [x] `reasoning.py` — fact-grounded, varied reasoning
- [x] Anti-stuffer design: title/career drives the score, skills list treated as untrusted
- [x] Plain-language Tier-5 fix: career-history evidence lifts adjacent titles (Search Eng, SWE(ML))
- [x] Reasoning fix: 100/100 unique, real companies + named skills + honest concerns

## Phase 2 — Run on Kaggle ✅
- [x] Upload `candidates.jsonl(.gz)` as Kaggle Dataset `redrob-candidates`
- [x] Precompute notebook (GPU, internet ON) → `artifacts/` (embeddings 100000×384)
- [x] Save Version with **output saved** (Quick Save → "Always save output")
- [x] Verify artifacts: shape `(100000, 384)`, L2-normalized, ~147 MB
- [x] Rank notebook (CPU) → `submission.csv` in ~42s (budget 300s)
- [x] Official validator: **"Submission is valid."**
- [x] 0 honeypots in top 100, stuffer CAND_0000083 excluded, 93 unique scores
- [ ] **Download `submission.csv` from the Output panel** ⬜

## Phase 3 — Quality verification 🔄
- [x] Top-10 are elite (6-8y core titles with real ranking/recsys evidence)
- [x] Plain-language Tier-5s present (11 adjacent titles in top 100)
- [x] Reasoning: specific facts, JD connection, honest concerns, variation, rank-tone
- [ ] Spot-check the 5 "AI Research Engineer" picks are production (not research-only) ⬜
- [ ] Manually read 10 random reasonings for hallucination (every claim in the profile) ⬜
- [ ] Confirm score is monotonically non-increasing + tie-breaks by candidate_id asc (validator covers this) ✅

## Phase 4 — Reproducibility / Stage-3 readiness 🔄
- [x] GitHub repo: https://github.com/nishanroy561/Data-AI-Challenge
- [x] `requirements.txt` present (pin versions before final)
- [x] `README.md` with single reproduce command
- [x] `validate_submission.py` committed
- [x] Real git history (incremental commits, not one dump)
- [ ] **Offline dry-run**: flip rank notebook to Internet OFF, reproduce CSV (mirrors Stage-3 Docker) ⬜
- [ ] Decide artifacts handling: commit `artifacts/` via git-lfs OR document `precompute.py` as regenerator ⬜
- [ ] `pip freeze` the exact versions used on Kaggle into `requirements.txt` ⬜

## Phase 5 — Sandbox demo (required) ⬜
- [ ] Build `app.py` (Gradio or Streamlit) that ranks a ≤100-candidate sample
- [ ] Deploy to HuggingFace Spaces (free CPU tier)
- [ ] Verify it runs end-to-end within budget and returns a ranked CSV
- [ ] Record the public Space URL for the portal

## Phase 6 — Metadata & portal prep ⬜
- [ ] Fill `submission_metadata.yaml`:
  - [ ] team_name, primary_contact (name/email/phone), team_members
  - [ ] github_repo ✅, sandbox_link ⬜
  - [ ] compute env (Kaggle, CPU, Python 3.12, no GPU/network during ranking)
  - [ ] `uses_gpu_for_inference: false`, `has_network_during_ranking: false`, `pre_computation_required: true`
  - [ ] AI tools declared honestly (e.g. Claude for architecture/code)
  - [ ] methodology_summary (≤200 words)
  - [ ] declarations (read_spec, original_work, no_collusion, reproduction_tested)
- [ ] Rename CSV to registered participant/team ID (e.g. `team_xxx.csv`)

## Phase 7 — Final submission ⬜
- [ ] Re-run validator on the final, renamed CSV → 0 errors
- [ ] Push final repo state (pinned requirements, README, metadata)
- [ ] Confirm sandbox link is live and public
- [ ] Submit via portal: CSV + GitHub URL + sandbox URL + metadata
- [ ] Remember: **last valid submission counts**, max 3 total

---

## Known limitations / future tuning (post-baseline)
- Weights in `jd_config.py` are hand-set (not tuned against held-out labels — there are none)
- AI Research Engineer inclusion relies on career-evidence lift; revisit if research-only leaks in
- Adjacent-title lift (0.50 + 0.45·evidence) could be tuned if top-10 quality regresses
- No location hard-gate (JD is "flexible"); currently not penalized

## Current best result snapshot
- Runtime: 41.6s CPU · Validator: PASS · Honeypots top-100: 0 · Stuffer: excluded
- Score range: 0.9817 → 0.8536 (93 unique) · Reasonings: 100/100 unique
- Top-100 title mix: 20 ML Eng, 16 Data Scientist, 13 Junior ML, 7 AI Eng, 5 Search Eng, 5 AI Research Eng, 1 SWE(ML), …
