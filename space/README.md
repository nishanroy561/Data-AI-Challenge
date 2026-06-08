---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Redrob Candidate Ranker — Sandbox

Hosted demo of the Redrob hackathon ranking system (Senior AI Engineer JD).

Upload up to 100 candidate profiles (`.jsonl` or `.json` matching
`candidate_schema.json`) and get the ranked shortlist with reasoning. Leave the
inputs empty to rank a bundled 50-candidate sample.

**How it works**
- Embeds the uploaded sample on the fly with `BAAI/bge-small-en-v1.5` (CPU).
- Scores with the exact same code as the competition `rank.py`
  (`features.combine_score`), pulled from the
  [GitHub repo](https://github.com/nishanroy561/Data-AI-Challenge) at startup so
  the demo never drifts from the real ranker.
- `score = base_fit × behavioral_multiplier × honeypot_gate`, where title +
  career-history evidence (not the skills list) is the decisive anti-stuffer
  signal.

The full 100K reproduction runs offline on CPU in ~2 min from the GitHub repo —
see its README. This Space is the small-sample sanity check required by the
submission spec (Section 10.5).
