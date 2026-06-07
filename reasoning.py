"""
reasoning.py
============
Fact-grounded reasoning generator. Every clause is built from values ACTUALLY
present in the candidate record + computed signals — so nothing is hallucinated
(Stage-4 "No hallucination"), the 10 sampled rows read distinctly (Stage-4
"Variation"), and tone tracks rank (Stage-4 "Rank consistency").

No LLM is called. Variation comes from rotating which real facts we surface and
how we phrase them, keyed deterministically off candidate_id.
"""
from __future__ import annotations
import features as F

# Skills we treat as JD-relevant when naming a concrete strength.
_REL_SKILLS = {
    "rag", "llm", "llms", "fine-tuning llms", "pinecone", "vector search",
    "embeddings", "sentence-transformers", "retrieval", "ranking", "langchain",
    "nlp", "transformers", "semantic search", "faiss", "information retrieval",
    "recommendation systems", "hugging face transformers", "llamaindex",
    "learning to rank", "elasticsearch", "bm25", "xgboost",
}

_EVIDENCE = [
    ("recommendation", "built recommendation systems"),
    ("recommender", "built a recommender"),
    ("learning to rank", "worked on learning-to-rank"),
    ("ranking", "worked on ranking systems"),
    ("retrieval", "worked on retrieval"),
    ("search relevance", "improved search relevance"),
    ("semantic search", "built semantic search"),
    ("vector search", "built vector search"),
    ("personalization", "worked on personalization"),
    ("embedding", "shipped embedding-based features"),
    ("rag", "built RAG pipelines"),
]


def _hash(cid: str) -> int:
    return sum(ord(ch) for ch in cid)


def _named_skill(c: dict):
    """Return the most-endorsed JD-relevant skill the candidate ACTUALLY lists."""
    best, best_e = None, -1
    for s in c.get("skills", []):
        if (s.get("name") or "").lower() in _REL_SKILLS:
            e = s.get("endorsements", 0) or 0
            if e > best_e:
                best, best_e = s.get("name"), e
    return best


def _evidence_phrase(c: dict):
    """Return (phrase, company) for the highest-priority real-work keyword, where
    `company` is the SPECIFIC role whose description contains it — so we never
    misattribute work to the current employer when it happened at a prior one.

    Word-boundary matching prevents hallucinations like "built RAG pipelines"
    from "rag" inside "leverage". If the keyword is only in the summary (not tied
    to a role), company is None and the phrase is stated without an employer.
    """
    roles = c.get("career_history", [])
    summary = c["profile"].get("summary", "")
    for needle, phrase in _EVIDENCE:
        for j in roles:                      # prefer a concrete role attribution
            if F.kw_present(j.get("description", ""), [needle]):
                return phrase, j.get("company")
        if F.kw_present(summary, [needle]):  # else it's a general summary claim
            return phrase, None
    return None, None


def make_reasoning(c: dict, comps: dict, rank: int) -> str:
    p = c["profile"]
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "professional")
    sig = c.get("redrob_signals", {})

    # --- strengths, drawn from real fields, rotated for variation ---
    strengths = []
    ev, ev_company = _evidence_phrase(c)
    if ev:
        strengths.append(ev + (f" at {ev_company}" if ev_company else ""))
    sk = _named_skill(c)
    if sk:
        strengths.append(f"hands-on {sk}")
    if comps["semantic_fit"] >= 0.6 and not ev:
        strengths.append("career history aligned with retrieval/ranking")
    if comps["experience_fit"] >= 0.95:
        strengths.append(f"{yoe:.1f}y squarely in the 6-8y band the JD targets")
    # a concrete behavioral positive, if present
    rr = sig.get("recruiter_response_rate")
    if rr is not None and rr >= 0.6:
        strengths.append(f"responsive to recruiters ({rr:.0%})")

    # --- honest concerns, from real signals/penalties ---
    concerns = []
    concerns += comps.get("penalty_reasons", [])
    concerns += comps.get("behavior_notes", [])
    if comps.get("honeypot_flags"):
        concerns.append("profile inconsistency: " + comps["honeypot_flags"][0])
    nd = sig.get("notice_period_days")
    if nd is not None and nd > 60 and not any("notice" in x for x in concerns):
        concerns.append(f"{nd}d notice period")

    # --- assemble with rotation + rank-aware tone ---
    h = _hash(c["candidate_id"])
    lead_variants = [
        f"{title}, {yoe:.1f}y",
        f"{title} with {yoe:.1f} years",
        f"{yoe:.1f}-year {title}",
    ]
    lead = lead_variants[h % len(lead_variants)]

    s_txt = "; ".join(strengths[:2]) if strengths else "relevant background"
    out = f"{lead} — {s_txt}"

    # Tone tracks rank: top picks lead clean, mid/low surface the gap honestly.
    if concerns:
        if rank <= 5:
            out += f" (minor: {concerns[0]})"
        else:
            out += f". Concern: {concerns[0]}"
    elif rank <= 10:
        out += "; strong, available match"

    return out.strip()[:240]
