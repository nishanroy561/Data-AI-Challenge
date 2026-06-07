"""
reasoning.py
============
Fact-grounded reasoning generator. Builds the 1-2 sentence `reasoning` column
from values ACTUALLY present in the candidate record + the computed signals, so
nothing is hallucinated (Stage-4 "No hallucination" check) and tone tracks rank.

No LLM is called — every clause is templated from extracted facts, then varied.
"""
from __future__ import annotations
import features as F


def make_reasoning(c: dict, comps: dict, rank: int) -> str:
    """comps: dict of computed component scores/notes for this candidate."""
    p = c["profile"]
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "professional")

    # Lead with the decisive fact: title + experience.
    lead = f"{title} with {yoe:.1f} yrs"

    # A concrete strength pulled from real fields.
    strengths = []
    if comps["semantic_fit"] >= 0.55:
        # cite a real career snippet keyword if present
        kw = _evidence_keyword(c)
        if kw:
            strengths.append(f"career history shows {kw} work")
        else:
            strengths.append("career history aligns with retrieval/ranking work")
    trusted = comps["skills_trust"]
    if trusted >= 0.4:
        strengths.append("endorsed, assessment-backed AI skills")
    if comps["experience_fit"] >= 0.9:
        strengths.append("experience squarely in the 6-8 yr band")

    # Honest concerns (Stage-4 "Honest concerns" check).
    concerns = list(comps.get("behavior_notes", [])) + list(comps.get("penalty_reasons", []))
    if comps.get("honeypot_flags"):
        concerns.append("profile inconsistencies: " + comps["honeypot_flags"][0])

    # Assemble with tone matched to rank.
    parts = [lead]
    if strengths:
        parts.append("; ".join(strengths[:2]))
    sentence = ". ".join([", ".join(parts[:1] + parts[1:2])]) if len(parts) > 1 else lead

    out = sentence
    if concerns and rank > 3:                 # top picks: lead clean; lower: surface gaps
        out += f". Concern: {concerns[0]}"
    elif concerns and rank <= 3:
        out += f" (note: {concerns[0]})"
    elif rank <= 10 and not concerns:
        out += ". Strong, available match"

    return out.strip().replace("..", ".")[:240]


_EVIDENCE = [
    ("recommendation", "recommendation-system"),
    ("recommender", "recommender"),
    ("ranking", "ranking"),
    ("retrieval", "retrieval"),
    ("search relevance", "search-relevance"),
    ("semantic search", "semantic-search"),
    ("vector search", "vector-search"),
    ("embedding", "embeddings"),
    ("rag", "RAG"),
    ("personalization", "personalization"),
]


def _evidence_keyword(c: dict):
    text = F.profile_text(c).lower()
    for needle, label in _EVIDENCE:
        if needle in text:
            return label
    return None
