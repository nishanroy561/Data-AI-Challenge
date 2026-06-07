"""
features.py
===========
Pure-Python, model-free feature extraction from a candidate record.

NOTHING here touches the network or a model, so rank.py can import it and stay
within the offline/CPU/5-min constraint. The semantic-similarity feature is the
only thing that needs embeddings, and that is precomputed (see precompute.py).

Each public function returns plain floats/bools so the whole pool can be turned
into a numpy/pandas table and scored vectorised.
"""
from __future__ import annotations
import datetime as _dt
import re
import jd_config as J

_TODAY = _dt.date(2026, 6, 7)   # dataset "now"; keep deterministic, not date.today()


def _lower(s) -> str:
    return (s or "").lower()


# ---------------------------------------------------------------------------
# Word-boundary keyword matching.
# Naive `kw in text` is unsafe for short tokens: "rag" matches "leveRAGe",
# "feed" matches "feedback", "tts" matches "waTTS", "search" matches "reSEARCH".
# That caused false work-evidence AND hallucinated reasoning ("built RAG..."),
# so all keyword checks go through a cached word-boundary regex.
# ---------------------------------------------------------------------------
_LIST_CACHE: dict[tuple, "re.Pattern"] = {}


def _list_pattern(keywords) -> "re.Pattern":
    """One compiled alternation regex per keyword LIST, so matching is a single
    pass over the text instead of len(keywords) separate searches — ~15x faster
    over the 100K pool while keeping word-boundary correctness."""
    key = tuple(keywords)
    p = _LIST_CACHE.get(key)
    if p is None:
        parts = []
        for kw in keywords:
            k = kw.strip()
            if not k:
                continue
            left = r"\b" if k[0].isalnum() else ""
            right = r"\b" if k[-1].isalnum() else ""
            parts.append(left + re.escape(k) + right)
        p = _LIST_CACHE[key] = re.compile("|".join(parts), re.IGNORECASE)
    return p


def kw_count(text: str, keywords) -> int:
    """Number of DISTINCT keywords that appear as whole tokens in text."""
    return len({m.group(0).lower() for m in _list_pattern(keywords).finditer(text)})


def kw_present(text: str, keywords) -> bool:
    return _list_pattern(keywords).search(text) is not None


def profile_text(c: dict) -> str:
    """The TRUTHFUL narrative we embed: summary + career titles + descriptions.

    Deliberately EXCLUDES the raw skills list, which is the adversarial channel
    keyword-stuffers exploit (e.g. CAND_0000083, a Graphic Designer listing RAG,
    Pinecone, FAISS). We let real work speak through career descriptions.
    """
    p = c["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in c.get("career_history", []):
        parts.append(f"{job.get('title','')} at {job.get('company','')}. "
                     f"{job.get('description','')}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Title / career trajectory
# ---------------------------------------------------------------------------
def career_work_evidence(c: dict) -> float:
    """0..1 from ranking/retrieval/recsys/eval keywords in CAREER DESCRIPTIONS +
    summary — i.e. what the candidate actually BUILT, not their skills list.

    This is the signal that separates a plain-language Tier-5 (real ranking work,
    adjacent title) from a keyword-stuffer (buzzwords in the skills list, work
    described in the descriptions is unrelated). We read descriptions, never skills.
    """
    parts = [c["profile"].get("summary", "")]
    for j in c.get("career_history", []):
        parts.append(j.get("description", ""))
    text = " ".join(parts)
    hits = kw_count(text, J.WORK_RETRIEVAL) + kw_count(text, J.WORK_EVAL)
    return min(1.0, hits / 5.0)


def title_career_fit(c: dict) -> float:
    """0..1. Anti-stuffer signal: title sets the band, real career evidence lifts it.

    Key design (per JD): a "Tier-5" candidate with an adjacent title (Search
    Engineer, Software Engineer (ML), Backend/Data Engineer) who DESCRIBES building
    ranking/retrieval/recsys is a strong fit — so career evidence can lift adjacent/
    neutral titles toward core. Irrelevant titles (Marketing/Graphic Designer) stay
    capped: a loaded AI-skills list cannot rescue them (CAND_0000083).
    """
    cur = _lower(c["profile"].get("current_title"))
    hist_titles = [_lower(j.get("title")) for j in c.get("career_history", [])]
    all_titles = [cur] + hist_titles

    def cls(t: str) -> str:
        if any(k in t for k in J.TITLE_CORE):
            return "core"
        if any(k in t for k in J.TITLE_IRRELEVANT):
            return "irrelevant"
        if any(k in t for k in J.TITLE_ADJACENT):
            return "adjacent"
        return "neutral"

    cur_cls = cls(cur)
    ev = career_work_evidence(c)   # 0..1 from descriptions, not skills

    if cur_cls == "core":
        base = 0.90 + 0.10 * ev
    elif cur_cls == "adjacent":
        base = 0.50 + 0.45 * ev    # strong evidence lifts Search Eng / SWE(ML) ~core
    elif cur_cls == "neutral":
        base = 0.28 + 0.42 * ev    # only lifts if they actually built ranking/recsys
    else:  # irrelevant -> capped; keywords/skills cannot rescue a wrong-career profile
        base = 0.08

    # Trajectory: prior ML/core roles nudge up (but never rescue an irrelevant arc).
    core_hist = sum(1 for t in hist_titles if cls(t) == "core")
    if core_hist and cur_cls != "irrelevant":
        base = min(1.0, base + 0.04 * core_hist)
    if all(cls(t) == "irrelevant" for t in all_titles if t):
        base = min(base, 0.05)
    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------
def experience_fit(c: dict) -> float:
    """0..1 triangular preference around the JD's 6-8 yr ideal, 4-12 acceptable."""
    yoe = float(c["profile"].get("years_of_experience", 0) or 0)
    if J.YOE_IDEAL_LOW <= yoe <= J.YOE_IDEAL_HIGH:
        score = 1.0
    elif yoe < J.YOE_IDEAL_LOW:
        lo = J.YOE_ACCEPT_LOW
        score = max(0.0, (yoe - lo) / (J.YOE_IDEAL_LOW - lo)) if yoe >= lo else 0.15
    else:
        hi = J.YOE_ACCEPT_HIGH
        score = max(0.0, (hi - yoe) / (hi - J.YOE_IDEAL_HIGH)) if yoe <= hi else 0.15
    # Recency: is the current role hands-on and recent? (penalise stale "architect")
    if not _has_current_role(c):
        score *= 0.85
    return max(0.0, min(1.0, score))


def _has_current_role(c: dict) -> bool:
    return any(j.get("is_current") for j in c.get("career_history", []))


# ---------------------------------------------------------------------------
# Skills trust  (AI skills DISCOUNTED — never the headline signal)
# ---------------------------------------------------------------------------
_AI_SKILL_TOKENS = {
    "rag", "llm", "llms", "fine-tuning llms", "pinecone", "vector search",
    "embeddings", "sentence-transformers", "retrieval", "ranking", "langchain",
    "nlp", "transformers", "semantic search", "faiss", "information retrieval",
    "recommendation systems", "hugging face transformers", "llamaindex",
}


def skills_trust(c: dict) -> float:
    """0..1. Counts AI skills ONLY when backed by endorsements + usage duration,
    AND gates on title plausibility so a stuffer's perfect list scores ~0."""
    title_plausible = title_career_fit(c) > 0.30
    if not title_plausible:
        return 0.0
    trusted = 0.0
    for s in c.get("skills", []):
        if _lower(s.get("name")) in _AI_SKILL_TOKENS:
            dur = s.get("duration_months", 0) or 0
            end = s.get("endorsements", 0) or 0
            prof = s.get("proficiency", "beginner")
            if dur >= 6 and end >= 3 and prof in ("advanced", "expert"):
                trusted += 1.0
            elif dur >= 3:
                trusted += 0.4
    # assessment scores corroborate
    assess = c.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    if any(v >= 60 for v in assess.values()):
        trusted += 1.0
    return min(1.0, trusted / 5.0)


# ---------------------------------------------------------------------------
# Disqualifier penalties  ->  total penalty in [0, ~1]
# ---------------------------------------------------------------------------
def penalties(c: dict) -> tuple[float, list[str]]:
    pen = 0.0
    reasons = []
    companies = [_lower(j.get("company")) for j in c.get("career_history", [])]
    text = profile_text(c)

    # Services-only entire career (company names are distinctive -> substring is fine)
    if companies and all(any(f in co for f in J.SERVICES_FIRMS) for co in companies):
        pen += J.PEN_SERVICES_ONLY
        reasons.append("services-firm-only career")

    # Title-hopping: short average tenure across >=3 roles
    durs = [j.get("duration_months", 0) or 0 for j in c.get("career_history", [])]
    if len(durs) >= 3 and (sum(durs) / len(durs)) < J.TITLE_HOP_MONTHS:
        pen += J.PEN_TITLE_HOP
        reasons.append("frequent job-hopping")

    # Wrong domain (CV/speech/robotics) with NO NLP/IR presence (word-boundary)
    has_wrong = kw_present(text, J.WRONG_DOMAIN)
    has_nlp = kw_present(text, ["nlp", "retrieval", "ranking", "recommendation",
                                "language model", "semantic search", "search relevance"])
    if has_wrong and not has_nlp:
        pen += J.PEN_WRONG_DOMAIN
        reasons.append("CV/speech/robotics focus without NLP/IR")

    # Research-only
    if kw_present(text, J.RESEARCH_ONLY) and not kw_present(text, ["production"]):
        pen += J.PEN_RESEARCH_ONLY
        reasons.append("research-only, no production")

    return pen, reasons


# ---------------------------------------------------------------------------
# Behavioral availability multiplier  ->  [BEHAVIORAL_FLOOR, 1.0]
# ---------------------------------------------------------------------------
def behavioral_multiplier(c: dict) -> tuple[float, list[str]]:
    s = c.get("redrob_signals", {})
    m = 1.0
    notes = []

    # Recency of activity
    days_idle = _days_since(s.get("last_active_date"))
    if days_idle is not None:
        if days_idle > 180:
            m *= 0.55; notes.append(f"inactive {days_idle}d")
        elif days_idle > 90:
            m *= 0.80; notes.append(f"low recent activity ({days_idle}d)")

    # Recruiter responsiveness
    rr = s.get("recruiter_response_rate")
    if rr is not None:
        if rr < 0.15:
            m *= 0.65; notes.append(f"low response rate {rr:.0%}")
        elif rr < 0.35:
            m *= 0.85

    # Open to work
    if s.get("open_to_work_flag") is False:
        m *= 0.80; notes.append("not open to work")

    # Notice period (JD prefers <=30d)
    nd = s.get("notice_period_days")
    if nd is not None and nd > 90:
        m *= 0.90; notes.append(f"{nd}d notice")

    # Interview follow-through
    icr = s.get("interview_completion_rate")
    if icr is not None and icr < 0.5:
        m *= 0.90

    m = max(J.BEHAVIORAL_FLOOR, min(1.0, m))
    return m, notes


# ---------------------------------------------------------------------------
# Honeypot / internal-consistency check  ->  (is_honeypot, flags)
# ---------------------------------------------------------------------------
def honeypot_flags(c: dict) -> list[str]:
    flags = []
    yoe = float(c["profile"].get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12

    # 1. "expert" proficiency with 0 months of usage
    for s in c.get("skills", []):
        if s.get("proficiency") == "expert" and (s.get("duration_months", 1) or 1) == 0:
            flags.append("expert skill with 0 months used")
            break

    # 2. skill used longer than the person has worked
    for s in c.get("skills", []):
        if (s.get("duration_months", 0) or 0) > yoe_months + 12:
            flags.append("skill duration exceeds total experience")
            break

    # 3. single-job tenure exceeds total experience
    for j in c.get("career_history", []):
        if (j.get("duration_months", 0) or 0) > yoe_months + 24:
            flags.append("single-role tenure exceeds total experience")
            break

    # 4. tenure at a company older than the company's plausible existence
    #    (proxy: a job that started before candidate could have begun working)
    for j in c.get("career_history", []):
        sy = _year(j.get("start_date"))
        if sy and sy < (_TODAY.year - yoe - 6):
            flags.append("role start predates plausible career start")
            break

    # 5. education end before start, or impossible years
    for e in c.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            flags.append("education ends before it starts")
            break

    return flags


# ---------------------------------------------------------------------------
# Combine — the single scoring function shared by rank.py and the HF sandbox.
# ---------------------------------------------------------------------------
def combine_score(c: dict, sem_f: float) -> tuple[float, dict]:
    """Final score for one candidate given its semantic similarity to the JD.

        score = base_fit x behavioral_multiplier x honeypot_gate
        base_fit = W_TITLE_CAREER*title + W_SEMANTIC*sem
                 + W_EXPERIENCE*exp + W_SKILLS_TRUST*skills  -  penalties

    Returns (score, comps) where comps carries the component values + notes used
    for reasoning. Keeping this in one place means rank.py (precomputed
    embeddings) and the sandbox (on-the-fly embeddings) can never drift.
    """
    title_f = title_career_fit(c)
    exp_f = experience_fit(c)
    skills_f = skills_trust(c)
    pen, pen_reasons = penalties(c)
    beh_m, beh_notes = behavioral_multiplier(c)
    hp = honeypot_flags(c)

    base = (J.W_TITLE_CAREER * title_f
            + J.W_SEMANTIC * float(sem_f)
            + J.W_EXPERIENCE * exp_f
            + J.W_SKILLS_TRUST * skills_f)
    base = max(0.0, min(1.0, base - pen))
    gate = J.HONEYPOT_FACTOR if len(hp) >= 1 else 1.0
    score = base * beh_m * gate

    comps = {
        "title_career_fit": title_f, "semantic_fit": float(sem_f),
        "experience_fit": exp_f, "skills_trust": skills_f,
        "penalty_reasons": pen_reasons, "behavior_notes": beh_notes,
        "honeypot_flags": hp,
    }
    return score, comps


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _days_since(date_str):
    d = _date(date_str)
    return (_TODAY - d).days if d else None


def _year(date_str):
    d = _date(date_str)
    return d.year if d else None


def _date(date_str):
    if not date_str:
        return None
    try:
        return _dt.date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return None
