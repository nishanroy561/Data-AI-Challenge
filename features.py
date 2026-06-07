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
import jd_config as J

_TODAY = _dt.date(2026, 6, 7)   # dataset "now"; keep deterministic, not date.today()


def _lower(s) -> str:
    return (s or "").lower()


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
def title_career_fit(c: dict) -> float:
    """0..1. Decisive anti-stuffer signal driven by titles, not skills."""
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
    # Current title dominates; history nudges.
    base = {"core": 0.95, "adjacent": 0.55, "neutral": 0.35, "irrelevant": 0.08}[cur_cls]

    # Reward a trajectory that has been in ML/relevant roles over time.
    core_hist = sum(1 for t in hist_titles if cls(t) == "core")
    if core_hist:
        base = min(1.0, base + 0.05 * core_hist)
    # A career that is ENTIRELY irrelevant titles cannot be a hidden gem.
    if all(cls(t) == "irrelevant" for t in all_titles if t):
        base = min(base, 0.05)
    return base


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
    text = profile_text(c).lower()

    # Services-only entire career
    if companies and all(any(f in co for f in J.SERVICES_FIRMS) for co in companies):
        pen += J.PEN_SERVICES_ONLY
        reasons.append("services-firm-only career")

    # Title-hopping: short average tenure across >=3 roles
    durs = [j.get("duration_months", 0) or 0 for j in c.get("career_history", [])]
    if len(durs) >= 3 and (sum(durs) / len(durs)) < J.TITLE_HOP_MONTHS:
        pen += J.PEN_TITLE_HOP
        reasons.append("frequent job-hopping")

    # Wrong domain (CV/speech/robotics) with NO NLP/IR presence
    has_wrong = any(w in text for w in J.WRONG_DOMAIN)
    has_nlp = any(w in text for w in ("nlp", "retrieval", "ranking", "search",
                                      "recommendation", "language model"))
    if has_wrong and not has_nlp:
        pen += J.PEN_WRONG_DOMAIN
        reasons.append("CV/speech/robotics focus without NLP/IR")

    # Research-only
    if any(r in text for r in J.RESEARCH_ONLY) and "production" not in text:
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
