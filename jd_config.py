"""
jd_config.py
============
The Senior AI Engineer JD encoded as machine-readable signals.

Everything the ranker "knows" about the role lives here, so the scoring logic
stays declarative and easy to defend at the Stage-5 interview. Edit weights and
keyword lists here rather than burying them in rank.py.
"""

# The JD text used to build the semantic anchor embedding (precompute.py).
# This is a distilled version of job_description.md focused on the REAL intent,
# deliberately phrased in plain language so it matches plain-language candidates.
JD_ANCHOR_TEXT = (
    "Senior AI engineer who has shipped end-to-end ranking, search, retrieval, "
    "or recommendation systems to real users at a product company. Production "
    "experience with embeddings-based retrieval (sentence-transformers, BGE, E5, "
    "OpenAI embeddings) and vector databases or hybrid search (FAISS, Pinecone, "
    "Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch). Strong Python. Designs "
    "evaluation frameworks for ranking quality (NDCG, MRR, MAP, offline-to-online "
    "correlation, A/B testing). Applied machine learning in production, not pure "
    "research. Pragmatic shipper who improves recommender and search relevance."
)

# ---------------------------------------------------------------------------
# Title classification (the decisive anti-keyword-stuffer signal)
# ---------------------------------------------------------------------------
# Core relevant: the role itself. Strong positive on title alone.
TITLE_CORE = [
    "machine learning engineer", "ml engineer", "ai engineer", "applied scientist",
    "applied ml", "data scientist", "research scientist", "nlp engineer",
    "machine learning", "ml researcher", "deep learning",
]

# Adjacent relevant: plausible Tier-5s, but only if the CAREER DESCRIPTION
# confirms ranking/retrieval/recsys work (handled via semantic_fit, not title alone).
TITLE_ADJACENT = [
    "software engineer", "backend engineer", "data engineer", "analytics engineer",
    "full stack", "platform engineer", "search engineer", "mlops",
]

# Explicitly NOT a fit. A loaded AI-skills list cannot rescue these (CAND_0000083).
TITLE_IRRELEVANT = [
    "hr manager", "marketing manager", "graphic designer", "mechanical engineer",
    "civil engineer", "accountant", "sales executive", "customer support",
    "content writer", "business analyst", "operations manager", "project manager",
    "qa engineer", "frontend engineer", "mobile developer", ".net developer",
    "java developer",
]

# ---------------------------------------------------------------------------
# Real-work keywords — matched against CAREER DESCRIPTIONS + summary, NOT skills.
# ---------------------------------------------------------------------------
WORK_RETRIEVAL = [
    "ranking", "re-rank", "rerank", "retrieval", "recommendation", "recommender",
    "search relevance", "semantic search", "vector search", "embedding", "embeddings",
    "rag", "bm25", "learning to rank", "matching", "personalization", "feed",
    "candidate generation", "nearest neighbor", "ann ", "faiss", "elasticsearch",
    "information retrieval", "relevance",
]
WORK_EVAL = ["ndcg", "mrr", "map@", "a/b test", "ab test", "offline metric",
             "precision@", "recall@", "evaluation framework", "ground truth"]
WORK_ML_PROD = ["production", "deployed", "real users", "at scale", "latency",
                "inference", "served", "shipped"]

# ---------------------------------------------------------------------------
# Disqualifier signals (JD "Things we explicitly do NOT want")
# ---------------------------------------------------------------------------
SERVICES_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                  "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
                  "ltimindtree", "mphasis"]

WRONG_DOMAIN = ["computer vision", "image classification", "object detection",
                "speech recognition", "tts", "robotics", "slam", "ocr",
                "video analytics"]  # only disqualifying if NLP/IR is ABSENT

RESEARCH_ONLY = ["phd researcher", "research assistant", "postdoc", "academic",
                 "research scholar", "lab "]

# ---------------------------------------------------------------------------
# Location preference (soft) — JD favors Pune/Noida, Tier-1 Indian cities
# ---------------------------------------------------------------------------
PREFERRED_CITIES = ["pune", "noida", "bangalore", "bengaluru", "hyderabad",
                    "mumbai", "delhi", "gurgaon", "gurugram", "ncr"]

# ---------------------------------------------------------------------------
# Scoring weights (tune in M4). base_fit components sum to 1.0.
# ---------------------------------------------------------------------------
W_TITLE_CAREER = 0.35   # title + trajectory fit (decisive)
W_SEMANTIC     = 0.30   # career-description embedding vs JD anchor
W_EXPERIENCE   = 0.20   # years + recency + product-vs-services
W_SKILLS_TRUST = 0.15   # AI skills DISCOUNTED by endorsements*duration & title plausibility

# Experience sweet spot (JD: "5-9 years", ideal 6-8)
YOE_IDEAL_LOW, YOE_IDEAL_HIGH = 6.0, 8.0
YOE_ACCEPT_LOW, YOE_ACCEPT_HIGH = 4.0, 12.0

# Penalty magnitudes (subtracted from base_fit, clamped >= 0)
PEN_SERVICES_ONLY = 0.30
PEN_TITLE_HOP     = 0.20    # avg tenure < TITLE_HOP_MONTHS
PEN_WRONG_DOMAIN  = 0.25
PEN_RESEARCH_ONLY = 0.25
TITLE_HOP_MONTHS  = 18

# Behavioral multiplier bounds (applied AFTER base_fit)
BEHAVIORAL_FLOOR = 0.30    # an unavailable perfect-on-paper candidate floors here
# Honeypot gate: consistency failures multiply score by this (pushes to bottom)
HONEYPOT_FACTOR  = 0.02
