import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Model profiles. "groq" = free tier, for development and testing.
# "claude" = paid Claude Haiku 5.5 for every role, for the backtest and demo day (about 1 cent a run).
# Switch with QUANTCREW_PROFILE; QUANTCREW_MODEL / QUANTCREW_CRITIC_MODEL override a profile.
PROFILES = {
    "groq": {"model": "groq:openai/gpt-oss-20b", "critic": "groq:openai/gpt-oss-120b"},
    "claude": {"model": "anthropic:claude-haiku-5-5", "critic": "anthropic:claude-haiku-5-5"},
}
PROFILE = os.getenv("QUANTCREW_PROFILE", "groq").strip().lower()
if PROFILE not in PROFILES:
    raise ValueError(f"QUANTCREW_PROFILE must be one of {list(PROFILES)}, got {PROFILE!r}")
MODEL = os.getenv("QUANTCREW_MODEL") or PROFILES[PROFILE]["model"]
CRITIC_MODEL = os.getenv("QUANTCREW_CRITIC_MODEL") or PROFILES[PROFILE]["critic"]
MAX_DEBATE_ROUNDS = int(os.getenv("QUANTCREW_MAX_DEBATE_ROUNDS", "1"))
MAX_TOOL_STEPS = int(os.getenv("QUANTCREW_MAX_TOOL_STEPS", "3"))
MAX_CONCURRENCY = int(os.getenv("QUANTCREW_MAX_CONCURRENCY", "2"))

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = Path(os.getenv("QUANTCREW_DATA_DIR", ROOT / "data"))
TRACE_DIR = Path(os.getenv("QUANTCREW_TRACE_DIR", ROOT / "traces"))
TRUST_FILE = DATA_DIR / "trust.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.jsonl"

SPECIALISTS = ["fundamental", "technical", "sentiment", "risk"]

# Trading days ahead used to judge each horizon's recommendation.
HORIZON_DAYS = {"short": 5, "medium": 20, "long": 60}

# Trust-weight learning (see docs/DESIGN.md, section 6)
TRUST_PRIOR = 5.0        # Beta(5,5) prior: weights start neutral and move slowly
TRUST_WINDOW = 60        # only the most recent N scored calls count
TRUST_MIN_SAMPLES = 15   # no weight change until an agent has this many scored calls
TRUST_BAND = 0.02        # |return| below 2% = "no meaningful move"
TRUST_CLAMP = (0.5, 1.5)

BENCHMARK = "^NSEI"

# Nifty 50 constituents (Yahoo symbols). Membership changes twice a year:
# verify against NSE's current list before your demo and edit here.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]
NIFTY50_SYMBOLS = [f"{s}.NS" for s in NIFTY50]


def normalise_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t
