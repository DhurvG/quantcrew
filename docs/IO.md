# Input and Output Definitions

## Input — `AnalysisRequest` (`quantcrew/schemas.py`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `ticker` | string | yes | Nifty 50 symbol, with or without `.NS` (e.g. `INFY`, `INFY.NS`). Others are rejected |
| `question` | string, 1–500 chars | no | Default: "Should I buy, hold or sell this stock?" |
| `horizon` | `short` \| `medium` \| `long` \| null | no | 5 / 20 / 60 trading days. If null, the Orchestrator infers it |
| `as_of` | date (YYYY-MM-DD) \| null | no | Analysis date; null = today. Used by backtests |

Sample: `docs/samples/sample_input.json`.

## Output — `FinalReport`

| Field | Type | Notes |
|---|---|---|
| `status` | `ok` \| `degraded` \| `rejected` | `degraded` = decided without one or more agents |
| `ticker`, `as_of`, `horizon` | string | |
| `recommendation` | `BUY` \| `HOLD` \| `SELL` | |
| `confidence` | float 0–1 | BUY/SELL: share of weighted conviction agreeing. HOLD: 1 − largest share pushing BUY or SELL |
| `weighted_score` | float −1–1 | Trust-weighted vote |
| `lean` | `BUY` \| `SELL` \| `NONE` | Which way the panel tilts |
| `minority_call_overruled` | bool | True if a BUY/SELL lacked majority conviction and became HOLD |
| `agent_votes` | map agent → {weight, signal, confidence, vote} | |
| `agent_opinions` | map agent → AgentOpinion + `round` | Final opinion after debate (round 1 = revised) |
| `failed_agents` | map agent → error | |
| `debate` | {rounds, critiques[], revisions[]} | Each revision records signal and confidence before and after |
| `trust` | map agent → {n, hits, raw_accuracy, posterior_accuracy, weight, active} | |
| `narrative` | {summary, bull_case[], bear_case[], key_risks[], what_would_change_the_view[]} | |
| `disclaimer` | string | |

`AgentOpinion` = {signal, confidence 0–1, rationale, key_evidence[]}.
`CriticReview` = {challenges[{agent, issue, question}], needs_revision, summary}.
Rejected requests return {status: "rejected", ticker, error, disclaimer}.

Sample: `docs/samples/final_output_INFY.json` (a real run; never hand-written).
Execution trace: `docs/samples/execution_trace_INFY.json`.
