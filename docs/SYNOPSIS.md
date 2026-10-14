# QuantCrew — Project Synopsis

**Team ZEUS:** Omayr Muqarrab Yunus (23070122156), Gangurde Dhruv Deepak (23070122091),
Lakshya Jain (23070122124), Ghule Om Ajit (23070122154)

**Problem.** A retail investor or finance student looking at a Nifty 50 stock has to combine very
different kinds of evidence: valuation and growth, price trend and momentum, recent news, and risk.
A single AI model asked "should I buy this?" tends to blend these into one confident answer, mixes up
units and currencies, and sometimes invents figures, with no way to see where the reasoning went wrong.

**Solution.** QuantCrew is a multi-agent system built with LangGraph in which specialised AI agents
research a stock together. An Orchestrator agent reads the user's question, decides which specialists
are needed and for what horizon, and runs them in parallel. Four specialists (Fundamental, Technical,
News Sentiment and Risk) each choose their own data tools (live NSE prices, fundamentals and headlines)
and return a BUY / HOLD / SELL opinion with evidence. A Critic agent then fact-checks their claims with
its own tools and challenges weak, unsupported or contradictory reasoning; challenged agents must revise
or defend their view. The Orchestrator combines the final opinions in a trust-weighted vote. Each
agent's weight is learned from whether its past calls were right, with guards against noisy short-term
results (a ±2% no-move band, a statistical prior, and a 15-call minimum before any weight changes).
Failed agents are retried, then dropped, and every message and tool call is recorded in an execution
trace. The output is a recommendation with confidence, a bull and bear case, and the full reasoning
trail. It is advisory decision support only; the system never places trades.

**Target user.** Retail investors and finance or computer-science students who want a transparent,
explainable second opinion on a Nifty 50 stock, and who value seeing how and why the analysts
disagreed as much as the final call.

**Changes since approval.** The approved plan included backtesting against buy-and-hold and against
single-agent versions. The backtest and ablation code is complete, leak-free and unit-tested, but the
full multi-stock study was reduced to a pilot (one stock, six decisions, full crew vs Technical agent
only), because the free LLM API allows about 200,000 tokens per model per day and one full analysis
uses roughly 40,000. With so few outcomes, agent trust weights remain at their neutral value: by
design they only change after 15 scored calls, the safeguard against noisy short-term outcomes
recommended in the approval review. The Fundamental and Sentiment agents are excluded from backtests
because no point-in-time fundamentals or historical news are freely available.
