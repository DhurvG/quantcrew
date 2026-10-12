"""The Critic Agent: reviews the panel, fact-checks claims against the data with its own tools,
and decides which agents must defend or revise their views."""
import json

from quantcrew.agents.base import ToolUsingAgent
from quantcrew.config import CRITIC_MODEL
from quantcrew.schemas import CriticReview


class CriticAgent(ToolUsingAgent):
    name = "critic"
    model = CRITIC_MODEL
    tool_names = ["get_technical_indicators", "get_risk_metrics", "get_fundamentals"]
    system_prompt = (
        "You are the Critic in QuantCrew, a multi-agent equity research team. You do not give your own "
        "BUY/HOLD/SELL. Your job is to find weak reasoning: claims not backed by numbers, numbers that do "
        "not match the data (use your tools to verify the most important ones), confidence that is too high "
        "for the evidence, conclusions that ignore the stated horizon, and direct contradictions between "
        "agents. Challenge only the agents whose answers could plausibly change. If the panel's reasoning "
        "is sound, set needs_revision to false - do not invent disagreements. "
        "Use only numbers from the panel's opinions or your own tool results - never cite industry "
        "norms or benchmarks from memory. Check the units note in tool results before calling a "
        "number unusual.")

    def review(self, ticker, horizon, opinions, conflict, round_no, as_of, unavailable_tools=()):
        tools = [t for t in self.tool_names if t not in unavailable_tools]
        task = (f"Stock: {ticker}\nHorizon: {horizon}\nDebate round: {round_no + 1}\n"
                f"Automatic conflict check: {json.dumps(conflict)}\n\n"
                f"Panel opinions:\n{json.dumps(opinions, indent=2)}")
        return self.run(task, CriticReview, as_of, max_steps=3, tool_names=tools, require_data=False)


CRITIC = CriticAgent()
