"""A small, transparent ReAct loop: the LLM decides which tools to call (and how many times),
sees the results, and finally commits to a structured output. Every tool call is traced."""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from quantcrew.config import MAX_TOOL_STEPS
from quantcrew.llm import get_llm, structured_invoke
from quantcrew.tools.toolkit import make_tools
from quantcrew.trace import event


class AgentFailed(RuntimeError):
    pass


class ToolUsingAgent:
    name: str = "agent"
    tool_names: list[str] = []
    system_prompt: str = ""
    model: str | None = None

    def run(self, task: str, output_schema, as_of: str | None, max_steps: int = MAX_TOOL_STEPS,
            tool_names: list[str] | None = None, require_data: bool = True):
        """Returns (structured_output, trace_events). Raises AgentFailed if no usable data."""
        all_tools = make_tools(as_of)
        tools = [all_tools[n] for n in (tool_names if tool_names is not None else self.tool_names)]
        tool_map = {t.name: t for t in tools}
        llm = get_llm(self.model)
        messages = [SystemMessage(self.system_prompt), HumanMessage(task)]
        events, ok_results, failed_results = [], 0, 0

        bound = llm.bind_tools(tools)
        for _ in range(max_steps):
            try:
                ai = bound.invoke(messages)
            except Exception as exc:
                if "tool_use_failed" not in str(exc):
                    raise
                # the model produced an invalid tool call: tell it and let it try again
                events.append(event(self.name, self.name, "recovered", "invalid tool call, asked model to retry"))
                messages.append(HumanMessage(
                    f"Your last tool call was invalid. The only tools are {list(tool_map)}. "
                    "Call one of them with correct arguments, or answer in plain text if you have enough data."))
                continue
            messages.append(ai)
            if ai.content:
                events.append(event(self.name, self.name, "thought", str(ai.content)[:600]))
            if not ai.tool_calls:
                break
            for call in ai.tool_calls:
                events.append(event(self.name, f"tool:{call['name']}", "tool_call", call["args"]))
                try:
                    result = tool_map[call["name"]].invoke(call["args"])
                except Exception as exc:  # unknown tool / bad args: tell the LLM, let it adapt
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                if isinstance(result, dict) and "error" in result:
                    failed_results += 1
                else:
                    ok_results += 1
                events.append(event(f"tool:{call['name']}", self.name, "tool_result",
                                    _truncate(result)))
                messages.append(ToolMessage(json.dumps(result, default=str)[:2500], tool_call_id=call["id"]))

        if require_data and ok_results == 0 and failed_results > 0:
            raise AgentFailed(f"{self.name}: every tool call failed")
        if require_data and ok_results == 0 and tools:
            raise AgentFailed(f"{self.name}: concluded without gathering any data")

        messages.append(HumanMessage(
            "Now commit to your final answer in the required structure. Use only numbers that "
            "appeared in tool results above; do not invent data."))
        output = structured_invoke(llm, output_schema, messages)
        return output, events


def _truncate(obj, limit: int = 1500):
    text = json.dumps(obj, default=str)
    return obj if len(text) <= limit else text[:limit] + "...<truncated>"
