import json
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from quantcrew.config import MODEL


@lru_cache(maxsize=8)
def get_llm(model: str | None = None, temperature: float = 0.0):
    """Provider-agnostic chat model. Temperature 0 for consistency between runs; max_retries makes
    the SDK wait and retry on 429 rate limits; low reasoning effort keeps gpt-oss token use down."""
    name = model or MODEL
    extra = {"reasoning_effort": "low"} if "gpt-oss" in name else {}
    return init_chat_model(name, temperature=temperature, max_retries=6, **extra)


def structured_invoke(llm, schema, messages):
    """Return a validated pydantic object, trying the most reliable method first:
    1) provider-native JSON schema mode, 2) function calling, 3) plain JSON validated by pydantic."""
    if isinstance(messages, str):
        messages = [HumanMessage(messages)]
    errors = []
    for method in ("json_schema", "function_calling"):
        try:
            out = llm.with_structured_output(schema, method=method).invoke(messages)
            if out is not None:
                return out
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        instruction = HumanMessage(
            "Reply with ONLY a JSON object (no prose, no code fences) matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema()))
        text = str(llm.invoke(messages + [instruction]).content).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return schema.model_validate_json(text)
    except Exception as exc:
        errors.append(f"plain_json: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError("Structured output failed: " + " | ".join(errors))
