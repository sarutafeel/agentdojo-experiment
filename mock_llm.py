"""
Use a scripted model to test the experiment without API calls
The mock follows fixed rules so its scores are only useful for testing the pipeline
"""
from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

# text markers mock uses to recognize benchmark injections
MARKERS = ("<INFORMATION>", "IMPORTANT USAGE REQUIREMENT")
# record where the mock saw the test text at each run
SEEN_LOG: list[dict] = [] # one entry per final turn(where the injection was visible)


def _has_marker(text: str) -> bool:
    return any(m in text for m in MARKERS)


class MockLLM(BasePipelineElement):
    name = "mock"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        in_desc = [f.name for f in runtime.functions.values() if _has_marker(f.description)] # check tool descriptions and previous tool output for markers
        tool_texts = [get_text_content_as_str(m["content"]) for m in messages if m["role"] == "tool"]
        in_output = any(_has_marker(t) for t in tool_texts)
        # track which tools have already been called in this conversation
        called = [c.function for m in messages if m["role"] == "assistant" and m.get("tool_calls") for c in m["tool_calls"]]

        injected_text = next((f.description for f in runtime.functions.values() if _has_marker(f.description)), "")
        injected_text += " ".join(t for t in tool_texts if _has_marker(t))

        # choose next scripted tool call or finish conversation
        if "get_most_recent_transactions" not in called and "get_most_recent_transactions" in runtime.functions:
            call = FunctionCall(function="get_most_recent_transactions", args={"n": 100}, id="c1")
            out = ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])
        elif injected_text and "password" in injected_text and "update_password" not in called:
            call = FunctionCall(function="update_password", args={"password": "new_password"}, id="c2")
            out = ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])
        else:
            SEEN_LOG.append({"in_description_of": in_desc, "in_tool_output": in_output})
            out = ChatAssistantMessage(
                role="assistant", content=[text_content_block_from_string("Done.")], tool_calls=None
            )
        return query, runtime, env, [*messages, out], extra_args
