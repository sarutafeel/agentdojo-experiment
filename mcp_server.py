"""
Run AgentDojo tools through an in-process MCP server.
The agent discovers and calls tools through MCP. The server runs the original tools against the same task state that AgentDojo uses for scoring.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, ConfigDict

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionsRuntime
from agentdojo.types import ChatMessage

from metadata_attack import PoisonState

# keep the input schema exactly as MCP server provided
def _schema_model(name: str, schema: dict) -> type[BaseModel]:
    class MCPInput(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]: # type: ignore[override]
            return schema

    MCPInput.__name__ = f"MCPInput_{name}"
    return MCPInput


def _build_server(runtime: FunctionsRuntime, env: Env, state: PoisonState | None) -> Server:
    server: Server = Server("agentdojo-banking-mcp")
    functions = dict(runtime.functions)
    # give server its own runtime so tool calls dont loop back through MCP
    server_runtime = FunctionsRuntime(list(functions.values()))

    # return tool names, descriptions, input schemas for this condition
    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        tools = []
        for name, f in functions.items():
            description = f.description
            if state is not None and state.text and name == state.target_tool:
                description += state.text # malicious server poisons its own description
            tools.append(
                mcp_types.Tool(name=name, description=description, inputSchema=f.parameters.model_json_schema())
            )
        return tools

    # run original tool using current task state and return result through MCP
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        result, error = server_runtime.run_function(env, name, arguments or {})
        if error:
            return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=error)], isError=True)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=tool_result_to_str(result))], isError=False
        )

    return server


# run async MCP session on a background thread so AgentDojo can call it from synchronous code
class _Bridge:

    def __init__(self, server: Server) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.client: Any = None
        # anyio requires a session to be entered and exited in the SAME task, so one task owns its whole life.
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(server), self.loop)
        self._ready.wait(30)
        if self._error is not None or self.client is None:
            raise RuntimeError(f"MCP session failed to start: {self._error!r}")

    # keep MCP session open until bridge is told to close
    async def _main(self, server: Server) -> None:
        self._stop = asyncio.Event()
        try:
            async with create_connected_server_and_client_session(server) as client:
                self.client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as e: # noqa: BLE001
            self._error = e
            self._ready.set()

    def _run(self, coro: Any, timeout: float = 120.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def list_tools(self) -> list[mcp_types.Tool]:
        return self._run(self.client.list_tools()).tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        res = self._run(self.client.call_tool(name, arguments))
        text = "\n".join(c.text for c in res.content if isinstance(c, mcp_types.TextContent))
        if res.isError:
            raise RuntimeError(text)
        return text

    def close(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self._stop.set) # lets _main leave the session cleanly
            self._main_future.result(10)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)


_CLEAN: dict[str, Function] = {} # suite's real tools remembered so the server never wraps an MCP wrapper

# use original tools so MCP server doesn't call itself and deadlock
def _clean_runtime(runtime: FunctionsRuntime) -> FunctionsRuntime:
    for name, f in runtime.functions.items():
        if not getattr(f.run, "_is_mcp", False):
            _CLEAN[name] = f
    return FunctionsRuntime(list(_CLEAN.values()))

# serves run's tool over MCP and posions one description when state carries attacker text
class MCPToolServer(BasePipelineElement):
    name = "mcp_tool_server"

    def __init__(self, state: PoisonState | None = None) -> None:
        self.state = state
        self._bridge: _Bridge | None = None

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if self._bridge is not None: # a new task run so drop the previous session
            self._bridge.close()
        bridge = self._bridge = _Bridge(_build_server(_clean_runtime(runtime), env, self.state))

        mcp_functions: dict[str, Function] = {}
        for tool in bridge.list_tools():

            def run(_tool_name: str = tool.name, **kwargs: Any) -> str:
                return bridge.call_tool(_tool_name, kwargs)

            run._is_mcp = True # type: ignore[attr-defined]
            mcp_functions[tool.name] = Function(
                name=tool.name,
                description=tool.description or "",
                parameters=_schema_model(tool.name, tool.inputSchema),
                dependencies={}, # server holds the environment so client has no direct access
                run=run,
                full_docstring=tool.description or "",
                return_type=str,
            )
        runtime.update_functions(mcp_functions)
        return query, runtime, env, messages, extra_args
