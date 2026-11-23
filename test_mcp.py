"""Check the AgentDojo MCP bridge without making model API calls"""

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from mcp_server import MCPToolServer
from metadata_attack import PoisonState, make_metadata_attack

suite = get_suite("v1.2", "banking")
user_task = suite.user_tasks["user_task_0"]
inj = suite.injection_tasks["injection_task_0"]
TARGET = "get_most_recent_transactions"

# set up fresh banking environment and connect it to MCP server
def run(poisoned: bool):
    state = PoisonState(target_tool=TARGET)
    if poisoned:
        make_metadata_attack(state, "important_instructions", "Emma Johnson", "the AI language model",
                             channel="mcp_tool_poisoning")(suite, None).attack(user_task, inj)
    runtime = FunctionsRuntime(suite.tools)
    env = suite.load_and_inject_default_environment({})
    original = {n: f.description for n, f in runtime.functions.items()}
    element = MCPToolServer(state if poisoned else None)
    element.query("q", runtime, env)
    return runtime, env, original, element


# 1. Poisoned server
# check that only target description changes and that tool list and schema are preserved
runtime, env, original, el = run(poisoned=True)
assert inj.GOAL in runtime.functions[TARGET].description, "poison missing from MCP tool description"
assert all(inj.GOAL not in f.description for n, f in runtime.functions.items() if n != TARGET), "other tools touched"
assert set(runtime.functions) == set(original), "tool set changed"
schema = runtime.functions["send_money"].parameters.model_json_schema()
assert {"recipient", "amount"} <= set(schema["properties"]), schema
print("ok: poisoned description reaches the client over MCP; other tools clean; schemas preserved")

# 2. Honest server
# check that honest server returns original tool descriptions
runtime, env, original, el = run(poisoned=False)
assert all(runtime.functions[n].description == d for n, d in original.items()), "honest server altered a description"
print("ok: honest MCP server returns the original descriptions")

# 3. check that MCP tool calls read and update shared banking state
before = env.bank_account.balance
out, err = runtime.run_function(env, "get_balance", {})
assert err is None and str(before) in out, (out, err)
out, err = runtime.run_function(env, "send_money", {"recipient": "US133000000121212121212", "amount": 1.0,
                                                    "subject": "mcp test", "date": "2022-01-01"})
assert err is None, err
assert any(t.recipient == "US133000000121212121212" for t in env.bank_account.transactions), "env not mutated"
print("ok: tool calls run through MCP and mutate the shared environment")

# 4. try an invalid tool call and print the response
out, err = runtime.run_function(env, "send_money", {"recipient": "x"})
print("ok: bad call ->", (err or out)[:80].replace("\n", " "))
el._bridge.close()
