"""
Run AgentDojo experiments to compare prompt-injection defenses.
Save the scores to a CSV and the full message traces as JSON.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

import agentdojo.benchmark as _adb
from agentdojo.functions_runtime import FunctionCall

# let agent AgentDojo resolve the tool-call type when loading cached results
_adb.TaskResults.model_rebuild(_types_namespace={"FunctionCall": FunctionCall})

from mcp_server import MCPToolServer
from metadata_attack import DEFAULT_TARGET_TOOL, PoisonState, ToolDescriptionPoisoner, make_data_attack, make_metadata_attack
from pipeline import DEFENSES, build_pipeline, make_llm

# a fixed slice for quick runs: 4 user tasks x 3 injection tasks = 12 attacked pairs
QUICK = {
    "banking": (["user_task_0", "user_task_3", "user_task_4", "user_task_7"], ["injection_task_0", "injection_task_4", "injection_task_7"]),
    "slack": (["user_task_0", "user_task_1", "user_task_5", "user_task_10"], ["injection_task_1", "injection_task_2", "injection_task_3"]),
}

FIELDS = ["timestamp", "suite", "model", "defense", "attack_location", "template", "target_tool",
          "user_task", "injection_task", "utility", "attack_success"]


# command line options for the experiment
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=["openai", "anthropic", "openai_compatible", "mock"])
    p.add_argument("--model", required=True, help="Model id as the provider names it, e.g. gpt-4o-mini")
    p.add_argument("--suite", default="banking", choices=["banking", "slack", "travel", "workspace"])
    p.add_argument("--benchmark-version", default="v1.2")
    p.add_argument("--attacks", nargs="+", default=["none", "data"], choices=["none", "data", "metadata", "mcp_benign", "mcp"],
                   help="none = benign utility; data = injection in tool outputs; metadata = injection in a tool description (simulated); "
                        "mcp_benign / mcp = same two conditions but the tools are served by a real in-process MCP server, which poisons its own description in `mcp`")
    p.add_argument("--defenses", nargs="+", default=["none"], choices=DEFENSES)
    p.add_argument("--quick", action="store_true", help="Run the small fixed slice in QUICK instead of the whole suite")
    p.add_argument("--user-tasks", nargs="*", help="Explicit user task ids (overrides --quick)")
    p.add_argument("--injection-tasks", nargs="*", help="Explicit injection task ids (overrides --quick)")
    p.add_argument("--target-tool", help="Tool whose description is poisoned (metadata attack). Default depends on suite")
    p.add_argument("--template", default="important_instructions", choices=["important_instructions", "tool_doc"],
                   help="Wording of the metadata-plane injection. important_instructions = same text as the data attack")
    p.add_argument("--victim-name", default="the AI language model", help="How the injection addresses the model")
    p.add_argument("--user-name", default="Emma Johnson", help="Name the injection claims to come from (AgentDojo default)")
    p.add_argument("--temperature", default="0", help="Sampling temperature, or 'none' for models that reject it")
    p.add_argument("--logdir", default="runs", help="Full message traces (JSON) go here")
    p.add_argument("--out", default="results/results.csv", help="One row per task pair is appended here")
    p.add_argument("--force-rerun", action="store_true", help="Ignore cached traces in --logdir")
    return p.parse_args()


def main():
    # read command line options and load environment settings
    args = parse_args()
    load_dotenv(".env")
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore", message="Not all injection tasks were solved")

    # convert temperature option and load selected AgentDojo suite
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    suite = get_suite(args.benchmark_version, args.suite)
    user_tasks, injection_tasks = (QUICK.get(args.suite, (None, None)) if args.quick else (None, None))
    user_tasks = args.user_tasks or user_tasks
    injection_tasks = args.injection_tasks or injection_tasks
    target_tool = args.target_tool or DEFAULT_TARGET_TOOL[args.suite]
    model_tag = args.model.replace("/", "_")

    # create folders for the results CSV and JSON run traces
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # append results to the CSV and write header when the file is new
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        # run each selected defense with selected attack condition
        for defense in args.defenses:
            for location in args.attacks:
                # set up model and extra pipeline steps needed for this condition
                llm = make_llm(args.provider, args.model, temperature)
                state = PoisonState(target_tool=target_tool)
                benign = location in ("none", "mcp_benign")
                if location == "metadata":
                    pre = [ToolDescriptionPoisoner(state)]
                elif location == "mcp":
                    pre = [MCPToolServer(state)]# the server poisons its own tool description
                elif location == "mcp_benign":
                    pre = [MCPToolServer(None)]# same MCP plumbing, honest server
                else:
                    pre = []
                # keep MCP traces separate so they do not reuse cached direct-tool runs
                tag = model_tag + ("-mcp" if location.startswith("mcp") else "")
                pipeline = build_pipeline(llm, tag, defense, pre_elements=pre)
                print(f"\n=== suite={args.suite} model={args.model} defense={defense} attack={location} ===")
                t0 = time.time()

                # save run traces and use AgentDojo runner for selected condition
                with OutputLogger(str(logdir)):
                    if benign:
                        res = benchmark_suite_without_injections(
                            pipeline, suite, logdir=logdir, force_rerun=args.force_rerun,
                            user_tasks=user_tasks, benchmark_version=args.benchmark_version,
                        )
                    else:
                        if location == "data":
                            attack_cls = make_data_attack(args.user_name, args.victim_name)
                        elif location == "mcp":
                            attack_cls = make_metadata_attack(
                                state, args.template, args.user_name, args.victim_name, channel="mcp_tool_poisoning"
                            )
                        else:
                            attack_cls = make_metadata_attack(state, args.template, args.user_name, args.victim_name)
                        res = benchmark_suite_with_injections(
                            pipeline, suite, attack_cls(suite, pipeline), logdir=logdir, force_rerun=args.force_rerun,
                            user_tasks=user_tasks, injection_tasks=injection_tasks,
                            benchmark_version=args.benchmark_version,
                        )

                stamp = datetime.now().isoformat(timespec="seconds")

                # write each result to a row in the CSV
                for key, utility in res["utility_results"].items():
                    ut, it = key if isinstance(key, tuple) else (key, "")
                    sec = res["security_results"].get(key) if not benign else ""
                    writer.writerow({
                        "timestamp": stamp, "suite": args.suite, "model": args.model, "defense": defense,
                        "attack_location": location,
                        "template": args.template if location in ("metadata", "mcp") else ("important_instructions" if location == "data" else ""),
                        "target_tool": target_tool if location in ("metadata", "mcp") else "",
                        "user_task": ut, "injection_task": it, "utility": int(bool(utility)),
                        "attack_success": "" if sec == "" else int(bool(sec)),
                    })
                fh.flush()

                # calculate and print summary for this condition
                u = list(res["utility_results"].values())
                msg = f"utility {100 * sum(u) / len(u):.1f}% over {len(u)} runs"
                if not benign:
                    s = list(res["security_results"].values())
                    msg += f" | attack success {100 * sum(s) / len(s):.1f}%"
                print(f"  -> {msg}  ({time.time() - t0:.0f}s)")

    print(f"\nRows appended to {out}. Summarize with: python summarize.py {out}")


if __name__ == "__main__":
    main()
