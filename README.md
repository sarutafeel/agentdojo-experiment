# AgentDojo prompt-injection experiment

CSCI 1640 progress check. This project compares prompt-injection defenses when
test content appears in a tool output or a tool description. It also checks
the tool-description condition using an in-process MCP server.

Built with [AgentDojo 0.1.35](https://github.com/ethz-spylab/agentdojo) and the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

## Setup

Run these commands from the project folder. The shell scripts use the same
virtual environment.

```bash
python3 -m venv ~/.venvs/agentdojo
source ~/.venvs/agentdojo/bin/activate
python -m pip install -r requirements.txt
```

## Run

Check the setup without an API key:

```bash
python test_mcp.py
python run_experiment.py --provider mock --model mock --quick --attacks none --out results/smoke.csv
```

For the GPT-4o-mini experiments, copy `.env.example` to `.env` and set
`OPENAI_API_KEY`. Run `bash run_all.sh` for the main comparison or
`bash run_mcp.sh` for the MCP comparison. Run
`python run_experiment.py --help` to see individual options.

## Files

| File | Purpose |
| --- | --- |
| `run_experiment.py` | Runs selected benchmark conditions and saves scores and traces. |
| `pipeline.py` | Sets up the model and selected defense. |
| `metadata_attack.py` | Defines the tool-output and tool-description conditions. |
| `mcp_server.py` | Connects AgentDojo tools to an in-process MCP server. |
| `test_mcp.py` | Checks tool discovery and execution through MCP. |
| `mock_llm.py` | Provides a scripted model for local checks. |
| `summarize.py` | Prints a summary table for direct-tool results. |
| `run_all.sh` | Runs the main comparison and the full-suite variants. |
| `run_diag.sh` | Runs the tool-description variants on a small task subset. |
| `run_sendmoney.sh` | Runs the separate `send_money` description comparison. |
| `run_mcp.sh` | Runs the quick and full MCP comparisons. |
| `run_mcp_rest.sh` | Runs the repeat-user-prompt portion of the MCP comparison. |
| `requirements.txt` | Lists the Python dependencies. |
| `.env.example` | Shows how to configure model providers. |
| `results/` | Contains the saved benchmark CSVs. |
| `runs/` | Stores JSON traces generated locally. |
