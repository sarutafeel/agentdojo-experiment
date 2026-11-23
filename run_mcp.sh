#!/bin/bash
# compare honest and modified MCP tool listings using GPT-4o-mini
cd "$(dirname "$0")"
source ~/.venvs/agentdojo/bin/activate
export PYTHONUNBUFFERED=1
M="--provider openai --model gpt-4o-mini --out results/mcp.csv"
python run_experiment.py $M --quick --attacks mcp_benign mcp --defenses none  # Start with a quick check without defenses.
python run_experiment.py $M --attacks mcp_benign mcp --defenses none spotlighting repeat_user_prompt  # Run the full suite with each of the three defenses.
echo MCP_DONE
