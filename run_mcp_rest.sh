#!/bin/bash
# finish MCP results by running only repeat-user-prompt condition
cd "$(dirname "$0")"
source ~/.venvs/agentdojo/bin/activate
export PYTHONUNBUFFERED=1
python run_experiment.py --provider openai --model gpt-4o-mini --out results/mcp.csv \
  --attacks mcp_benign mcp --defenses repeat_user_prompt
echo MCP_DONE
