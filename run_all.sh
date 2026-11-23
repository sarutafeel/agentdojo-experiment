#!/bin/bash
# run the full GPT-4o-mini comparison and save the results to results/full.csv
cd "$(dirname "$0")"
source ~/.venvs/agentdojo/bin/activate
export PYTHONUNBUFFERED=1
M="--provider openai --model gpt-4o-mini --out results/full.csv"
# compare the benign baseline, tool outputs, and tool descriptions across defenses
python run_experiment.py $M --attacks none data metadata --defenses none spotlighting repeat_user_prompt
# run the two additional tool-description conditions on the full suite
python run_experiment.py $M --attacks metadata --template tool_doc --defenses none spotlighting
python run_experiment.py $M --attacks metadata --target-tool send_money --defenses none spotlighting
echo ALL_DONE
