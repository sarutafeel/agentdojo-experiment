#!/bin/bash
# use the quick subset to compare three tool-description variants without a defense
cd "$(dirname "$0")"
source ~/.venvs/agentdojo/bin/activate
export PYTHONUNBUFFERED=1
M="--provider openai --model gpt-4o-mini --quick --out results/diag.csv"
python run_experiment.py $M --attacks metadata --template tool_doc --defenses none
python run_experiment.py $M --attacks metadata --target-tool send_money --defenses none
python run_experiment.py $M --attacks metadata --template tool_doc --target-tool send_money --defenses none
echo DIAG_DONE
