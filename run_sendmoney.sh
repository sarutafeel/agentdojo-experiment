#!/bin/bash
# run full-suite send_money description condition with and without spotlighting
cd "$(dirname "$0")"
source ~/.venvs/agentdojo/bin/activate
export PYTHONUNBUFFERED=1
python run_experiment.py --provider openai --model gpt-4o-mini --attacks metadata --target-tool send_money --defenses none spotlighting --out results/sendmoney.csv
echo SENDMONEY_DONE
