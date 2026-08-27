#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export FL_DATA_ROOT="${FL_DATA_ROOT:-$HOME/datasets/cifar10}"
python -c "from fl.data import make_splits; make_splits()"
