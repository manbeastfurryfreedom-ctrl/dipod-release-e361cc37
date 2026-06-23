#!/usr/bin/env bash
# Minimal end-to-end reproduction of the DiPOD paper's Section 4.1 / Appendix D
# two-token diagnostic. CPU-only work (no GPU, no 8B model, no API). Runs FPO vs
# FPO+DiPOD on a fully-enumerable 2-token masked-diffusion policy and shows that
# DiPOD controls the ELBO-likelihood gap (the paper's "double drift").

# Resolve repo root from this script's location, regardless of caller cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTDIR="$REPO_ROOT/.openresearch/artifacts"
mkdir -p "$ARTDIR"

# Tee everything to an artifact so we have a log even if the platform log is
# unavailable. Do NOT use `set -e`: we want to reach EVAL.md and the run-log
# artifact even if an optional step (e.g. plotting) hiccups.
LOG="$ARTDIR/run.log"
exec > >(tee "$LOG") 2>&1

echo "=== environment ==="
echo "pwd=$(pwd)  REPO_ROOT=$REPO_ROOT"
uname -a || true

# Find a python.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then echo "FATAL: no python found"; exit 1; fi
echo "python: $PY ($($PY --version 2>&1))"

# numpy is required; matplotlib is optional (only for the PNG).
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "Installing numpy (matplotlib best-effort) ..."
  "$PY" -m pip install --no-input numpy matplotlib 2>&1 \
    || "$PY" -m pip install --no-input --break-system-packages numpy matplotlib 2>&1 \
    || "$PY" -m ensurepip 2>&1 || true
  "$PY" -m pip install --no-input numpy 2>&1 \
    || "$PY" -m pip install --no-input --break-system-packages numpy 2>&1 || true
fi
"$PY" -c "import numpy; print('numpy', numpy.__version__)" \
  || { echo "FATAL: numpy unavailable"; exit 1; }
"$PY" -c "import matplotlib; print('matplotlib', matplotlib.__version__)" \
  && HAVE_MPL=1 || { echo "matplotlib unavailable -> skipping PNG"; HAVE_MPL=0; }

cd "$SCRIPT_DIR"

echo "=== Training: all three evidence axes (paper single, reward robustness, beta ablation) ==="
"$PY" train.py --steps 1500 --lr 0.1 --n_rewards 20 --n_seeds 12 --outdir "$ARTDIR" \
  || { echo "FATAL: training failed"; exit 1; }

if [ "$HAVE_MPL" = "1" ]; then
  echo "=== Plotting ==="
  "$PY" plot.py "$ARTDIR" || echo "WARN: plotting failed (continuing)"
fi

echo "=== Writing EVAL.md ==="
"$PY" eval.py "$ARTDIR" "$REPO_ROOT" || { echo "FATAL: eval failed"; exit 1; }

echo "=== Done. Artifacts in $ARTDIR ==="
ls -la "$ARTDIR"
