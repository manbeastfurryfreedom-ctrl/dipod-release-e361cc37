#!/usr/bin/env bash
# Minimal end-to-end reproduction of the DiPOD paper's Section 5.1 two-token
# diagnostic. CPU-only, no GPU, no 8B model, no API. Runs FPO vs FPO+DiPOD on a
# fully-enumerable 2-token masked-diffusion policy and shows that DiPOD controls
# the ELBO-likelihood gap (the paper's "double drift") while improving reward.
set -euo pipefail

# Resolve repo root from this script's location, regardless of caller cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTDIR="$REPO_ROOT/.openresearch/artifacts"
mkdir -p "$ARTDIR"

# Find a python and ensure numpy + matplotlib are importable (install if needed).
PY="$(command -v python3 || command -v python)"
echo "Using python: $PY ($($PY --version 2>&1))"
if ! "$PY" -c "import numpy, matplotlib" 2>/dev/null; then
  echo "Installing numpy + matplotlib ..."
  "$PY" -m pip install --quiet --no-input numpy matplotlib \
    || "$PY" -m pip install --quiet --no-input --break-system-packages numpy matplotlib
fi
"$PY" -c "import numpy, matplotlib; print('numpy', numpy.__version__)"

cd "$SCRIPT_DIR"

echo "=== Training FPO vs FPO+DiPOD (paper Appendix D settings) ==="
# Exact policy gradients (no Monte-Carlo) -> deterministic; one seed suffices.
"$PY" train.py \
  --steps 1500 --lr 0.1 \
  --betas 0.0 0.2 \
  --seeds 1 \
  --outdir "$ARTDIR"

echo "=== Plotting ==="
"$PY" plot.py "$ARTDIR"

echo "=== Writing EVAL.md ==="
"$PY" eval.py "$ARTDIR" "$REPO_ROOT"

echo "=== Done. Artifacts in $ARTDIR ==="
ls -la "$ARTDIR"
