# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Initialize git submodules when fpo-control is checked out as its own repo.
# If embedded under a larger repo or synced without .git, use the fallback clones below.
GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ "$GIT_TOPLEVEL" = "$SCRIPT_DIR" ]; then
  git submodule sync --recursive
  git submodule update --init --recursive
fi

ensure_git_checkout() {
  local path=$1
  local url=$2
  local rev=$3

  if [ -d "$path/.git" ] || [ -f "$path/.git" ]; then
    return
  fi

  if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
    echo "$path exists and is not empty; skipping fallback clone."
    return
  fi

  git clone "$url" "$path"
  git -C "$path" checkout "$rev"
}

# When fpo-control is embedded under DiPOD, the original submodule metadata is
# not owned by this directory. Fall back to explicit checkouts if the folders are empty.
if [ ! -f IsaacLab/isaaclab.sh ]; then
  ensure_git_checkout IsaacLab https://github.com/isaac-sim/IsaacLab.git 21f7136325136ca3f6ca4e0a8125edffe5c24f7e
fi
if [ ! -d whole_body_tracking/source/whole_body_tracking ]; then
  ensure_git_checkout whole_body_tracking https://github.com/HybridRobotics/whole_body_tracking.git cd65172032893724b445448818c34165846d847d
fi

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/isaaclab_fpo
SENTINEL_FILE=.env_setup_finished

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  if [[ "$(lsb_release -is)" == "Ubuntu" ]]; then
    sudo apt install -y build-essential
  fi

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n isaaclab_fpo python=3.10
  fi

  source $CONDA_ROOT/bin/activate isaaclab_fpo

  pip install "numpy==1.26.4"
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  # setuptools<71 keeps pkg_resources, needed by flatdict==4.0.1's setup.py
  pip install --upgrade pip "setuptools<71"

  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

  # Pre-install flatdict==4.0.1 (source-only, uses pkg_resources which is
  # missing in pip's build-isolation with modern setuptools).
  pip install --no-build-isolation "flatdict==4.0.1"
  bash IsaacLab/isaaclab.sh --install rsl_rl

  # isaaclab.sh may fail to install the core 'isaaclab' package (flatdict
  # version conflict during the find loop). Re-install it explicitly.
  # --no-deps avoids downgrading packages isaacsim already installed at
  # specific versions (onnx, prettytable, pillow, etc.)
  pip install toml prettytable
  pip install --no-deps --no-build-isolation --editable IsaacLab/source/isaaclab

  # Misc dependencies
  pip install "opencv-python==4.9.0.80" "numba==0.61.2" \
    "websockets==15.0.1" "wandb==0.25.1" "viser==1.0.24"

  # Download robot description files for whole_body_tracking
  WBT_ASSETS="$SCRIPT_DIR/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets"
  if [ ! -d "$WBT_ASSETS/unitree_description" ]; then
    curl -L -o /tmp/unitree_description.tar.gz \
      https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
    tar -xzf /tmp/unitree_description.tar.gz -C "$WBT_ASSETS/"
    rm /tmp/unitree_description.tar.gz
  fi

  # Our packages
  pip install -e ./rsl_rl -e ./fpo_rsl_rl -e ./isaaclab_fpo \
    -e ./whole_body_tracking/source/whole_body_tracking

  # Download LAFAN1 motion data and convert to NPZ via IsaacSim FK
  export OMNI_KIT_ACCEPT_EULA=YES
  python $SCRIPT_DIR/whole_body_tracking_reference_data/download_lafan_data.py --headless

  touch $SENTINEL_FILE
fi

source $CONDA_ROOT/bin/activate isaaclab_fpo

if ! python -m pip show isaacsim >/dev/null 2>&1; then
  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
fi

# Keep lightweight Python dependencies present even when an older setup already
# created the sentinel before this list was updated.
pip install "opencv-python==4.9.0.80" "numba==0.61.2" \
  "websockets==15.0.1" "wandb==0.25.1" "viser==1.0.24"

# Reinstall local packages in editable mode in case the sentinel was created
# before this step completed.
pip install -e ./rsl_rl -e ./fpo_rsl_rl -e ./isaaclab_fpo \
  -e ./whole_body_tracking/source/whole_body_tracking

export OMNI_KIT_ACCEPT_EULA=YES
