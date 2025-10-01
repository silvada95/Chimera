#!/usr/bin/env bash
# Setup script for chimera environment with GPU-enabled causal-conv1d
# Requires CUDA for causal-conv1d installation

set -euo pipefail

# Configuration
ENV_NAME="chimera_env"
YML_FILE=""
REUSE="0"
CUDA_TOOLKIT_VERSION="11.8"
TORCH_CUDA_ARCH_LIST_DEFAULT="7.5;8.0;8.6"
CREATE_YML="0"
YML_TYPE="pytorch"

usage() {
  echo "Usage: $0 [-f <environment.yml>] [--create-yml pytorch|conda-forge] [--reuse] [--arch \"7.5;8.0;8.6\"]"
  echo "Options:"
  echo "  -f, --file          Path to environment.yml (or use --create-yml to generate)"
  echo "  --create-yml TYPE   Generate environment.yml (pytorch or conda-forge)"
  echo "  --reuse             Reuse existing environment if it exists"
  echo "  --arch <list>       Set TORCH_CUDA_ARCH_LIST for GPU build (default: 7.5;8.0;8.6)"
  echo "  --cuda-version      CUDA toolkit version (default: 11.8)"
  echo ""
  echo "Examples:"
  echo "  Generate and use PyTorch channel YAML: $0 --create-yml pytorch"
  echo "  Use existing YAML: $0 -f environment.yml"
  echo "  Custom GPU arch: $0 --create-yml pytorch --arch \"8.0;8.6;8.9\""
  echo ""
  echo "Note: This script requires CUDA for causal-conv1d installation"
}

ARCH_LIST="${TORCH_CUDA_ARCH_LIST_DEFAULT}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file) YML_FILE="$2"; shift 2;;
    --create-yml) CREATE_YML="1"; YML_TYPE="$2"; shift 2;;
    --reuse) REUSE="1"; shift;;
    --arch) ARCH_LIST="$2"; shift 2;;
    --cuda-version) CUDA_TOOLKIT_VERSION="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

# Create YAML if requested
if [[ "${CREATE_YML}" == "1" ]]; then
  YML_FILE="environment.yml"
  echo "Creating ${YML_FILE} with ${YML_TYPE} channel configuration..."
  
  if [[ "${YML_TYPE}" == "pytorch" ]]; then
    cat > "${YML_FILE}" <<EOF
name: chimera_env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - setuptools<81
  - wheel
  - numpy=1.26.4
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio=2.1.2
  - pytorch-cuda=${CUDA_TOOLKIT_VERSION}
  - cuda-toolkit=${CUDA_TOOLKIT_VERSION}
  - typing_extensions=4.8.0
  - ninja
  - pyyaml
  - packaging<24
  - pip:
      - einops==0.8.0
      - transformers==4.44.2
      - timm==1.0.9
      - wandb==0.17.7
      - matplotlib==3.7.0
      - seaborn==0.13.2
      - pandas==2.1.0
      - scikit-learn==1.3.0
      - scipy==1.14.1
      - tensorboard==2.16.2
      - opencv-python==4.10.0.84
      - thop==0.1.1.post2209072238
      - omegaconf==2.3.0
      - mamba-ssm==2.2.2
      - h5py==3.14.0
      - hdf5plugin==5.1.0
EOF
  elif [[ "${YML_TYPE}" == "conda-forge" ]]; then
    cat > "${YML_FILE}" <<EOF
name: chimera_env
channels:
  - conda-forge
  - nvidia
channel_priority: strict
dependencies:
  - python=3.10
  - pip
  - setuptools<81
  - wheel
  - numpy=1.26.4
  - pytorch>=2.1,<2.2
  - torchvision>=0.16,<0.17
  - torchaudio>=2.1,<2.2
  - cudatoolkit=${CUDA_TOOLKIT_VERSION}
  - typing_extensions=4.8.0
  - ninja
  - pyyaml
  - packaging<24
  - pip:
      - einops==0.8.0
      - transformers==4.44.2
      - timm==1.0.9
      - wandb==0.17.7
      - matplotlib==3.7.0
      - seaborn==0.13.2
      - pandas==2.1.0
      - scikit-learn==1.3.0
      - scipy==1.14.1
      - tensorboard==2.16.2
      - opencv-python==4.10.0.84
      - thop==0.1.1.post2209072238
      - omegaconf==2.3.0
      - mamba-ssm==2.2.2
      - h5py==3.14.0
      - hdf5plugin==5.1.0
EOF
  else
    echo "Error: Invalid YML_TYPE. Use 'pytorch' or 'conda-forge'."
    exit 1
  fi
  echo "Created ${YML_FILE}"
fi

# Verify YAML file exists
if [[ -z "${YML_FILE}" ]]; then
  echo "Error: No environment.yml specified. Use -f or --create-yml."
  usage
  exit 1
fi

if [[ ! -f "${YML_FILE}" ]]; then
  echo "Error: ${YML_FILE} not found."
  exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
else
  echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
  echo "Continuing anyway..."
fi

# Check conda availability
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Please install Miniconda/Anaconda."
  exit 1
fi

# Source conda
eval "$(conda shell.bash hook)"

# Try to use mamba for faster solving
SOLVER_CMD="conda"
if command -v mamba >/dev/null 2>&1; then
  SOLVER_CMD="mamba"
  echo "Using mamba for faster environment creation..."
else
  # Try to enable libmamba solver
  echo "Enabling libmamba solver for faster environment creation..."
  conda install -n base -c conda-forge conda-libmamba-solver -y 2>/dev/null || true
  conda config --set solver libmamba 2>/dev/null || true
fi

# Extract environment name from YAML
YML_ENV_NAME=$(grep "^name:" "${YML_FILE}" | awk '{print $2}')
if [[ -n "${YML_ENV_NAME}" ]]; then
  ENV_NAME="${YML_ENV_NAME}"
fi

echo "Environment name: ${ENV_NAME}"

# Handle existing environment
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${REUSE}" == "1" ]]; then
    echo "Reusing existing environment: ${ENV_NAME}"
    conda activate "${ENV_NAME}"
  else
    echo "Removing existing environment: ${ENV_NAME}"
    conda remove -n "${ENV_NAME}" --all -y
    echo "Creating environment from ${YML_FILE}..."
    ${SOLVER_CMD} env create -f "${YML_FILE}" || { echo "Environment creation failed"; exit 1; }
    conda activate "${ENV_NAME}"
  fi
else
  echo "Creating environment from ${YML_FILE}..."
  ${SOLVER_CMD} env create -f "${YML_FILE}" || { echo "Environment creation failed"; exit 1; }
  conda activate "${ENV_NAME}"
fi

# Upgrade pip/setuptools/wheel
echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade "pip" "setuptools<81" wheel

# Fix packaging version for causal-conv1d
echo "Ensuring compatible packaging version..."
python -m pip install "packaging<24"

# Verify PyTorch installation with CUDA
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('ERROR: CUDA is not available! GPU support is required.')
    exit(1)
" || { echo "PyTorch CUDA verification failed"; exit 1; }

# Function to install causal-conv1d with GPU support
install_causal_conv1d_gpu() {
  local success=0
  
  echo "Configuring for GPU build..."
  export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
  export CAUSAL_CONV1D_FORCE_CPU=0
  export CAUSAL_CONV1D_FORCE_BUILD=1  # Force building from source
  
  # Ensure CUDA paths are set
  if [[ -z "${CUDA_HOME:-}" ]]; then
    # Try to find CUDA installation
    if [[ -d "/usr/local/cuda" ]]; then
      export CUDA_HOME="/usr/local/cuda"
    elif [[ -d "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime" ]]; then
      export CUDA_HOME="${CONDA_PREFIX}"
    elif [[ -d "${CONDA_PREFIX}" ]]; then
      export CUDA_HOME="${CONDA_PREFIX}"
    fi
    echo "Setting CUDA_HOME=${CUDA_HOME}"
  fi
  
  # Strategy 1: Build from source with no-build-isolation
  echo "Strategy 1: Building from source with no-build-isolation..."
  python -m pip install --no-build-isolation --no-cache-dir "causal-conv1d>=1.1.0,<1.2.0" 2>&1 | tee build.log && success=1
  
  # Strategy 2: Install from GitHub with specific commit
  if [[ "${success}" -eq 0 ]]; then
    echo "Strategy 2: Installing from GitHub (v1.1.1)..."
    python -m pip install --no-cache-dir --no-build-isolation "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.1" 2>&1 | tee build.log && success=1
  fi
  
  # Strategy 3: Manual build with fixes
  if [[ "${success}" -eq 0 ]]; then
    echo "Strategy 3: Manual build with version fixes..."
    
    # Create a temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "${TEMP_DIR}"
    
    # Clone the repository
    git clone https://github.com/Dao-AILab/causal-conv1d.git
    cd causal-conv1d
    git checkout v1.1.1
    
    # Apply fix for version parsing issue if needed
    if [[ -f "setup.py" ]]; then
      sed -i 's/parse(torch_version\.cuda)/parse(str(torch_version.cuda) if torch_version.cuda else "0.0")/g' setup.py 2>/dev/null || true
      sed -i 's/packaging\.version\.parse/lambda x: packaging.version.parse(str(x) if x else "0.0")/g' setup.py 2>/dev/null || true
    fi
    
    # Build and install
    python -m pip install --no-build-isolation . 2>&1 | tee build.log && success=1
    
    # Cleanup
    cd -
    rm -rf "${TEMP_DIR}"
  fi
  
  # Strategy 4: Try with specific setuptools version
  if [[ "${success}" -eq 0 ]]; then
    echo "Strategy 4: Trying with setuptools 68..."
    python -m pip install "setuptools==68.0.0"
    python -m pip install --no-cache-dir --no-build-isolation "causal-conv1d>=1.1.0,<1.2.0" 2>&1 | tee build.log && success=1
    python -m pip install "setuptools<81"
  fi
  
  return $((1 - success))
}

# Install causal-conv1d with GPU support
echo -e "\n=== Installing causal-conv1d with GPU support ==="
echo "TORCH_CUDA_ARCH_LIST=${ARCH_LIST}"

if install_causal_conv1d_gpu; then
  echo "✓ causal-conv1d installed successfully with GPU support"
else
  echo ""
  echo "ERROR: causal-conv1d GPU installation failed!"
  echo ""
  echo "Please check the build.log file for details."
  echo ""
  echo "Manual installation steps:"
  echo "1. Activate the environment:"
  echo "   conda activate ${ENV_NAME}"
  echo ""
  echo "2. Set environment variables:"
  echo "   export TORCH_CUDA_ARCH_LIST=\"${ARCH_LIST}\""
  echo "   export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"
  echo ""
  echo "3. Try manual installation:"
  echo "   git clone https://github.com/Dao-AILab/causal-conv1d.git"
  echo "   cd causal-conv1d"
  echo "   pip install ."
  echo ""
  exit 1
fi

# Final verification
echo -e "\n=== Final Verification ==="
python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')

# Check GPU details
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} (compute capability: {props.major}.{props.minor})')

try:
    import torchvision
    print(f'torchvision: {torchvision.__version__}')
except ImportError:
    print('torchvision: NOT INSTALLED')

try:
    import torchaudio
    print(f'torchaudio: {torchaudio.__version__}')
except ImportError:
    print('torchaudio: NOT INSTALLED')

try:
    import causal_conv1d
    print(f'causal-conv1d: {getattr(causal_conv1d, \"__version__\", \"installed\")}')
    import causal_conv1d_cuda
    print('  ✓ CUDA kernels available')
except ImportError as e:
    print(f'causal-conv1d: ERROR - {e}')
    sys.exit(1)

try:
    import einops
    print(f'einops: {einops.__version__}')
except ImportError:
    print('einops: NOT INSTALLED')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError:
    print('transformers: NOT INSTALLED')

try:
    import thop
    print(f'thop: {getattr(thop, \"__version__\", \"installed\")}')
except ImportError:
    print('thop: NOT INSTALLED')
    sys.exit(1)

try:
    import omegaconf
    print(f'omegaconf: {omegaconf.__version__}')
except ImportError:
    print('omegaconf: NOT INSTALLED')
    sys.exit(1)

try:
    import mamba_ssm
    print(f'mamba-ssm: {getattr(mamba_ssm, \"__version__\", \"installed\")}')
except ImportError:
    print('mamba-ssm: NOT INSTALLED')
    sys.exit(1)

try:
    import h5py, hdf5plugin
    print(f'h5py: {h5py.__version__}')
    print(f'hdf5plugin: {hdf5plugin.__version__}')
except Exception as e:
    print(f'h5py/hdf5plugin: ERROR - {e}')
    sys.exit(1)

try:
    import seaborn, matplotlib
    print(f'seaborn: {seaborn.__version__}')
    print(f'matplotlib: {matplotlib.__version__}')
except Exception as e:
    print(f'seaborn/matplotlib: ERROR - {e}')
    sys.exit(1)

print('\\n✓ All components successfully installed with GPU support!')
"

echo -e "\n=== Setup Complete ==="
echo "Environment activated: ${ENV_NAME}"
echo "CUDA support: Enabled"
echo "TORCH_CUDA_ARCH_LIST: ${ARCH_LIST}"
echo ""
echo "To activate the environment in a new terminal:"
echo "  conda activate ${ENV_NAME}"
echo "  export TORCH_CUDA_ARCH_LIST=\"${ARCH_LIST}\""




