#!/bin/bash
# Installs Python environment dependencies
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 [pip_command] [platform_type]"
    echo ""
    echo "Arguments:"
    echo "  pip_command     The pip executable to use (default: pip3)"
    echo "  platform_type   The environment type, must be 'nvidia' or 'amd' (default: nvidia)"
    echo ""
    echo "Example:"
    echo "  $0 'uv pip' nvidia   # Use uv pip instead of pip3, platform type 'nvidia'"
    echo "  $0                   # Defaults to pip3 and nvidia"
    echo "  $0 --help            # Show this help message"
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

pip=${1:-pip3}
platform=${2:-nvidia}

echo "Using ${pip} to install AI Benchmarking Python dependencies for ${platform} platform"
if [ -z "${VIRTUAL_ENV}" ]; then
    echo -e "\e[1;33m[Warning] Installing dependencies outside of a virtual environment.\e[0m"
else
    echo "- Installing dependencies in virtual environment located at: ${VIRTUAL_ENV}"
fi
echo ""

if [[ "$platform" == "nvidia" ]]; then
    $pip install -r requirements_torch_nvidia.txt
    $pip install $(cat requirements_flashattn.txt) --no-build-isolation
    $pip install -r requirements_main.txt
elif [[ "$platform" == "amd" ]]; then
    # Install PyTorch: https://pytorch.org/get-started/locally/
    $pip install -r requirements_torch_amd.txt
    # Still need other packages for AMD so adding them here so they are grouped similarly to Nvidia
    # Can't add in requirements because the index-url doesn't have them available
    $pip install packaging setuptools wheel

    # Install steps to build FlashAttention: https://github.com/Dao-AILab/flash-attention
    echo "Building FlashAttention for AMD/ROCm"
    if [ -d triton ]; then
        echo "triton already exists.  Skipping clone."
    else
        git clone https://github.com/triton-lang/triton
    fi
    pushd triton > /dev/null
    git checkout 3ca2f498e98ed7249b82722587c511a5610e00c4
    $pip install --verbose -e python
    popd > /dev/null

    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
    if [ -d 'flash-attention' ]; then
        echo "flash-attention already exists.  Skipping clone."
    else
        git clone https://github.com/Dao-AILab/flash-attention.git
    fi
    pushd flash-attention > /dev/null
    python setup.py install
    popd > /dev/null

    # tensorrt won't install on AMD SKUs
    $pip install $(grep -v tensorrt requirements_main.txt)
else
    echo "Specified target platform ${platform} not recognized. Supported platform 'nvidia' or 'amd'"
    exit 1
fi
