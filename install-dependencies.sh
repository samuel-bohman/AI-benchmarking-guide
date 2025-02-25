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

# Check if help was requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Assign defaults if no arguments are provided
pip=${1:-pip3}
platform=${2:-nvidia}

echo "Using ${pip} to install AI Benchmarking Python dependencies for ${platform} platform"
if [ -z "${VIRTUAL_ENV}" ]; then
    echo -e "\e[1;33m[Warning] Installing dependencies outside of a virtual environment.\e[0m"
else
    echo "- Installing dependencies in virtual environment located at: ${VIRTUAL_ENV}"
fi
echo ""
echo ""

$pip install -r requirements_torch.txt
$pip install $(cat requirements_flashattn.txt) --no-build-isolation
if [[ "$platform" == "nvidia" ]]; then
    $pip install -r requirements_main.txt
elif [[ "$platform" == "amd" ]]; then
    # tensorrt breaks install on AMD SKUs
    $pip install $(grep -v tensorrt requirements_main.txt)
else
    echo "Specified target platform ${platform} not recognized. Supported platform 'nvidia' or 'amd'"
    exit 1
fi
