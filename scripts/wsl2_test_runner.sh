#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# VNTX - Headless WSL2 & Linux Test Suite Runner (LavaPipe Vulkan Driver)
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}======================================================================${RESET}"
echo -e "${BOLD}${BLUE}  VNTX - End-to-End Headless LavaPipe Vulkan Test Suite Runner        ${RESET}"
echo -e "${BOLD}${BLUE}======================================================================${RESET}\n"

# 1. Locate LavaPipe ICD Manifest
LVP_ICD=""
if [ -f "/usr/share/vulkan/icd.d/lvp_icd.json" ]; then
    LVP_ICD="/usr/share/vulkan/icd.d/lvp_icd.json"
elif [ -f "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json" ]; then
    LVP_ICD="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
fi

if [ -z "$LVP_ICD" ]; then
    echo -e "${YELLOW}[!] Warning: LavaPipe ICD manifest not found in /usr/share/vulkan/icd.d/${RESET}"
    echo -e "${YELLOW}[!] To install Mesa Vulkan software driver on Ubuntu/Debian/WSL2:${RESET}"
    echo -e "${YELLOW}    sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan-dev${RESET}\n"
else
    echo -e "${GREEN}[✓] Detected LavaPipe ICD manifest:${RESET} ${LVP_ICD}"
    export VK_DRIVER_FILES="${LVP_ICD}"
    export VK_ICD_FILENAMES="${LVP_ICD}"
fi

export VK_LOADER_DEBUG=warn

# 2. Run Rust Backend Test Suite
echo -e "\n${BOLD}[1/3] Executing Rust Workspace Test Suite...${RESET}"
cargo test --workspace --all-targets

# 3. Configure and Build C++ Test Suite
echo -e "\n${BOLD}[2/3] Building C++ LavaPipe Headless Test Suite...${RESET}"
cmake -B build -DVNTX_BUILD_TESTS=ON
cmake --build build

# 4. Execute CTest Headless Suite
echo -e "\n${BOLD}[3/3] Running Headless CTest Suite via LavaPipe...${RESET}"
ctest --test-dir build --output-on-failure

echo -e "\n${BOLD}${GREEN}======================================================================${RESET}"
echo -e "${BOLD}${GREEN}  [✓] All VNTX WSL2 Headless & End-to-End Tests Passed Successfully!  ${RESET}"
echo -e "${BOLD}${GREEN}======================================================================${RESET}\n"
