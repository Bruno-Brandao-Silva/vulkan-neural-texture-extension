# NTC Stack & Automated Testing Specification (v1.0)

## 1. Toolchain & Environment Specifications

The development environment and build tools are pinned to ensure maximum compatibility across Arch-based systems (CachyOS) and containerized build environments.

### Core Dependencies
- **C++ Compiler:** GCC 13+ or Clang 16+ (C++20 Standard strict conformance).
- **Rust Toolchain:** 1.75+ (Cargo Workspace edition 2021).
- **Vulkan SDK:** 1.3.260+ (Includes `vulkan/vulkan.h`, `glslangValidator`, and `SPIRV-Tools`).
- **Python Runtime:** Python 3.11+ (Used for PyTorch prototype scripts and SSIM visual quality verification).
- **Build Systems:** CMake >= 3.25 (For C++ layer and shaders) and Cargo (For Rust CLI).

---

## 2. Testing & Safety Architecture for AI Coding Agents

To allow automated AI coding assistants (such as Claude Code CLI) to write, compile, and execute test suites without risks of desktop freezing, GPU driver crashes, or system hangs, tests are strictly isolated into three tiers:

+-------------------------------------------------------------------------------+
|                       TIER 1: NATIVE UNIT TESTS                               |
|  - Cargo Test (`cargo test --workspace`)                                      |
|  - C++ GoogleTest (`ctest --test-dir build/`)                                 |
|  - Validates binary structs (NtcHeader 64-byte layout), xxHash3 matching,      |
|    and file I/O safely on host CPU in milliseconds.                           |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                 TIER 2: HEADLESS VULKAN TESTS (LavaPipe CPU)                  |
|  - Isolated Vulkan execution using Mesa's CPU driver (lvp_icd.x86_64.json)    |
|  - Tests vkCreateImage hooks, memory allocation overrides, and SPIR-V patching|
|  - Memory crashes or invalid pointers generate isolated SIGSEGV in terminal   |
|    without freezing host Wayland/X11 display or GPU hardware.                 |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                TIER 3: VISUAL FIDELITY GATES (Headless SSIM)                  |
|  - Headless C++ app renders NTC-compressed quads into an offscreen FrameBuffer|
|  - Python verification script computes SSIM (Structural Similarity Index)     |
|    and PSNR against native uncompressed texture renders.                      |
|  - Test fails if SSIM < 0.95.                                                 |
+-------------------------------------------------------------------------------+

---

## 3. Strict Execution Rules for AI Coding Agents

AI coding agents MUST adhere to these operational rules during code modification tasks:

1. **RULE 1 (LavaPipe Isolation):** NEVER execute the Vulkan Layer against physical GPU hardware (`/dev/dri/renderD128`) during automated build-test loops. Always use the LavaPipe CPU driver.
2. **RULE 2 (Sanitizer Requirement):** Debug builds of `libvk_ntc_layer.so` MUST be compiled with AddressSanitizer (`-fsanitize=address`) and UndefinedBehaviorSanitizer (`-fsanitize=undefined`) during Tier 1 and Tier 2 tests.
3. **RULE 3 (Completion Gate):** A code task is considered successfully completed ONLY when:
   - All Tier 1 unit tests pass with zero errors.
   - All Tier 2 LavaPipe Vulkan integration tests pass with zero memory leaks.
   - Tier 3 visual quality check achieves SSIM >= 0.95.

---

## 4. Standard Verification Commands

### Execute Tier 1 Unit Tests (Rust & C++)
cargo test --workspace
cmake -B build -S . -DNTC_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build --output-on-failure

### Execute Tier 2 Headless Vulkan Tests (LavaPipe)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
VK_INSTANCE_LAYERS=VK_LAYER_NTC_mesh \
./build/bin/ntc_headless_test

### Execute Tier 3 Visual Quality Verification
python3 scripts/verify_quality.py \
  --original tests/fixtures/sample_4k.png \
  --rendered build/outputs/rendered_ntc.png \
  --min-ssim 0.95