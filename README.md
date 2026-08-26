# VNTX - Vulkan Neural Texture Compression Extension

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-orange.svg)](https://kernel.org)
[![API: Vulkan 1.3+](https://img.shields.io/badge/Vulkan-1.3%2B-red.svg)](https://www.vulkan.org/)
[![Language: C++20 / Rust 2021](https://img.shields.io/badge/Language-C%2B%2B20%20%2F%20Rust-brightgreen.svg)]()

**VNTX (`vulkan-neural-texture-extension`)** is an open-source, hardware-accelerated memory optimization framework for Linux. It addresses GPU VRAM constraints (specifically GPUs with 4GB–6GB VRAM) by replacing heavy static 2K/4K textures with lightweight Multi-Layer Perceptron (MLP) neural networks evaluated on-the-fly via Vulkan Compute Shaders.

By moving texture evaluation to the GPU's AI/Compute cores, VNTX reduces static texture VRAM footprint by **70% to 90%** with minimal visual fidelity degradation, eliminating PCIe thrashing and micro-stuttering in modern Direct3D 12 and Vulkan games.

---

## Key Features

- **Universal Vulkan Implicit Layer (`libvntx_layer.so`):** Intercepts texture allocations globally. Works transparently with native Linux Vulkan games and Windows games running via Proton / VKD3D-Proton / DXVK.
- **Dual Inference Engine:**
  - **Path A (NVIDIA Tensor Cores):** Leverages `VK_NV_cooperative_matrix` for near-zero latency decompression on hardware matrix cores.
  - **Path B (Generic SIMD Fallback):** Uses FP16 vector arithmetic (`VK_EXT_shader_explicit_arithmetic_types_int16`) for broad compatibility across AMD RDNA, Intel Arc, and legacy GPUs.
- **High-Performance CLI (`vntx-cli`):** Rust-based command-line tool for game library scanning, xxHash3 checksum generation, offline texture compression, and local cache management.
- **Zero-Invasiveness & Fallback Safety:** Modifies zero game files on disk. If a cache miss or allocation error occurs, VNTX instantly falls back to standard native Vulkan allocation without crashing the application.

---

## System Architecture Overview

```text
+-------------------------------------------------------------------------------+
|                             GAME PROCESS (User Space)                         |
|  (Native Vulkan Game OR Direct3D 12 Game running via Proton / VKD3D-Proton)  |
+-------------------------------------------------------------------------------+
                                       |
                   Calls Vulkan API (vkCreateImage, etc.)
                                       |
                                       v
+-------------------------------------------------------------------------------+
|            VNTX IMPLICIT VULKAN LAYER (libvntx_layer.so)                      |
|                                                                               |
|  1. Filter Hook (vkCreateImage): Check dimensions (>=1024x1024) & flags      |
|  2. Hash Check: Calculate xxHash3 of texture parameters                       |
|  3. Cache Lookup: Check ~/.cache/vntx/<app_id>/<texture_hash>.ntc             |
|                                                                               |
|  [CACHE HIT]                                  [CACHE MISS]                    |
|  - Resize VkImage allocation to NTC size       - Forward original request      |
|  - Upload .ntc weights directly to VRAM        - Pass through to driver        |
|  - Inject SPIR-V Compute Shader inference                                     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                   PHYSICAL GPU HARDWARE (e.g. RTX 3050 4GB)                   |
|  - VRAM: Stores .ntc neural weights (~2MB) instead of full texture (~32MB)   |
|  - Tensor Cores / Compute Units: Evaluates MLP inference shader on-the-fly    |
+-------------------------------------------------------------------------------+