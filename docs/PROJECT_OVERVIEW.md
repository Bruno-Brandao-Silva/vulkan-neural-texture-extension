# NTC (Neural Texture Compression) - Project Overview & Technical Requirements (v1.0)

## 1. Executive Summary
Neural Texture Compression (NTC) is an open-source, hardware-accelerated memory optimization framework for Linux. It addresses hardware VRAM constraints (specifically GPUs with 4GB-6GB VRAM, such as the NVIDIA RTX 3050 Mobile) by replacing heavy, uncompressed or block-compressed static textures with lightweight Multi-Layer Perceptron (MLP) neural networks.

These neural weights are evaluated on-the-fly directly on the GPU via Vulkan Compute Shaders, reducing static texture VRAM usage by 70% to 90% with minimal visual fidelity degradation.

---

## 2. Problem Statement
Modern 3D games targeting Direct3D 12 and Vulkan stream high-resolution textures (2K/4K) continuously into VRAM. When VRAM capacity is exceeded:
- **PCIe Bus Thrashing:** The GPU driver drops textures into System RAM via PCIe, causing severe micro-stuttering and frametime spikes.
- **Out Of Memory (OOM) Crashes:** The system terminates the application or the Vulkan driver returns `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
- **Texture Pop-in / Degraded Rendering:** Mipmaps fail to stream, resulting in blurry surfaces or missing materials.

Existing OS-level paging solutions suffer from high latency. NTC solves this at the texture allocation layer by reducing the physical size of texture assets before they hit VRAM.

---

## 3. Target Hardware & Environment Constraints

### Target Reference System
- **CPU:** AMD Ryzen 5 5600H (Cezanne)
- **GPU:** NVIDIA GeForce RTX 3050 Mobile (4GB VRAM)
- **Host OS:** Linux (CachyOS / Arch-based, Kernel 7.x)
- **Display Server / Desktop:** Wayland / X11 (KDE Plasma 6)
- **Graphics API:** Vulkan 1.3+ (Native & via Proton / DXVK / VKD3D-Proton)

### Hardware Requirements
- **Primary Execution Path:** NVIDIA Tensor Cores via `VK_NV_cooperative_matrix` (RTX 2000 Series or newer).
- **Fallback Execution Path:** Generic Float16 vector math (`VK_EXT_shader_explicit_arithmetic_types_int16`) for AMD RDNA / Intel Arc / Legacy GPUs.

---

## 4. Key Performance Indicators (KPIs) for v1.0
- **VRAM Reduction:** Reduce static texture footprint from 32MB (per 4K BC7 texture) to ~2MB-4MB per `.ntc` asset.
- **Frametime Stability:** Eliminate 1% low frame drops caused by VRAM paging over PCIe.
- **Inference Latency:** Keeps GPU inference execution time under 0.5ms per frame on Tensor Cores.
- **Zero Invasiveness:** Zero modification to native game executables or game asset files on disk.

---

## 5. System Boundary & Scope

### In-Scope for v1.0
- Offline extraction, hashing, and training pipeline (`ntc-cli` in Rust / PyTorch).
- Custom binary container specification (`.ntc`).
- Implicit Vulkan Layer (`libvk_ntc_layer.so`) intercepting image allocations globally.
- Runtime SPIR-V shader interception to evaluate MLPs using Vulkan Compute Shaders.
- Local hash-based cache lookup in `~/.cache/ntc/`.

### Out-of-Scope for v1.0
- Real-time texture training during gameplay (all training is Ahead-Of-Time / Offline).
- Dynamic render targets, depth buffers, UI textures, and writable compute storage images.
- Support for DirectX 11 / OpenGL legacy APIs without VKD3D / DXVK translation layers.