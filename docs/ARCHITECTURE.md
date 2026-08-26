# NTC (Neural Texture Compression) - Architecture Specification (v1.0)

## 1. System High-Level Diagram

+-------------------------------------------------------------------------------+
|                             GAME PROCESS (User Space)                         |
|  (Native Vulkan Game OR Direct3D 12 Game running via Proton / VKD3D-Proton)  |
+-------------------------------------------------------------------------------+
                                       |
                   Calls Vulkan API (vkCreateImage, etc.)
                                       |
                                       v
+-------------------------------------------------------------------------------+
|              VULKAN LOADER FRAMEWORK (/usr/lib/libvulkan.so.1)                |
+-------------------------------------------------------------------------------+
                                       |
                   Discovers & Loads Implicit Layers
                                       |
                                       v
+-------------------------------------------------------------------------------+
|            NTC IMPLICIT VULKAN LAYER (libvk_ntc_layer.so) [OUR LAYER]         |
|                                                                               |
|  1. Filter Hook (vkCreateImage): Check dimensions & usage flags               |
|  2. Hash Check: Calculate xxHash3 of texture parameters                       |
|  3. Cache Lookup: Check ~/.cache/ntc/<game_hash>/<texture_hash>.ntc           |
|                                                                               |
|  [CACHE HIT]                                  [CACHE MISS]                    |
|  - Resize VkImage allocation to NTC size       - Forward original request      |
|  - Upload .ntc weights directly to VRAM        - Pass through to driver        |
|  - Modify SPIR-V Shaders (vkCreateShaderModule)                               |
|    to substitute texture sampling with                                        |
|    NTC Compute Shader inference                                               |
+-------------------------------------------------------------------------------+
                                       |
                   Passes Modified API Calls Downstream
                                       |
                                       v
+-------------------------------------------------------------------------------+
|              HARDWARE GPU DRIVER (NVIDIA / Mesa AMDGPU / NVK / Intel)         |
+-------------------------------------------------------------------------------+
                                       |
                   Executes Compute Commands on Hardware
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                   PHYSICAL GPU HARDWARE (e.g. RTX 3050 4GB)                   |
|  - VRAM: Stores .ntc neural weights (~2MB) instead of full texture (~32MB)   |
|  - Tensor Cores / Compute Units: Evaluates MLP inference shader on-the-fly    |
+-------------------------------------------------------------------------------+

---

## 2. Component Breakdown

### 2.1 Implicit Vulkan Layer (`libvntx_layer.so`)
The core interceptor library registered with Vulkan Loader via JSON manifest.
- **Manifest Location:** `/usr/share/vulkan/implicit_layer.d/vntx_layer.json`
- **Activation Control:** Activated automatically (100% Plug & Play). Can be disabled per-process via environment variable (`DISABLE_VNTX=1`).

### 2.2 Texture Candidate Filtering Engine
To prevent system instability or rendering corruption, the layer applies strict eligibility rules during `vkCreateImage` interception:

**ELEGIBILITY CRITERIA (ALL MUST BE TRUE):**
1. **Usage Flags:** `VK_IMAGE_USAGE_SAMPLED_BIT` MUST be present.
2. **Exclusion Flags:** MUST NOT contain:
   - `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` (Render Targets)
   - `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT` (Depth Buffers)
3. **Dimensions:** `width >= 1024` AND `height >= 1024`.
4. **Image Type:** MUST be `VK_IMAGE_TYPE_2D`.

If an image fails any filter criteria, the layer instantly forwards the `vkCreateImage` call down the layer chain without modification (Pass-Through mode).

### 2.3 Cache Lookup & Hash Matching Engine
- **Hash Generation:** Computes a 64-bit `xxHash3` checksum using image dimensions, format, mip levels, and raw buffer metadata provided during resource creation/upload.
- **Cache Path:** `~/.cache/ntc/<app_id>/<texture_hash>.ntc`
- **Validation:** Reads the 64-byte `NtcHeader` from disk. Checks `magic == "NTC1"`, `version == 1`, and matches `texture_hash`.

### 2.4 Allocation Resizing Engine (Cache Hit)
When a valid `.ntc` file is matched:
1. Intercepts `vkCreateImage` and modifies `VkImageCreateInfo`:
   - Adjusts extent to match NTC buffer requirements.
   - Adjusts memory allocation size via `vkAllocateMemory` hooks to allocate weight buffer memory.
2. Direct-maps weights from `.ntc` file directly into GPU VRAM via DMA staging memory.
3. Suppresses original uncompressed texture payload upload requests from application staging buffers.

### 2.5 SPIR-V Shader Transformation Engine
Intercepts `vkCreateShaderModule` during pipeline compilation:
1. Parses input SPIR-V bytecode using `SPIRV-Tools`.
2. Locates image sampling instructions (`OpImageSampleImplicitLod`, `OpImageSampleExplicitLod`) targeting resources flagged as NTC-compressed.
3. Replaces standard hardware texture fetch instructions with inlined function calls to the NTC inference shader module.

---

## 3. Shader Inference Engine (SPIR-V Module)

The inference shader evaluates a 3-layer MLP per sampled fragment:
- **Input:** Normalized UV coordinates `(u, v)` (2 floats).
- **Hidden Layers:** Matrix multiplications + activation function (ReLU / Sine).
- **Output:** Decoded pixel color `(r, g, b, a)` (4 floats).

### Execution Paths:
- **Path A (NVIDIA Tensor Cores):** Uses `VK_NV_cooperative_matrix` extension. Computes matrix multiplications across 16x16 warp fragments simultaneously on dedicated AI hardware. Near-zero impact on general CUDA shader cores.
- **Path B (Generic Fallback):** Uses `VK_EXT_shader_explicit_arithmetic_types_int16` / standard SIMD float16 vector math. Runs on standard Compute Units across AMD RDNA, Intel Arc, or legacy NVIDIA GPUs.

---

## 4. Fallback, Error Handling & Safety Guarantees

1. **File Corrupt or Missing (Cache Miss):** Instantly falls back to native Vulkan allocation. No impact on game execution.
2. **VRAM Allocation Failure:** If weight memory allocation fails, layer logs warning and falls back to uncompressed allocation.
3. **Shader Transformation Failure:** If SPIR-V patching fails for a complex custom shader, layer disables NTC for that specific draw call resource.