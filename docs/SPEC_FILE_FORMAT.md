# NTC Binary File Format Specification (.ntc v1.0)

## 1. Memory Layout & Endianness
All multi-byte fields in the `.ntc` binary container use **Little-Endian** byte order. 
The header layout uses tight C-compatible byte alignment (`#[repr(C)]` in Rust, `#pragma pack(push, 1)` in C/C++) to allow direct memory-mapped I/O (`mmap`) into Vulkan staging buffers without serialization overhead.

Total Fixed Header Size: **64 Bytes**

---

## 2. Fixed File Header (64 Bytes)

| Offset (Bytes) | Field Name | Type | Size | Description |
|---|---|---|---|---|
| 0x00 - 0x03 | `magic` | `u8[4]` | 4 Bytes | Magic Identifier Bytes. Must be ASCII `"NTC1"` (`[0x4E, 0x54, 0x43, 0x31]`). |
| 0x04 - 0x07 | `version` | `u32` | 4 Bytes | Format Version. Must be integer `1`. |
| 0x08 - 0x0F | `texture_hash` | `u64` | 8 Bytes | 64-bit `xxHash3` checksum of original raw texture payload. |
| 0x10 - 0x13 | `original_width` | `u32` | 4 Bytes | Original uncompressed texture width in pixels (e.g., `4096`). |
| 0x14 - 0x17 | `original_height` | `u32` | 4 Bytes | Original uncompressed texture height in pixels (e.g., `4096`). |
| 0x18 | `channels` | `u8` | 1 Byte | Color channel count: `3` = RGB, `4` = RGBA. |
| 0x19 | `precision` | `u8` | 1 Byte | Weight storage precision: `0` = Float16 (FP16), `1` = Int8 (INT8 Quantized). |
| 0x1A - 0x1B | `layers_count` | `u16` | 2 Bytes | Number of hidden layers in MLP (Default: `3`). |
| 0x1C - 0x1D | `hidden_dim` | `u16` | 2 Bytes | Neurons per hidden layer (Default: `64`). |
| 0x1E - 0x1F | `reserved_flags` | `u16` | 2 Bytes | Flags for future expansion (Default: `0x0000`). |
| 0x20 - 0x27 | `weights_offset` | `u64` | 8 Bytes | Absolute byte offset from file start to raw weight payload (Default: `64`). |
| 0x28 - 0x2F | `weights_size` | `u64` | 8 Bytes | Total payload size in bytes of raw neural weights. |
| 0x30 - 0x3F | `padding` | `u8[16]` | 16 Bytes | Zero-filled padding to align header to 64-byte boundary. |

---

## 3. C/C++ Header Definition

typedef struct #pragma pack(push, 1) {
    uint8_t  magic[4];          // "NTC1"
    uint32_t version;           // 1
    uint64_t texture_hash;      // xxHash3 64-bit
    uint32_t original_width;    // e.g. 4096
    uint32_t original_height;   // e.g. 4096
    uint8_t  channels;          // 3 or 4
    uint8_t  precision;         // 0 = FP16, 1 = INT8
    uint16_t layers_count;      // e.g. 3
    uint16_t hidden_dim;        // e.g. 64
    uint16_t reserved_flags;    // 0x0000
    uint64_t weights_offset;    // Offset to payload (64)
    uint64_t weights_size;      // Size in bytes
    uint8_t  padding[16];       // Padding to 64 bytes
} NtcHeader;
#pragma pack(pop)

---

## 4. Rust Struct Definition

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct NtcHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub texture_hash: u64,
    pub original_width: u32,
    pub original_height: u32,
    pub channels: u8,
    pub precision: u8,
    pub layers_count: u16,
    pub hidden_dim: u16,
    pub reserved_flags: u16,
    pub weights_offset: u64,
    pub weights_size: u64,
    pub padding: [u8; 16],
}

---

## 5. Weight Payload Structure

Directly following `weights_offset` (byte 64), weight matrices are stored contiguously in row-major order:

1. **Layer 1 Weights:** `[2, hidden_dim]` matrix (Input (U,V) -> Hidden 1)
2. **Layer 1 Biases:** `[hidden_dim]` vector
3. **Hidden Layers Weights (2 to N-1):** `[hidden_dim, hidden_dim]` matrices
4. **Hidden Layers Biases:** `[hidden_dim]` vectors
5. **Output Layer Weights:** `[hidden_dim, channels]` matrix (Hidden N -> Output RGB/RGBA)
6. **Output Layer Biases:** `[channels]` vector

No external compression (like gzip or zstd) is applied to the payload stream to allow the Vulkan Layer to upload data directly to VRAM via DMA mapped memory without CPU decompression latency.