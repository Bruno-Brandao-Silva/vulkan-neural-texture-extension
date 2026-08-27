//! Hardware capability detection for GPU neural inference acceleration.

use crate::format::NtcPrecision;
use std::fs;
use std::path::Path;

/// Detected GPU hardware capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCapabilities {
    /// Whether an NVIDIA GPU is detected.
    pub is_nvidia: bool,
    /// Whether the GPU possesses Tensor Cores (Turing, Ampere, Ada Lovelace, Blackwell, Hopper, Volta).
    pub has_tensor_cores: bool,
    /// Optimal inference and storage precision for the detected hardware (`Int8` for Tensor Cores, `Fp16` fallback).
    pub optimal_precision: NtcPrecision,
}

/// Detects host GPU capabilities to select optimal inference & training precision.
///
/// If an NVIDIA GPU with Tensor Cores (Turing, Ampere, Ada Lovelace, Blackwell, Hopper, Volta, RTX series)
/// is detected, returns [`NtcPrecision::Int8`] to maximize VRAM savings and throughput.
/// Otherwise, defaults to [`NtcPrecision::Fp16`].
#[must_use]
pub fn detect_gpu_hardware() -> GpuCapabilities {
    let mut is_nvidia = Path::new("/proc/driver/nvidia").exists()
        || Path::new("/dev/nvidiactl").exists()
        || Path::new("/dev/nvidia0").exists();

    let mut has_tensor_cores = false;

    if let Ok(entries) = fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in entries.flatten() {
            let info_file = entry.path().join("information");
            if let Ok(content) = fs::read_to_string(info_file) {
                is_nvidia = true;
                let upper = content.to_uppercase();
                if upper.contains("RTX")
                    || upper.contains("TURING")
                    || upper.contains("AMPERE")
                    || upper.contains("ADA")
                    || upper.contains("BLACKWELL")
                    || upper.contains("HOPPER")
                    || upper.contains("VOLTA")
                    || upper.contains("A100")
                    || upper.contains("H100")
                    || upper.contains("T4")
                    || upper.contains("TITAN V")
                    || upper.contains("TITAN RTX")
                    || upper.contains("GTX 16")
                {
                    has_tensor_cores = true;
                    break;
                }
            }
        }
    }

    let optimal_precision = if has_tensor_cores {
        NtcPrecision::Int8
    } else {
        NtcPrecision::Fp16
    };

    GpuCapabilities {
        is_nvidia,
        has_tensor_cores,
        optimal_precision,
    }
}
