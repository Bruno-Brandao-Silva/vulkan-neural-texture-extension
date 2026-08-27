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

/// Real-time GPU and VRAM hardware telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GpuTelemetry {
    /// GPU Device name.
    pub device_name: String,
    /// Total VRAM in megabytes.
    pub total_vram_mb: u64,
    /// Used VRAM in megabytes.
    pub used_vram_mb: u64,
    /// Free VRAM in megabytes.
    pub free_vram_mb: u64,
    /// GPU core utilization percentage (0-100).
    pub gpu_utilization: u32,
    /// GPU temperature in degrees Celsius.
    pub temperature_c: u32,
    /// Whether hardware telemetry was queried successfully.
    pub is_available: bool,
}

/// Queries real-time GPU telemetry using `nvidia-smi` or fallback hardware detection.
#[must_use]
pub fn query_gpu_telemetry() -> GpuTelemetry {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                if let Some(line) = text.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                    if parts.len() >= 6 {
                        let name = parts[0].to_string();
                        let total = parts[1].parse::<u64>().unwrap_or(0);
                        let used = parts[2].parse::<u64>().unwrap_or(0);
                        let free = parts[3].parse::<u64>().unwrap_or(0);
                        let util = parts[4].parse::<u32>().unwrap_or(0);
                        let temp = parts[5].parse::<u32>().unwrap_or(0);

                        return GpuTelemetry {
                            device_name: name,
                            total_vram_mb: total,
                            used_vram_mb: used,
                            free_vram_mb: free,
                            gpu_utilization: util,
                            temperature_c: temp,
                            is_available: true,
                        };
                    }
                }
            }
        }
    }

    // Fallback: query capabilities
    let caps = detect_gpu_hardware();
    let device_name = if caps.is_nvidia {
        if caps.has_tensor_cores {
            "NVIDIA RTX / Tensor GPU (DirectX 12 / Vulkan)".to_string()
        } else {
            "NVIDIA GPU (Vulkan)".to_string()
        }
    } else {
        "Vulkan Compatible GPU".to_string()
    };

    GpuTelemetry {
        device_name,
        total_vram_mb: 8192,
        used_vram_mb: 0,
        free_vram_mb: 8192,
        gpu_utilization: 0,
        temperature_c: 0,
        is_available: false,
    }
}

/// Recommended hardware settings and engine guidance based on detected GPU capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedSettings {
    /// Recommended quality preset identifier ("fast", "balanced", "max-savings").
    pub recommended_quality: &'static str,
    /// Recommended target precision identifier ("fp16", "int8", "fp32").
    pub recommended_precision: &'static str,
    /// Technical explanation of why this configuration is recommended for the detected hardware.
    pub reason: String,
    /// User-friendly guidance summary for display in GUI system capabilities diagnostics.
    pub guidance_box: String,
}

/// Computes recommended preset and engine guidance based on host GPU architecture.
#[must_use]
pub fn get_recommended_settings() -> RecommendedSettings {
    let caps = detect_gpu_hardware();
    if caps.has_tensor_cores {
        RecommendedSettings {
            recommended_quality: "max-savings",
            recommended_precision: "int8",
            reason: "NVIDIA Tensor Cores detected: hardware INT8 matrix multiplication enables 4x VRAM reduction at maximum throughput.".to_string(),
            guidance_box: "Modo Ideal: INT8 Quantized (Max Savings) com aceleração por Tensor Cores. Máxima economia de VRAM com alta fidelidade visual.".to_string(),
        }
    } else if caps.is_nvidia {
        RecommendedSettings {
            recommended_quality: "balanced",
            recommended_precision: "fp16",
            reason: "NVIDIA GPU detected: FP16 Standard delivers high fidelity and solid VRAM savings.".to_string(),
            guidance_box: "Modo Ideal: FP16 Standard (Balanced) com 3 camadas MLP. Equilíbrio perfeito entre compressão e fidelidade visual em texturas 2K/4K.".to_string(),
        }
    } else {
        RecommendedSettings {
            recommended_quality: "balanced",
            recommended_precision: "fp16",
            reason: "Vulkan GPU detected: FP16 provides optimal compatibility and decompression speed.".to_string(),
            guidance_box: "Modo Ideal: FP16 Standard com Guardrails Anti-Stutter (2.5ms). Compatibilidade total e transcodificação suave no buffer de staging.".to_string(),
        }
    }
}


