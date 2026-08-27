use vntx_core::{detect_gpu_hardware, get_recommended_settings, query_gpu_telemetry};

#[test]
fn test_detect_gpu_hardware_sanity() {
    let caps = detect_gpu_hardware();
    // optimal_precision must be either Fp16 or Int8
    assert!(
        caps.optimal_precision == vntx_core::NtcPrecision::Fp16
            || caps.optimal_precision == vntx_core::NtcPrecision::Int8
    );
}

#[test]
fn test_query_gpu_telemetry_fields() {
    let telemetry = query_gpu_telemetry();
    assert!(!telemetry.device_name.is_empty());
    assert!(telemetry.total_vram_mb > 0);
}

#[test]
fn test_get_recommended_settings_logic() {
    let rec = get_recommended_settings();
    assert!(!rec.recommended_quality.is_empty());
    assert!(!rec.recommended_precision.is_empty());
    assert!(!rec.reason.is_empty());
    assert!(!rec.guidance_box.is_empty());

    let caps = detect_gpu_hardware();
    if caps.has_tensor_cores {
        assert_eq!(rec.recommended_quality, "max-savings");
        assert_eq!(rec.recommended_precision, "int8");
    } else {
        assert_eq!(rec.recommended_quality, "balanced");
        assert_eq!(rec.recommended_precision, "fp16");
    }
}
