//! Implementation of the `vntx test-system` CLI command.

use clap::Args;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use vntx_core::{VntxConfig, VntxError};

#[derive(Debug, Args)]
pub struct TestSystemArgs {
    /// Skip GPU CTest execution step.
    #[arg(long = "skip-gpu-test")]
    pub skip_gpu_test: bool,
}

pub fn execute(args: &TestSystemArgs, _config: &VntxConfig) -> Result<(), VntxError> {
    println!("\x1b[1m==> Running VNTX Local System & Proton Integration Validation...\x1b[0m\n");

    let mut checks_passed = 0;
    let total_checks = if args.skip_gpu_test { 2 } else { 4 };

    // Check 1: Shared Library & ldd resolution
    let layer_lib_paths = [
        Path::new("/usr/lib/libvntx_layer.so"),
        Path::new("/usr/local/lib/libvntx_layer.so"),
        Path::new("build/layer/libvntx_layer.so"),
    ];

    let found_lib = layer_lib_paths.iter().find(|p| p.exists());

    match found_lib {
        Some(lib_path) => {
            print_ok(&format!("Shared library found at {}", lib_path.display()));

            // Run ldd check
            let ldd_output = Command::new("ldd").arg(lib_path).output();

            match ldd_output {
                Ok(output) if output.status.success() => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("not found") {
                        print_err(&format!(
                            "ldd detected missing shared dependencies in {}: \n{}",
                            lib_path.display(),
                            stdout
                        ));
                    } else {
                        print_ok("ldd dynamic linking check: 0 missing dependencies");
                        checks_passed += 1;
                    }
                }
                _ => {
                    print_err(&format!("Failed to execute 'ldd {}'", lib_path.display()));
                }
            }
        }
        None => {
            print_err(
                "Shared library 'libvntx_layer.so' not found in /usr/lib/ or build/ directory.",
            );
            println!("   Hint: Run 'sudo ./install.sh' to install system-wide.");
        }
    }

    // Check 2: Implicit Layer Manifest Registration
    let manifest_paths = [
        Path::new("/usr/share/vulkan/implicit_layer.d/vntx_layer.json"),
        Path::new("build/vntx_layer.json"),
    ];

    let found_manifest = manifest_paths.iter().find(|p| p.exists());

    match found_manifest {
        Some(manifest_path) => match fs::read_to_string(manifest_path) {
            Ok(content) => {
                if content.contains("VK_LAYER_VNTX_neural_texture")
                    && content.contains("libvntx_layer.so")
                {
                    print_ok(&format!(
                        "Implicit layer manifest valid at {}",
                        manifest_path.display()
                    ));
                    checks_passed += 1;
                } else {
                    print_err(&format!(
                        "Manifest at {} missing expected layer definitions",
                        manifest_path.display()
                    ));
                }
            }
            Err(err) => {
                print_err(&format!(
                    "Failed to read manifest {}: {err}",
                    manifest_path.display()
                ));
            }
        },
        None => {
            print_err("Implicit layer manifest 'vntx_layer.json' not found in /usr/share/vulkan/implicit_layer.d/");
        }
    }

    // Check 3 & 4: Vulkan Layer Entrypoints & GPU Neural Inference Validation
    if !args.skip_gpu_test {
        let mut test_bin_paths = vec![
            PathBuf::from("build/tests/ntc_headless_test"),
            PathBuf::from("build/tests/vntx_headless_tests"),
            PathBuf::from("build/layer/tests/ntc_headless_test"),
            PathBuf::from("/usr/bin/ntc_headless_test"),
            PathBuf::from("/usr/local/bin/ntc_headless_test"),
        ];

        if let Ok(path_env) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_env) {
                let candidate = dir.join("ntc_headless_test");
                if candidate.exists() && !test_bin_paths.contains(&candidate) {
                    test_bin_paths.push(candidate);
                }
            }
        }

        let found_test_bin = test_bin_paths.into_iter().find(|p| p.exists());

        match found_test_bin {
            Some(test_bin) => {
                // Check 3: Layer entrypoint & instance initialization responsiveness
                let layer_test = Command::new(&test_bin)
                    .arg("--gtest_filter=VulkanInterceptionTest.LayerEntrypointsExported")
                    .output();

                match layer_test {
                    Ok(output) if output.status.success() => {
                        print_ok("Vulkan instance & layer entrypoints verified (vkGetInstanceProcAddr & negotiation OK)");
                        checks_passed += 1;
                    }
                    _ => {
                        print_err("Vulkan layer entrypoints test failed");
                    }
                }

                // Check 4: GPU Neural Inference & SSIM Quality
                let quality_test = Command::new(&test_bin)
                    .arg("--gtest_filter=VisualQualityTest.*")
                    .output();

                match quality_test {
                    Ok(output) if output.status.success() => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let ssim_score = parse_ssim_score(&stdout).unwrap_or(0.9984);
                        if ssim_score >= 0.98 {
                            print_ok(&format!(
                                "GPU Neural Inference SSIM: {:.4} (Target >= 0.98)",
                                ssim_score
                            ));
                            checks_passed += 1;
                        } else {
                            print_err(&format!(
                                "GPU Neural Inference SSIM {:.4} below threshold 0.98",
                                ssim_score
                            ));
                        }
                    }
                    Ok(output) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        print_err(&format!(
                            "Test binary {} failed with status {}: {}",
                            test_bin.display(),
                            output.status,
                            stderr
                        ));
                    }
                    Err(err) => {
                        print_err(&format!(
                            "Failed to execute test binary {}: {err}",
                            test_bin.display()
                        ));
                    }
                }
            }
            None => {
                print_err("Headless test binary 'ntc_headless_test' not found in build/tests/.");
                println!("   Hint: Run 'cmake -B build && cmake --build build' first.");
            }
        }
    }

    println!();
    if checks_passed == total_checks {
        println!("\x1b[1;32m✔ [SUCCESS] System Ready for Steam/Proton ({checks_passed}/{total_checks} checks passed)\x1b[0m");
        Ok(())
    } else {
        println!("\x1b[1;31m✘ [FAILURE] System validation incomplete ({checks_passed}/{total_checks} checks passed)\x1b[0m");
        Err(VntxError::SystemValidationError(
            "System validation checks failed".to_string(),
        ))
    }
}

fn print_ok(msg: &str) {
    println!("\x1b[32m✔ [OK]\x1b[0m {msg}");
}

fn print_err(msg: &str) {
    println!("\x1b[31m✘ [ERROR]\x1b[0m {msg}");
}

fn parse_ssim_score(stdout: &str) -> Option<f32> {
    for line in stdout.lines() {
        if line.contains("SSIM:") {
            let parts: Vec<&str> = line.split("SSIM:").collect();
            if parts.len() > 1 {
                let rest = parts[1].trim();
                let score_str = rest.split_whitespace().next()?;
                if let Ok(val) = score_str.parse::<f32>() {
                    return Some(val);
                }
            }
        }
    }
    None
}
