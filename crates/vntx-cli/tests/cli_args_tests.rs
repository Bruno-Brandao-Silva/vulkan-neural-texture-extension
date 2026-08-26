#![allow(clippy::all, clippy::pedantic)]

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_help_flag() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Vulkan games"))
        .stdout(predicate::str::contains("scan"))
        .stdout(predicate::str::contains("compress"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("clean"));
}

#[test]
fn test_cli_scan_help() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.args(["scan", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--game"))
        .stdout(predicate::str::contains("--min-size"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn test_cli_compress_help() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.args(["compress", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--game"))
        .stdout(predicate::str::contains("--quality"))
        .stdout(predicate::str::contains("--jobs"))
        .stdout(predicate::str::contains("--output"));
}

#[test]
fn test_cli_status_command() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("VNTX SYSTEM & CACHE STATUS"))
        .stdout(predicate::str::contains("Cache Directory:"))
        .stdout(predicate::str::contains("Vulkan Implicit Layer:"));
}

#[test]
fn test_cli_clean_all_command() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.args(["clean", "--all"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Purged all cached NTC files"));
}

#[test]
fn test_cli_fetch_command() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.args(["fetch", "-g", "Cyberpunk 2077"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Fetching pre-trained NTC textures",
        ));
}

#[test]
fn test_cli_test_system_help() {
    let mut cmd = Command::cargo_bin("vntx").expect("binary exists");
    cmd.args(["test-system", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--skip-gpu-test"));
}
