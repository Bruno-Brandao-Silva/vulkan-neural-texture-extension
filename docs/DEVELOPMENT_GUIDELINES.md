# VNTX Development Guidelines & Agent Rules (`DEVELOPMENT_GUIDELINES.md`)

## 1. Core Principles & General Standards

1. **English Standard:** All source code, identifiers, inline documentation, comments, commit messages, and pull requests MUST be written exclusively in English.
2. **Conventional Commits:** All git commits MUST follow the Conventional Commits 1.0.0 specification (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`).
3. **Latest Stable Dependencies:** Always utilize the most recent stable releases of external crates, libraries, and C++ toolchains. Avoid deprecated features or unmaintained packages.
4. **Zero-Tolerance Quality Bar:** Maintain production-grade code quality, strict typing, and full documentation starting from the very first code generation. No quick-and-dirty hacks or temporary workarounds.

---

## 2. Architecture, Design & Code Style

5. **Zero Magic Numbers:** Raw numeric literals in logic are strictly forbidden. All constant values MUST be explicitly named using `const` or `constexpr`.
6. **Explicit Overhead Awareness:** Every dynamic memory allocation (`malloc`, `new`, `Box`, `Vec`) MUST have a clear architectural justification. Prefer stack allocation, arena allocation, or buffer reuse whenever possible.
7. **Strict Type Safety:** Implicit numerical casting (e.g., converting `u64` to `u32` or `float` to `int` without bounds checks) is prohibited. Use explicit, safe conversions.
8. **Single Responsibility Principle (SRP):** Functions MUST NOT exceed 50 lines of code. Larger procedures MUST be decomposed into small, pure helper functions.
9. **No Global State:** Mutable global state is strictly banned. Use explicit dependency injection or instance contexts.
10. **Const-Correctness Absolute (C++):** All variables, pointers, references, and class methods that do not mutate state MUST be declared `const` (or `constexpr`).
11. **Explicit Ownership (Rust):** Avoid unnecessary `.clone()` invocations. Leverage references and borrows (`&`/`&mut`) by default.
12. **RAII Universal (C++):** Manual resource management is forbidden. All heap pointers, files, sockets, and Vulkan handles MUST be wrapped in RAII containers or smart pointers (`std::unique_ptr`, `std::shared_ptr`).

---

## 3. Error Handling & Logging

13. **No Silent Failures:** Catching exceptions or ignoring `Result`/`Option` with empty blocks or unvalidated `unwrap()` in production code is strictly forbidden.
14. **Contextual Error Messages:** Every returned error MUST provide rich context explaining where and why the failure occurred (e.g., leveraging `anyhow::Context` in Rust).
15. **Structured Logging:** Use appropriate log levels (`TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`). Standard stdout prints (`printf`, `println!`) for runtime debugging are prohibited in production builds.
16. **Graceful Degradation:** Internal failures within the Vulkan Layer MUST NEVER crash the host game application. Fallback to native Vulkan allocation MUST be immediate and transparent.
17. **Fail-Fast in Development:** Debug and test builds MUST aggressively use assertions (`assert!`, `assert()`) to validate invariants and pre-conditions.

---

## 4. Vulkan & Low-Level Practices

18. **Zero Specification Violation:** Vulkan API calls MUST NOT trigger any warnings or errors in the official Vulkan Validation Layers (`VK_LAYER_KHRONOS_validation`).
19. **Explicit Memory Alignments:** Structures mapped to GPU memory (Push Constants, Uniform Buffers, Storage Buffers) MUST strictly conform to `STD140` or `STD430` layout rules.
20. **No Dynamic Shader Compilation in Draw Loop:** SPIR-V transformation and injection MUST occur during pipeline or shader module creation (`vkCreateShaderModule`), NEVER during render/draw loops.
21. **C-ABI Strictness:** The public entry points of `libvntx_layer.so` MUST strictly expose a C-compatible ABI without C++ mangling, virtual tables, or complex types.

---

## 5. Testing & Quality Assurance

22. **Test-Driven AI Workflow:** Automated AI agents MUST generate unit tests (`gtest` or `cargo test`) prior to or alongside the implementation of new functionality.
23. **No Skipped Tests:** Disabled tests (`#[ignore]`, `DISABLED_`) are prohibited unless linked to an active issue tracking a temporary blocking upstream bug.
24. **Deterministic Tests:** All unit and integration tests MUST be 100% deterministic, with zero reliance on system clocks, unseeded random values, or race-prone concurrency.
25. **Zero Compiler Warnings:** All code bases MUST compile cleanly with maximum warning flags enabled (`-Wall -Wextra -Werror` for C++ and `#![deny(warnings)]` for Rust).

---

## 6. AI Agent Execution Rules

26. **No Unrequested Refactoring:** AI agents MUST NOT refactor, rename, or modify files outside the explicit scope of the current user prompt without prior authorization.
27. **Atomic Incremental Modifications:** AI agents MUST make small, logical, and incremental modifications, compiling and testing each step in the container before proceeding.
28. **Self-Correction Circuit Breaker:** If an AI agent fails compilation or test verification 3 times consecutively for the same root cause, it MUST halt, summarize the diagnostic logs, and request human intervention.
29. **Documentation Parity:** Whenever an AI agent alters a function signature, data structure, or CLI flag, it MUST update the corresponding documentation files in `./docs/` within the same task.
30. **No Placeholders or TODOs:** Generated code MUST be fully functional and production-ready. Stubbing methods with `// TODO: implement this` or `unimplemented!()` is strictly forbidden.

---

## 7. Repository, Security & Hygiene

31. **Atomic Commits:** Each commit MUST represent a single logical change and MUST leave the build system in a green, fully testable state.
32. **Clean Workspace Hygiene:** Temporary build artifacts, object files, binaries, and system files MUST NEVER be committed. The `.gitignore` file MUST be rigorously maintained.
33. **No Secrets or Hardcoded Paths:** Hardcoding absolute local filesystem paths (e.g., `/home/user/`) or secrets is strictly prohibited. Use environment variables or relative system paths.
34. **Minimal Dependencies:** Every new crate or external library dependency MUST be strictly justified. Prefer standard library solutions or pre-existing dependencies whenever feasible.
35. **Mandatory Pre-Commit Automated Formatting:** All developers and automated AI agents MUST execute `./scripts/fmt_and_fix.sh` (enforcing `cargo fmt`, `cargo clippy --fix`, and `clang-format`) as the mandatory final step before generating any git commit or pull request.