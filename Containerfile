# Dev & Test Environment for vulkan-neural-texture-extension (VNTX)
FROM archlinux:latest

# Update system & install dependencies for C++20, Rust, Vulkan SDK, and LavaPipe
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        base-devel \
        cmake \
        ninja \
        clang \
        llvm \
        git \
        rust \
        vulkan-devel \
        vulkan-headers \
        vulkan-icd-loader \
        vulkan-tools \
        vulkan-swrast \
        glslang \
        spirv-tools \
        python \
        python-pip \
        gtest \
        valgrind \
        pkgconf && \
    pacman -Scc --noconfirm

# Configure LavaPipe as the default CPU Vulkan driver for headless testing
ENV VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
ENV RUST_BACKTRACE=1

WORKDIR /workspace

# Default command runs Tier 1 & Tier 2 tests
CMD ["bash"]
