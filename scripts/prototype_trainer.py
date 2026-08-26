#!/usr/bin/env python3
"""
VNTX Prototype Texture Trainer & NTC v1.0 Binary Exporter.

This script trains a lightweight Multi-Layer Perceptron (MLP) on a 2D texture,
extracts the FP16 weights in row-major order, and packs them into a valid
.ntc binary container matching the VNTX v1.0 specification.
"""

import argparse
import math
import os
import struct
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import xxhash

# Fixed NTC Specification Constants
NTC_MAGIC = b"NTC1"
NTC_VERSION = 1
HEADER_SIZE_BYTES = 64
WEIGHTS_OFFSET_DEFAULT = 64
PADDING_SIZE_BYTES = 16
HEADER_STRUCT_FORMAT = "<4sIQIIBBHHHQQ16s"

PRECISION_FP16 = 0
PRECISION_INT8 = 1

DEFAULT_HIDDEN_DIM = 64
DEFAULT_LAYERS_COUNT = 3
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 65536


class NeuralTextureMLP(nn.Module):
    """3-layer or N-layer Multi-Layer Perceptron mapping (u, v) -> (r, g, b) or (r, g, b, a)."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, out_dim: int = 4, layers_count: int = 3):
        super().__init__()
        if layers_count < 2:
            raise ValueError(f"layers_count must be >= 2, got {layers_count}")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers_count = layers_count

        layers = []
        # Layer 1: Input (u, v) -> Hidden 1
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Hidden Layers (2 to N-1)
        for _ in range(layers_count - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output Layer: Hidden N -> Output Channels
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_linear_layers(self):
        """Returns all linear layers in the network in execution order."""
        return [layer for layer in self.net if isinstance(layer, nn.Linear)]


def generate_synthetic_texture(width: int = 1024, height: int = 1024, channels: int = 4) -> np.ndarray:
    """Generates a high-frequency synthetic test texture (UV gradient + radial waves)."""
    u = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    r = (0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * uu)).astype(np.float32)
    g = (0.5 + 0.5 * np.cos(2.0 * np.pi * 4.0 * vv)).astype(np.float32)
    b = (0.5 + 0.5 * np.sin(2.0 * np.pi * 6.0 * (uu + vv))).astype(np.float32)

    if channels == 3:
        img_array = np.stack([r, g, b], axis=-1)
    elif channels == 4:
        a = np.ones_like(r, dtype=np.float32)
        img_array = np.stack([r, g, b, a], axis=-1)
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    return (img_array * 255.0).clip(0, 255).astype(np.uint8)


def load_texture(image_path: str, channels: int = 4) -> np.ndarray:
    """Loads an image from disk and converts to uint8 array."""
    mode = "RGBA" if channels == 4 else "RGB"
    with Image.open(image_path) as img:
        img = img.convert(mode)
        return np.array(img, dtype=np.uint8)


def compute_xxhash3(raw_bytes: bytes) -> int:
    """Computes the 64-bit xxHash3 checksum of a raw byte buffer."""
    return xxhash.xxh3_64_intdigest(raw_bytes)


def create_uv_coordinate_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates normalized UV coordinates in [0.0, 1.0] range."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    uu, vv = np.meshgrid(u, v)
    coords = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    return coords


def train_mlp_texture(
    image_data: np.ndarray,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    layers_count: int = DEFAULT_LAYERS_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
) -> NeuralTextureMLP:
    """Trains the MLP on the input texture pixel coordinates."""
    height, width, channels = image_data.shape
    coords = create_uv_coordinate_grid(width, height)
    targets = (image_data.reshape(-1, channels).astype(np.float32)) / 255.0

    coords_tensor = torch.from_numpy(coords).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)

    model = NeuralTextureMLP(
        in_dim=2,
        hidden_dim=hidden_dim,
        out_dim=channels,
        layers_count=layers_count,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_pixels = coords.shape[0]
    num_batches = max(1, num_pixels // batch_size)

    print(f"[*] Starting training: {width}x{height} ({num_pixels} pixels), {epochs} epochs, device: {device}")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_pixels, device=device)
        total_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_coords = coords_tensor[idx]
            batch_targets = targets_tensor[idx]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        if epoch % 50 == 0 or epoch == epochs:
            psnr = -10.0 * math.log10(max(avg_loss, 1e-10))
            print(f"  [Epoch {epoch:04d}/{epochs:04d}] MSE Loss: {avg_loss:.6f} | Train PSNR: {psnr:.2f} dB")

    return model


def export_ntc_file(
    model: NeuralTextureMLP,
    raw_image_data: np.ndarray,
    output_path: str,
    precision: int = PRECISION_FP16,
) -> None:
    """Extracts weights in row-major order and writes 64-byte NTC container."""
    height, width, channels = raw_image_data.shape
    texture_hash = compute_xxhash3(raw_image_data.tobytes())

    linear_layers = model.get_linear_layers()

    # Collect row-major weight matrices and bias vectors
    weight_bytes = bytearray()

    for i, layer in enumerate(linear_layers):
        # PyTorch weight shape: (out_features, in_features)
        # NTC specification requires row-major: (in_features, out_features)
        # Transpose (.t()) makes it contiguous in (in_features, out_features) order
        w = layer.weight.detach().cpu().t().contiguous()
        b = layer.bias.detach().cpu().contiguous()

        if precision == PRECISION_FP16:
            w_bytes = w.to(torch.float16).numpy().tobytes()
            b_bytes = b.to(torch.float16).numpy().tobytes()
        else:
            raise NotImplementedError("INT8 quantization not implemented in prototype")

        weight_bytes.extend(w_bytes)
        weight_bytes.extend(b_bytes)

    weights_size = len(weight_bytes)
    padding = b"\x00" * PADDING_SIZE_BYTES

    header_bytes = struct.pack(
        HEADER_STRUCT_FORMAT,
        NTC_MAGIC,
        NTC_VERSION,
        texture_hash,
        width,
        height,
        channels,
        precision,
        model.layers_count,
        model.hidden_dim,
        0,  # reserved_flags
        WEIGHTS_OFFSET_DEFAULT,
        weights_size,
        padding,
    )

    if len(header_bytes) != HEADER_SIZE_BYTES:
        raise RuntimeError(f"Header size error: expected {HEADER_SIZE_BYTES}, got {len(header_bytes)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header_bytes)
        f.write(weight_bytes)

    print(f"[+] Successfully exported NTC file: {output_path}")
    print(f"    Header: {len(header_bytes)} bytes, Weights: {weights_size} bytes, Total: {len(header_bytes) + weights_size} bytes")
    print(f"    xxHash3: 0x{texture_hash:016x}")


def verify_ntc_file(ntc_path: str, original_image: np.ndarray, save_reconstruction_path: str = None) -> float:
    """Reads .ntc container, evaluates model inference, and computes visual PSNR."""
    with open(ntc_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE_BYTES)
        payload_bytes = f.read()

    (
        magic,
        version,
        texture_hash,
        width,
        height,
        channels,
        precision,
        layers_count,
        hidden_dim,
        reserved_flags,
        weights_offset,
        weights_size,
        padding,
    ) = struct.unpack(HEADER_STRUCT_FORMAT, header_bytes)

    assert magic == NTC_MAGIC, f"Invalid magic: {magic}"
    assert version == 1, f"Invalid version: {version}"
    assert weights_size == len(payload_bytes), f"Payload size mismatch: {weights_size} vs {len(payload_bytes)}"

    # Reconstruct PyTorch model from raw FP16 weights
    model = NeuralTextureMLP(in_dim=2, hidden_dim=hidden_dim, out_dim=channels, layers_count=layers_count)
    linear_layers = model.get_linear_layers()

    offset = 0
    for layer in linear_layers:
        in_f = layer.in_features
        out_f = layer.out_features

        # Weight: in_f * out_f elements (FP16 = 2 bytes)
        w_size = in_f * out_f * 2
        w_raw = payload_bytes[offset : offset + w_size]
        offset += w_size
        w_np = np.frombuffer(w_raw, dtype=np.float16).reshape((in_f, out_f))
        layer.weight.data = torch.from_numpy(w_np.T.astype(np.float32)).contiguous()

        # Bias: out_f elements
        b_size = out_f * 2
        b_raw = payload_bytes[offset : offset + b_size]
        offset += b_size
        b_np = np.frombuffer(b_raw, dtype=np.float16)
        layer.bias.data = torch.from_numpy(b_np.astype(np.float32)).contiguous()

    model.eval()

    coords = create_uv_coordinate_grid(width, height)
    with torch.no_grad():
        coords_t = torch.from_numpy(coords)
        preds_t = model(coords_t)
        rendered_np = (preds_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rendered_img = rendered_np.reshape((height, width, channels))

    # Compute PSNR
    orig_f = original_image.astype(np.float32)
    rend_f = rendered_img.astype(np.float32)
    mse = np.mean((orig_f - rend_f) ** 2)
    psnr = 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-10))

    print(f"[✓] NTC Verification Passed: Reconstruction PSNR = {psnr:.2f} dB (MSE = {mse:.4f})")

    if save_reconstruction_path:
        mode = "RGBA" if channels == 4 else "RGB"
        recon_image = Image.fromarray(rendered_img, mode=mode)
        os.makedirs(os.path.dirname(os.path.abspath(save_reconstruction_path)), exist_ok=True)
        recon_image.save(save_reconstruction_path)
        print(f"[+] Saved reconstructed image: {save_reconstruction_path}")

    return psnr


def main():
    parser = argparse.ArgumentParser(description="VNTX Neural Texture Trainer & Exporter")
    parser.add_argument("--input", type=str, help="Path to input texture image")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test texture")
    parser.add_argument("--width", type=int, default=1024, help="Width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (default: 1024)")
    parser.add_argument("--channels", type=int, default=4, choices=[3, 4], help="Channel count (3=RGB, 4=RGBA)")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Neurons per hidden layer (default: 64)")
    parser.add_argument("--layers-count", type=int, default=DEFAULT_LAYERS_COUNT, help="Total layer count (default: 3)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs (default: 300)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (default: 0.01)")
    parser.add_argument("--output", type=str, default="tests/fixtures/sample_texture.ntc", help="Output .ntc path")
    parser.add_argument("--verify", action="store_true", help="Verify exported .ntc file by reconstructing image")
    parser.add_argument("--save-reconstruction", type=str, help="Path to save reconstructed PNG")

    args = parser.parse_args()

    if args.input:
        image_data = load_texture(args.input, channels=args.channels)
    else:
        image_data = generate_synthetic_texture(width=args.width, height=args.height, channels=args.channels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_mlp_texture(
        image_data=image_data,
        hidden_dim=args.hidden_dim,
        layers_count=args.layers_count,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    export_ntc_file(model, image_data, args.output)

    if args.verify:
        verify_ntc_file(args.output, image_data, args.save_reconstruction)


if __name__ == "__main__":
    main()
