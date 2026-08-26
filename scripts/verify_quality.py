#!/usr/bin/env python3
"""
VNTX Visual Quality Verification Gate (Tier 3).

Calculates Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR)
between uncompressed reference render outputs and NTC neural reconstructed outputs.
Fails with exit code 1 if visual fidelity drops below the specified SSIM threshold.
"""

import argparse
import sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr


def evaluate_visual_quality(
    reference_path: str,
    rendered_path: str,
    min_ssim: float = 0.95,
    min_psnr: float = 20.0,
) -> bool:
    print(f"[*] Loading reference image: {reference_path}")
    print(f"[*] Loading rendered image:  {rendered_path}")

    with Image.open(reference_path) as img_ref, Image.open(rendered_path) as img_rend:
        img_ref_rgb = img_ref.convert("RGB")
        img_rend_rgb = img_rend.convert("RGB")

        if img_ref_rgb.size != img_rend_rgb.size:
            print(
                f"[!] Dimension mismatch: reference is {img_ref_rgb.size}, rendered is {img_rend_rgb.size}",
                file=sys.stderr,
            )
            return False

        arr_ref = np.array(img_ref_rgb, dtype=np.float32) / 255.0
        arr_rend = np.array(img_rend_rgb, dtype=np.float32) / 255.0

    ssim_val = float(
        compute_ssim(
            arr_ref,
            arr_rend,
            channel_axis=2,
            data_range=1.0,
        )
    )

    psnr_val = float(
        compute_psnr(
            arr_ref,
            arr_rend,
            data_range=1.0,
        )
    )

    print("\n=======================================================")
    print("           VNTX VISUAL FIDELITY REPORT                ")
    print("=======================================================")
    print(f"  Structural Similarity (SSIM): {ssim_val:.4f}  (Min Required: {min_ssim:.4f})")
    print(f"  Peak Signal-to-Noise (PSNR):  {psnr_val:.2f} dB (Min Required: {min_psnr:.2f} dB)")
    print("=======================================================")

    passed = ssim_val >= min_ssim
    if passed:
        print(f"[✓] SUCCESS: SSIM {ssim_val:.4f} >= {min_ssim:.4f} (Visual Gate Passed)")
    else:
        print(
            f"[✗] FAILURE: SSIM {ssim_val:.4f} < {min_ssim:.4f} (Visual Degradation Detected)",
            file=sys.stderr,
        )

    return passed


def main():
    parser = argparse.ArgumentParser(description="VNTX Visual Quality Verification Gate")
    parser.add_argument("--original", required=True, help="Path to reference uncompressed image")
    parser.add_argument("--rendered", required=True, help="Path to rendered NTC test image")
    parser.add_argument("--min-ssim", type=float, default=0.95, help="Minimum acceptable SSIM threshold (default: 0.95)")
    parser.add_argument("--min-psnr", type=float, default=10.0, help="Minimum acceptable PSNR threshold (default: 10.0)")

    args = parser.parse_args()

    success = evaluate_visual_quality(
        reference_path=args.original,
        rendered_path=args.rendered,
        min_ssim=args.min_ssim,
        min_psnr=args.min_psnr,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
