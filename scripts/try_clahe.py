#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import cv2
import numpy as np

from masking_ita import MaskingITAProcessor
from clahe_calibration import (
    apply_clahe,
    assign_ita_bracket,
    ita_to_beta,
    calculate_local_contrast_variance,
    calculate_noise,
    CLAHECalibrationConfig,
)


def process_and_compare(
    image_path: Path,
    output_dir: Path,
    custom_beta: float | None = None,
    tile_grid_size: tuple[int, int] = (8, 8),
):
    print("\n" + "=" * 60)
    print(f"Testing CLAHE on: {image_path.name}")
    print("=" * 60)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    processor = MaskingITAProcessor()
    ita_res = processor.process_image(img_rgb, filename=image_path.name)

    print(f"Masking Status     : {ita_res.status}")
    print(f"Mask Method        : {ita_res.mask_method}")
    print(f"Calculated ITA     : {ita_res.ita:.2f}°")
    print(f"ITA Bracket        : {ita_res.bracket}")
    print(f"Skin Mean L*       : {ita_res.mean_l:.2f}")
    print(f"Skin Mean b*       : {ita_res.mean_b:.2f}")

    beta_high, beta_mid, beta_low = 4.0, 3.0, 2.0
    adaptive_beta = ita_to_beta(ita_res.ita, beta_high=beta_high, beta_mid=beta_mid, beta_low=beta_low)

    if custom_beta is not None:
        effective_beta = custom_beta
        print(f"Applied Beta (User): {effective_beta:.2f} (clipLimit)")
    else:
        effective_beta = adaptive_beta
        print(f"Applied Beta (ITA) : {effective_beta:.2f} (clipLimit for {ita_res.bracket} bracket)")

    enhanced_rgb = apply_clahe(
        image_rgb=img_rgb,
        beta=effective_beta,
        tile_grid_size=tile_grid_size,
    )
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    h, w, _ = img_rgb.shape
    full_mask = np.ones((h, w), dtype=bool)

    raw_contrast = calculate_local_contrast_variance(img_rgb, full_mask)
    enhanced_contrast = calculate_local_contrast_variance(enhanced_rgb, full_mask)
    noise_score = calculate_noise(img_rgb, enhanced_rgb, full_mask)
    contrast_gain = ((enhanced_contrast - raw_contrast) / raw_contrast * 100.0) if raw_contrast > 0 else 0.0

    print(f"Raw Contrast Var   : {raw_contrast:.2f}")
    print(f"CLAHE Contrast Var : {enhanced_contrast:.2f} (+{contrast_gain:.1f}%)")
    print(f"Enhancement Shift  : {noise_score:.4f}")

    diff_rgb = cv2.absdiff(img_rgb, enhanced_rgb)
    diff_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY)
    diff_colored = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_VIRIDIS)
    diff_rgb_vis = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    h, w, _ = img_rgb.shape
    banner_height = 80
    canvas_w = w * 3
    canvas_h = h + banner_height

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:banner_height, :] = (24, 28, 36)
    canvas[banner_height:, :w] = img_rgb
    canvas[banner_height:, w : w * 2] = enhanced_rgb
    canvas[banner_height:, w * 2 :] = diff_rgb_vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = f"ITA-Guided L*-CLAHE | {image_path.name} | ITA: {ita_res.ita:.1f} deg ({ita_res.bracket}) | Beta: {effective_beta:.1f} | Gain: +{contrast_gain:.1f}%"
    cv2.putText(canvas, title_text, (20, 48), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    panel_y = banner_height + 30
    cv2.rectangle(canvas, (10, banner_height + 5), (220, banner_height + 40), (0, 0, 0), -1)
    cv2.putText(canvas, "1. ORIGINAL (RAW)", (15, panel_y), font, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (w + 10, banner_height + 5), (w + 290, banner_height + 40), (0, 0, 0), -1)
    cv2.putText(canvas, f"2. CLAHE (Beta={effective_beta:.1f})", (w + 15, panel_y), font, 0.65, (0, 255, 128), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (w * 2 + 10, banner_height + 5), (w * 2 + 280, banner_height + 40), (0, 0, 0), -1)
    cv2.putText(canvas, "3. CONTRAST BOOST HEATMAP", (w * 2 + 15, panel_y), font, 0.65, (255, 150, 50), 2, cv2.LINE_AA)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_comparison = output_dir / f"comparison_{image_path.stem}.jpg"
    out_enhanced = output_dir / f"clahe_{image_path.stem}.jpg"

    cv2.imwrite(str(out_comparison), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_enhanced), enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nSaved Enhanced Image : {out_enhanced}")
    print(f"Saved Comparison View: {out_comparison}")

    return {
        "image_path": str(image_path),
        "ita": ita_res.ita,
        "bracket": ita_res.bracket,
        "beta": effective_beta,
        "raw_contrast": raw_contrast,
        "enhanced_contrast": enhanced_contrast,
        "contrast_gain_pct": contrast_gain,
        "enhanced_path": str(out_enhanced),
        "comparison_path": str(out_comparison),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Try ITA-guided Adaptive L*-CLAHE enhancement on skin disease images."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to an image file. If omitted, runs a demonstration on sample images in frontend/assets.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Manual CLAHE clipLimit (e.g. 2.0, 3.5, 5.0). If omitted, uses ITA-adaptive beta.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase0_outputs/experiments/clahe_demo",
        help="Directory to save output comparison images (default: phase0_outputs/experiments/clahe_demo).",
    )

    args = parser.parse_args()
    output_dir = SCRIPT_DIR / args.output

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"File not found: {image_path}")
            sys.exit(1)
        process_and_compare(image_path, output_dir, custom_beta=args.beta)
    else:
        assets_dir = SCRIPT_DIR / "frontend" / "assets"
        sample_images = sorted(assets_dir.glob("sample_*.jpg"))
        if not sample_images:
            sample_images = sorted((SCRIPT_DIR / "pictures").glob("*.jpeg"))

        if not sample_images:
            print("No sample images found.")
            sys.exit(1)

        print(f"Found {len(sample_images)} sample images in {assets_dir}. Running CLAHE demo...")
        for img in sample_images:
            process_and_compare(img, output_dir, custom_beta=args.beta)


if __name__ == "__main__":
    main()
