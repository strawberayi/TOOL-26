"""
MEMBER 2 - CLAHE CALIBRATION AND EXPORT
=======================================

Responsibilities:
    #9  ITA -> beta piecewise mapping
    #10 CLAHE beta calibration
    #11 Mask and CLAHE validation / QC sheet
    #12 Final frozen phase0_calibration.json

This module DOES NOT perform:
    - CIELAB conversion for ITA
    - K-Means masking
    - Silhouette selection
    - Skin-mask generation
    - ITA calculation

Those are handled by Member 1's masking_ita.py.

INPUT:
    A manifest CSV containing successfully processed images.

Required manifest columns:
    image_id
    image_path
    mask_path
    ITA
    bracket
    calibration_eligible

Recommended additional columns:
    disease_label
    mask_method
    selected_k
    silhouette_score
    mask_area_percent

OUTPUT:
    phase0_outputs/
        beta_search_results.csv
        mask_clahe_validation.csv
        image_ita_manifest.csv
        phase0_calibration.json
        clahe_all_manifest.csv
        clahe_all/
            Darkest/
            Medium/
            Lightest/
        clahe_samples/
"""


from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CLAHECalibrationConfig:
    """
    Configuration for Member 2.

    NOTE:
    The assignment specifies the beta search range 2.0-10.0,
    but the final noise limit, contrast formula, and tile-grid
    size must be agreed upon by the research team.

    Therefore these values remain configurable.
    """

    # --------------------------------------------------------
    # #10 BETA SEARCH
    # --------------------------------------------------------

    beta_start: float = 2.0
    beta_end: float = 10.0
    beta_step: float = 0.5

    # --------------------------------------------------------
    # CLAHE TILE GRID
    # --------------------------------------------------------

    tile_grid_width: int = 8
    tile_grid_height: int = 8

    # --------------------------------------------------------
    # NOISE LIMIT
    # --------------------------------------------------------
    #
    # PROVISIONAL VALUE.
    # Replace after team/adviser approval.
    #

    noise_limit: float = 0.15

    # --------------------------------------------------------
    # CONTRAST REQUIREMENT
    # --------------------------------------------------------
    #
    # PROVISIONAL VALUE.
    # Replace after team/adviser approval.
    #

    minimum_contrast_gain: float = 0.0

    # --------------------------------------------------------
    # ACCEPTANCE RATE
    # --------------------------------------------------------
    #
    # A beta must pass the noise criterion on this proportion
    # of images in the bracket.
    #

    minimum_acceptance_rate: float = 0.80

    # --------------------------------------------------------
    # BETA SELECTION RULE
    # --------------------------------------------------------
    #
    # "knee"     : point of diminishing contrast returns (default)
    # "max_gain" : original rule, largest admissible contrast gain
    #

    selection_rule: str = "knee"

    # --------------------------------------------------------
    # RANDOMNESS
    # --------------------------------------------------------

    random_seed: int = 42

    # --------------------------------------------------------
    # VALIDATION SAMPLE
    # --------------------------------------------------------

    validation_samples_per_stratum: int = 5

    # --------------------------------------------------------
    # DATASET VERSION
    # --------------------------------------------------------

    calibration_dataset_version: str = "public_training_data"


# ============================================================
# #9 ITA BRACKET
# ============================================================

def assign_ita_bracket(ita: float) -> str:
    """
    Piecewise ITA bracket.

    Darkest:
        ITA < 28

    Medium:
        28 <= ITA <= 41

    Lightest:
        ITA > 41
    """

    if ita < 28.0:
        return "Darkest"

    elif ita <= 41.0:
        return "Medium"

    else:
        return "Lightest"


# ============================================================
# #9 ITA -> BETA MAPPING
# ============================================================

def ita_to_beta(
    ita: float,
    beta_high: float,
    beta_mid: float,
    beta_low: float,
) -> float:
    """
    Piecewise mapping:

        ITA < 28       -> beta_high
        28 <= ITA <= 41 -> beta_mid
        ITA > 41       -> beta_low
    """

    bracket = assign_ita_bracket(ita)

    if bracket == "Darkest":
        return beta_high

    elif bracket == "Medium":
        return beta_mid

    elif bracket == "Lightest":
        return beta_low

    raise ValueError(
        f"Invalid ITA bracket: {bracket}"
    )


# ============================================================
# BETA GRID
# ============================================================

def generate_beta_grid(
    config: CLAHECalibrationConfig
) -> list[float]:
    """
    Generate beta values from 2.0 to 10.0.

    Example with step 0.5:

        2.0
        2.5
        3.0
        ...
        9.5
        10.0
    """

    values = np.arange(
        config.beta_start,
        config.beta_end + config.beta_step / 2,
        config.beta_step,
    )

    return [
        round(float(value), 4)
        for value in values
    ]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb_image(
    image_path: str | Path
) -> np.ndarray:
    """
    Load image as RGB.

    Primary loader:
        OpenCV

    Fallback loader:
        Pillow (PIL)

    Some valid JPEG files can be opened by Pillow but fail
    with cv2.imread() because of decoder/encoding differences.
    The Pillow fallback keeps those valid images in the same
    CLAHE pipeline instead of marking them as failed.
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # PRIMARY: OpenCV
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is not None:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # --------------------------------------------------------
    # FALLBACK: Pillow
    # --------------------------------------------------------

    try:
        with Image.open(image_path) as pil_image:
            pil_image = pil_image.convert("RGB")
            image_rgb = np.asarray(
                pil_image,
                dtype=np.uint8
            )

        # Make sure the array is contiguous for OpenCV operations.
        image_rgb = np.ascontiguousarray(image_rgb)

        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.size == 0
        ):
            raise ValueError(
                f"Invalid RGB image shape: {image_rgb.shape}"
            )

        return image_rgb

    except Exception as exc:
        raise FileNotFoundError(
            f"Cannot read image with OpenCV or Pillow: "
            f"{image_path} | {exc}"
        ) from exc


# ============================================================
# MASK LOADING
# ============================================================

def load_mask(
    mask_path: str | Path
) -> np.ndarray:
    """
    Load a binary mask.

    Any non-zero pixel becomes True.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"Cannot read mask: {mask_path}"
        )

    return mask > 0


# ============================================================
# #10 CLAHE
# ============================================================

def apply_clahe(
    image_rgb: np.ndarray,
    beta: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """
    Apply CLAHE to the L channel only.

    Pipeline:

        RGB
         ↓
        OpenCV LAB
         ↓
        L / A / B
         ↓
        CLAHE on L
         ↓
        recombine A + B
         ↓
        RGB
    """

    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    )

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(beta),
        tileGridSize=tile_grid_size
    )

    enhanced_L = clahe.apply(L)

    enhanced_lab = cv2.merge(
        [
            enhanced_L,
            A,
            B
        ]
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced_rgb


# ============================================================
# L CHANNEL
# ============================================================

def get_l_channel(
    image_rgb: np.ndarray
) -> np.ndarray:
    """
    Extract OpenCV LAB L channel.

    This is used for contrast measurement.
    """

    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    )

    return lab[:, :, 0].astype(
        np.float32
    )


# ============================================================
# #10 LOCAL CONTRAST
# ============================================================

def calculate_local_contrast_variance(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Calculate local contrast variance inside the valid skin mask.

    Local contrast is defined here as:

        C(x,y) = L(x,y) - GaussianBlur(L)(x,y)

    Then:

        Local Contrast Variance =
            Var(C(x,y))

    evaluated only inside the valid mask.

    IMPORTANT:
    The assignment requires the team to define/document the exact
    local-contrast variance formula. This implementation provides
    a transparent, configurable baseline rather than claiming that
    this formula has already been approved by the research team.
    """

    L = get_l_channel(image_rgb)

    # Smooth luminance
    blurred = cv2.GaussianBlur(
        L,
        ksize=(5, 5),
        sigmaX=0
    )

    # Local contrast
    local_contrast = (
        L - blurred
    )

    valid = local_contrast[
        mask
    ]

    if valid.size == 0:
        return float("nan")

    return float(
        np.var(valid)
    )


# ============================================================
# #10 NOISE
# ============================================================

def calculate_noise(
    raw_image_rgb: np.ndarray,
    enhanced_image_rgb: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Estimate enhancement-induced change inside the skin mask.

    The metric is:

        mean(|L_enhanced - L_raw|) / 255

    This is a screening metric for the beta search.

    The final noise definition must be documented/approved by
    the research team.
    """

    raw_L = get_l_channel(
        raw_image_rgb
    )

    enhanced_L = get_l_channel(
        enhanced_image_rgb
    )

    difference = np.abs(
        enhanced_L - raw_L
    )

    valid = difference[
        mask
    ]

    if valid.size == 0:
        return float("nan")

    return float(
        np.mean(valid) / 255.0
    )


# ============================================================
# #10 EVALUATE ONE BETA
# ============================================================

def evaluate_beta(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    beta: float,
    config: CLAHECalibrationConfig,
) -> dict:
    """
    Evaluate one beta value for one image.
    """

    enhanced = apply_clahe(
        image_rgb=image_rgb,
        beta=beta,
        tile_grid_size=(
            config.tile_grid_width,
            config.tile_grid_height,
        ),
    )

    raw_contrast = (
        calculate_local_contrast_variance(
            image_rgb,
            mask
        )
    )

    enhanced_contrast = (
        calculate_local_contrast_variance(
            enhanced,
            mask
        )
    )

    noise = calculate_noise(
        raw_image_rgb=image_rgb,
        enhanced_image_rgb=enhanced,
        mask=mask,
    )

    contrast_gain = (
        enhanced_contrast
        - raw_contrast
    )

    # --------------------------------------------------------
    # ACCEPT / REJECT
    # --------------------------------------------------------

    accepted = (
        np.isfinite(noise)
        and np.isfinite(contrast_gain)
        and noise <= config.noise_limit
        and contrast_gain >= config.minimum_contrast_gain
    )

    if noise > config.noise_limit:
        reason = "REJECT_NOISE_LIMIT"

    elif contrast_gain < config.minimum_contrast_gain:
        reason = "REJECT_CONTRAST_GAIN"

    else:
        reason = "ACCEPT"

    return {
        "beta": float(beta),
        "raw_local_contrast_variance":
            float(raw_contrast),
        "clahe_local_contrast_variance":
            float(enhanced_contrast),
        "contrast_gain":
            float(contrast_gain),
        "noise":
            float(noise),
        "accepted":
            bool(accepted),
        "decision_reason":
            reason,
    }


# ============================================================
# #10 CALIBRATE ONE ITA BRACKET
# ============================================================

def calibrate_bracket(
    dataframe: pd.DataFrame,
    bracket: Optional[str],
    config: CLAHECalibrationConfig,
):
    """
    Search beta values for one ITA bracket.

    bracket=None pools every calibration-eligible image regardless of
    ITA. That pooled result is the single "global" beta used by the
    fixed-beta ablation control, so the control and the proposed model
    are calibrated by exactly the same procedure.

    Each image and mask is loaded once and evaluated for every beta.
    """

    eligible = dataframe["calibration_eligible"].astype(str).str.lower() == "true"
    if bracket is None:
        subset = dataframe[eligible].copy()
        label = "ALL"
    else:
        subset = dataframe[(dataframe["bracket"] == bracket) & eligible].copy()
        label = bracket

    if subset.empty:
        raise RuntimeError(
            f"No calibration-eligible images "
            f"found for bracket '{label}'."
        )

    beta_values = generate_beta_grid(
        config
    )

    all_results = []

    for _, row in subset.iterrows():

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        base = {
            "image_id": row["image_id"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "disease_label": row.get("disease_label", ""),
            "ITA": float(row["ITA"]),
            "bracket": label,
            "mask_method": row.get("mask_method", ""),
            "selected_k": row.get("selected_k", ""),
        }

        try:
            image = load_rgb_image(image_path)
            mask = load_mask(mask_path)

            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    "Image and mask dimensions "
                    "do not match."
                )

        except Exception as exc:
            for beta in beta_values:
                all_results.append({
                    **base,
                    "beta": beta,
                    "raw_local_contrast_variance": np.nan,
                    "clahe_local_contrast_variance": np.nan,
                    "contrast_gain": np.nan,
                    "noise": np.nan,
                    "accepted": False,
                    "decision_reason": f"PROCESSING_ERROR: {exc}",
                })
            continue

        for beta in beta_values:
            result = evaluate_beta(
                image_rgb=image,
                mask=mask,
                beta=beta,
                config=config,
            )
            all_results.append({**base, **result})

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = summarize_beta_search(
        results_df,
        label,
        config
    )

    best_beta = select_beta(
        summary_df,
        config
    )

    summary_df["selected"] = summary_df["beta"] == best_beta

    return (
        best_beta,
        results_df,
        summary_df,
    )


# ============================================================
# #10 SUMMARIZE EACH BETA
# ============================================================

def summarize_beta_search(
    results_df: pd.DataFrame,
    label: str,
    config: CLAHECalibrationConfig,
) -> pd.DataFrame:
    """
    Aggregate per-image beta results into one row per beta.
    """

    summaries = []

    for beta in generate_beta_grid(config):

        current = results_df[
            results_df["beta"] == beta
        ]

        accepted_count = int(
            current["accepted"]
            .fillna(False)
            .sum()
        )

        total_count = len(current)

        summaries.append({
            "bracket": label,
            "beta": float(beta),
            "mean_contrast_gain": float(current["contrast_gain"].mean()),
            "mean_noise": float(current["noise"].mean()),
            "accepted_images": accepted_count,
            "total_images": total_count,
            "acceptance_rate": (
                accepted_count / total_count
                if total_count > 0
                else 0.0
            ),
        })

    return pd.DataFrame(
        summaries
    )


# ============================================================
# #10 SELECT BETA FROM A BRACKET SUMMARY
# ============================================================

def select_beta(
    summary_df: pd.DataFrame,
    config: CLAHECalibrationConfig,
) -> float:
    """
    Choose one beta from the admissible betas of a bracket.

    Admissible: acceptance_rate >= minimum_acceptance_rate.

    "knee" (default):
        Local contrast rises monotonically with the clip limit, so
        "highest contrast gain" always selects the largest admissible
        beta, i.e. whatever beta the noise ceiling happens to allow.
        That makes every bracket collapse to the same value.

        Instead, locate the point of diminishing returns on the
        admissible mean-contrast-gain curve (Kneedle: after scaling
        beta and gain to [0, 1], maximise gain_norm - beta_norm).
        Brackets whose contrast responds differently to CLAHE get
        different betas. Ties prefer the smaller beta.

    "max_gain":
        The original rule: highest mean contrast gain, ties broken by
        lower mean noise.
    """

    candidates = summary_df[
        summary_df["acceptance_rate"]
        >= config.minimum_acceptance_rate
    ].sort_values("beta")

    if candidates.empty:
        raise RuntimeError(
            f"No beta passed the acceptance criterion for "
            f"{summary_df['bracket'].iloc[0]}."
        )

    if config.selection_rule == "max_gain" or len(candidates) < 3:
        best = candidates.sort_values(
            by=["mean_contrast_gain", "mean_noise"],
            ascending=[False, True],
        ).iloc[0]
        return float(best["beta"])

    if config.selection_rule != "knee":
        raise ValueError(
            f"Unknown selection_rule: {config.selection_rule}"
        )

    beta = candidates["beta"].to_numpy(dtype=float)
    gain = candidates["mean_contrast_gain"].to_numpy(dtype=float)

    gain_span = gain.max() - gain.min()
    if not np.isfinite(gain_span) or gain_span <= 0:
        # Flat curve: extra clipping buys nothing, use the gentlest beta.
        return float(beta[0])

    beta_norm = (beta - beta[0]) / (beta[-1] - beta[0])
    gain_norm = (gain - gain.min()) / gain_span

    # argmax returns the first (smallest-beta) index on ties.
    return float(beta[int(np.argmax(gain_norm - beta_norm))])



# ============================================================
# #11 CREATE VALIDATION SAMPLE
# ============================================================

def create_validation_sample(
    dataframe: pd.DataFrame,
    config: CLAHECalibrationConfig,
) -> pd.DataFrame:
    """
    Create a stratified validation sample.

    Preferred strata:
        disease_label + bracket

    This helps ensure the QC sample covers different disease
    classes and ITA brackets.
    """

    rng = np.random.default_rng(
        config.random_seed
    )

    df = dataframe[
        dataframe["calibration_eligible"].astype(str).str.lower() == "true"
    ].copy()

    if df.empty:
        return df

    # --------------------------------------------------------
    # If disease_label exists, stratify by disease + bracket.
    # Otherwise stratify by bracket only.
    # --------------------------------------------------------

    if (
        "disease_label" in df.columns
        and
        df["disease_label"].notna().any()
    ):

        grouped = df.groupby(
            [
                "disease_label",
                "bracket"
            ],
            dropna=False
        )

    else:

        grouped = df.groupby(
            ["bracket"],
            dropna=False
        )

    selected_rows = []

    for _, group in grouped:

        indices = group.index.to_numpy()

        sample_size = min(
            config.validation_samples_per_stratum,
            len(indices)
        )

        selected = rng.choice(
            indices,
            size=sample_size,
            replace=False
        )

        selected_rows.extend(
            selected.tolist()
        )

    return df.loc[
        selected_rows
    ].copy()


# ============================================================
# #11 SAVE VISUAL QC SAMPLES
# ============================================================

def save_qc_samples(
    validation_df: pd.DataFrame,
    beta_mapping: dict,
    output_dir: str | Path,
    config: CLAHECalibrationConfig,
) -> pd.DataFrame:
    """
    Save raw image, mask overlay, and CLAHE output for the
    validation sample.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for _, row in validation_df.iterrows():

        image_id = str(
            row["image_id"]
        )

        image_path = row[
            "image_path"
        ]

        mask_path = row[
            "mask_path"
        ]

        bracket = row[
            "bracket"
        ]

        beta = beta_mapping[
            bracket
        ]

        try:

            image = load_rgb_image(
                image_path
            )

            mask = load_mask(
                mask_path
            )

            enhanced = apply_clahe(
                image,
                beta,
                (
                    config.tile_grid_width,
                    config.tile_grid_height
                )
            )

            # ------------------------------------------------
            # MASK OVERLAY
            # ------------------------------------------------

            overlay = image.copy()

            # Create visible mask overlay
            overlay[mask] = (
                0.5 * overlay[mask]
                + 0.5 * np.array(
                    [0, 255, 0],
                    dtype=np.float32
                )
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            raw_path = (
                output_dir
                / f"{image_id}_raw.png"
            )

            mask_overlay_path = (
                output_dir
                / f"{image_id}_mask.png"
            )

            clahe_path = (
                output_dir
                / f"{image_id}_clahe.png"
            )

            cv2.imwrite(
                str(raw_path),
                cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )
            )

            cv2.imwrite(
                str(mask_overlay_path),
                cv2.cvtColor(
                    overlay,
                    cv2.COLOR_RGB2BGR
                )
            )

            cv2.imwrite(
                str(clahe_path),
                cv2.cvtColor(
                    enhanced,
                    cv2.COLOR_RGB2BGR
                )
            )

            records.append({

                "image_id":
                    image_id,

                "disease_label":
                    row.get(
                        "disease_label",
                        ""
                    ),

                "ITA":
                    row["ITA"],

                "bracket":
                    bracket,

                "selected_beta":
                    beta,

                "raw_image":
                    str(raw_path),

                "mask_overlay":
                    str(mask_overlay_path),

                "clahe_output":
                    str(clahe_path),

                "reviewer_1":
                    "",

                "reviewer_2":
                    "",

                "final_decision":
                    "",

                "review_reason":
                    "",
            })

        except Exception as exc:

            records.append({

                "image_id":
                    image_id,

                "disease_label":
                    row.get(
                        "disease_label",
                        ""
                    ),

                "ITA":
                    row["ITA"],

                "bracket":
                    bracket,

                "selected_beta":
                    beta,

                "raw_image":
                    "",

                "mask_overlay":
                    "",

                "clahe_output":
                    "",

                "reviewer_1":
                    "",

                "reviewer_2":
                    "",

                "final_decision":
                    "ERROR",

                "review_reason":
                    str(exc),
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# #12 APPLY FROZEN CLAHE TO ALL TRAINING IMAGES
# ============================================================

def export_all_clahe(
    dataframe: pd.DataFrame,
    beta_mapping: dict,
    output_dir: str | Path,
    config: CLAHECalibrationConfig,
) -> pd.DataFrame:
    """
    Apply the frozen ITA -> beta mapping to EVERY image in the
    Member 1 manifest, not only calibration-eligible images.

    Calibration uses only calibration_eligible=True images, but
    the frozen beta values are then applied to the complete
    training manifest. Therefore, if the manifest contains 1,776
    training images, this function attempts to export 1,776 CLAHE
    images.

    No images are removed here. Any processing error is recorded
    in the export manifest so the final count can be checked.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep all final CLAHE images separated by their frozen ITA bracket.
    # The bracket is determined by Member 1 ITA and is recorded in the
    # export manifest.
    bracket_dirs = {
        "Darkest": output_dir / "Darkest",
        "Medium": output_dir / "Medium",
        "Lightest": output_dir / "Lightest",
    }
    for bracket_dir in bracket_dirs.values():
        bracket_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(dataframe)

    print("\n[5/7] Applying frozen CLAHE to ALL training images...")
    print(f"Total images to export: {total}")

    for index, (_, row) in enumerate(dataframe.iterrows(), start=1):
        image_id = str(row["image_id"])
        image_path = row["image_path"]
        bracket = str(row["bracket"])

        # Use the frozen beta corresponding to the image's ITA bracket.
        if bracket not in beta_mapping:
            records.append({
                "image_id": image_id,
                "disease_label": row.get("disease_label", ""),
                "ITA": row.get("ITA", np.nan),
                "bracket": bracket,
                "selected_beta": np.nan,
                "input_image": str(image_path),
                "clahe_output": "",
                "status": "FAILED",
                "error": f"Unknown ITA bracket: {bracket}",
            })
            continue

        beta = float(beta_mapping[bracket])
        # Save the image inside the folder matching its ITA bracket.
        output_path = bracket_dirs[bracket] / f"{image_id}_clahe.png"

        try:
            image = load_rgb_image(image_path)

            enhanced = apply_clahe(
                image_rgb=image,
                beta=beta,
                tile_grid_size=(
                    config.tile_grid_width,
                    config.tile_grid_height,
                ),
            )

            ok = cv2.imwrite(
                str(output_path),
                cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
            )

            if not ok:
                raise IOError(f"Failed to write output: {output_path}")

            records.append({
                "image_id": image_id,
                "disease_label": row.get("disease_label", ""),
                "ITA": row.get("ITA", np.nan),
                "bracket": bracket,
                "selected_beta": beta,
                "input_image": str(image_path),
                "clahe_output": str(output_path),
                "status": "SUCCESS",
                "error": "",
            })

        except Exception as exc:
            records.append({
                "image_id": image_id,
                "disease_label": row.get("disease_label", ""),
                "ITA": row.get("ITA", np.nan),
                "bracket": bracket,
                "selected_beta": beta,
                "input_image": str(image_path),
                "clahe_output": "",
                "status": "FAILED",
                "error": str(exc),
            })

        if index % 100 == 0 or index == total:
            success_count = sum(
                1 for record in records
                if record["status"] == "SUCCESS"
            )
            print(
                f"  Progress: {index}/{total} | "
                f"successful: {success_count} | "
                f"failed: {index - success_count}"
            )

    export_df = pd.DataFrame(records)
    export_manifest = output_dir.parent / "clahe_all_manifest.csv"
    export_df.to_csv(export_manifest, index=False)

    success_count = int((export_df["status"] == "SUCCESS").sum())
    failed_count = int((export_df["status"] == "FAILED").sum())

    print("\nALL-IMAGE CLAHE EXPORT COMPLETE")
    print(f"  Input images : {total}")
    print(f"  Successful   : {success_count}")
    print(f"  Failed       : {failed_count}")
    print(f"  Manifest     : {export_manifest}")
    print(f"  Output folder: {output_dir}")

    if success_count > 0:
        print("\n  OUTPUTS BY ITA BRACKET:")
        for bracket_name in ["Darkest", "Medium", "Lightest"]:
            bracket_success = int(((export_df["status"] == "SUCCESS") & (export_df["bracket"] == bracket_name)).sum())
            print(f"    {bracket_name:<8}: {bracket_success}")
            print(f"      Folder: {bracket_dirs[bracket_name]}")

    if failed_count > 0:
        print("\nWARNING: Some images were not exported. Check clahe_all_manifest.csv.")

    return export_df


# ============================================================
# #12 FROZEN CONFIGURATION
# ============================================================

def create_frozen_configuration(
    beta_high: float,
    beta_mid: float,
    beta_low: float,
    config: CLAHECalibrationConfig,
    beta_global: Optional[float] = None,
) -> dict:
    """
    Create the final phase0_calibration.json.
    """

    return {

        # ----------------------------------------------------
        # VERSION
        # ----------------------------------------------------

        "phase": "phase0",
        # Only the research team may change this to FROZEN after
        # reviewing the QC sheet and beta_search_summary.csv.
        "status": "PROVISIONAL",

        "configuration_version":
            "phase0_v1",

        "calibration_date":
            datetime.now().isoformat(),

        # ----------------------------------------------------
        # #9 ITA BRACKETS
        # ----------------------------------------------------

        "ITA_brackets": {

            "Darkest":
                "ITA < 28",

            "Medium":
                "28 <= ITA <= 41",

            "Lightest":
                "ITA > 41",
        },

        # ----------------------------------------------------
        # #9 PIECEWISE MAPPING
        # ----------------------------------------------------

        "clip_limit_mapping": {

            "ITA < 28":
                "beta_high",

            "28 <= ITA <= 41":
                "beta_mid",

            "ITA > 41":
                "beta_low",
        },

        "beta_high":
            beta_high,

        "beta_mid":
            beta_mid,

        "beta_low":
            beta_low,

        # Same calibration procedure on all brackets pooled; used by the
        # fixed-beta ablation control (no ITA).
        "beta_global":
            beta_global,

        # ----------------------------------------------------
        # #10 SEARCH
        # ----------------------------------------------------

        "beta_search": {

            "start":
                config.beta_start,

            "end":
                config.beta_end,

            "step":
                config.beta_step,
            "selection_rule":
                config.selection_rule,
        },

        # ----------------------------------------------------
        # CLAHE
        # ----------------------------------------------------

        "CLAHE": {

            "tile_grid_size": [

                config.tile_grid_width,

                config.tile_grid_height
            ],

            "channel":
                "L",
        },

        # ----------------------------------------------------
        # VALIDATION PARAMETERS
        # ----------------------------------------------------

        "validation": {

            "noise_limit":
                config.noise_limit,

            "minimum_contrast_gain":
                config.minimum_contrast_gain,

            "minimum_acceptance_rate":
                config.minimum_acceptance_rate,

            "contrast_metric":
                "variance of local contrast L - GaussianBlur(L), evaluated inside valid skin mask",

            "noise_metric":
                "mean absolute L-channel change divided by 255, evaluated inside valid skin mask",
        },

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        "random_seed":
            config.random_seed,

        "calibration_dataset_version":
            config.calibration_dataset_version,

        # ----------------------------------------------------
        # MEMBER 1 REFERENCE
        # ----------------------------------------------------

        "upstream_module":
            "masking_ita.py",

        "upstream_requirements": [

            "successful skin mask",

            "ITA calculated from corrected CIELAB values",

            "ITA bracket",

            "calibration_eligible=True"
        ],
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data: dict,
    path: str | Path
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase0_calibration(
    manifest_path: str | Path,
    output_dir: str | Path = "phase0_outputs",
    config: Optional[
        CLAHECalibrationConfig
    ] = None,
):
    """
    Execute Member 2's complete Phase 0 work.
    """

    config = (
        config
        or CLAHECalibrationConfig()
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("MEMBER 2 - PHASE 0 CLAHE CALIBRATION")
    print("=" * 60)

    # ========================================================
    # READ MANIFEST
    # ========================================================

    print("\n[1/6] Reading manifest...")

    dataframe = pd.read_csv(
        manifest_path
    )

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    required = [
        "image_id",
        "image_path",
        "mask_path",
        "ITA",
        "bracket",
        "calibration_eligible",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Manifest is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # ========================================================
    # RECOMPUTE BRACKET TO VERIFY MEMBER 1
    # ========================================================

    dataframe[
        "computed_bracket"
    ] = dataframe[
        "ITA"
    ].apply(
        lambda ita: assign_ita_bracket(ita) if pd.notna(ita) else ""
    )

    # MASK_FAILED rows have no ITA and no bracket.
    dataframe["bracket"] = dataframe["bracket"].fillna("")

    dataframe[
        "bracket_check"
    ] = (
        dataframe["bracket"]
        ==
        dataframe["computed_bracket"]
    )

    mismatches = dataframe[
        ~dataframe["bracket_check"]
    ]

    if not mismatches.empty:

        print(
            "\nWARNING:"
            f" {len(mismatches)} ITA bracket "
            "mismatches detected."
        )

    # ========================================================
    # SAVE IMAGE ITA MANIFEST
    # ========================================================

    manifest_output = (
        output_dir
        / "image_ita_manifest.csv"
    )

    dataframe.to_csv(
        manifest_output,
        index=False
    )

    print(
        f"Manifest saved: {manifest_output}"
    )

    # ========================================================
    # #10 CALIBRATION
    # ========================================================

    print("\n[2/6] Running beta calibration...")

    beta_search_records = []
    beta_search_summaries = []
    best_betas = {}

    brackets = [
        "Darkest",
        "Medium",
        "Lightest"
    ]

    for bracket in brackets:

        print(
            f"\nCalibrating: {bracket}"
        )

        best_beta, image_results, summary = (
            calibrate_bracket(
                dataframe=dataframe,
                bracket=bracket,
                config=config,
            )
        )

        best_betas[
            bracket
        ] = best_beta

        beta_search_summaries.append(
            summary
        )

        beta_search_records.extend(
            image_results.to_dict(
                orient="records"
            )
        )

        print(
            f"Best beta = {best_beta:.2f}"
        )

    # ========================================================
    # SAVE BETA SEARCH RESULTS
    # ========================================================

    print(
        "\n[3/6] Saving beta search results..."
    )

    beta_results_path = (
        output_dir
        / "beta_search_results.csv"
    )

    pd.DataFrame(
        beta_search_records
    ).to_csv(
        beta_results_path,
        index=False
    )

    # Pooled (ITA-agnostic) calibration for the fixed-beta control.
    # The pool is the union of the three brackets, so the per-image
    # results already computed are reused instead of re-evaluated.
    print("\nCalibrating: ALL brackets pooled")
    global_summary = summarize_beta_search(
        pd.DataFrame(beta_search_records),
        "ALL",
        config
    )
    beta_global = select_beta(
        global_summary,
        config
    )
    global_summary["selected"] = global_summary["beta"] == beta_global
    beta_search_summaries.append(global_summary)
    print(f"Best beta = {beta_global:.2f}")

    beta_summary_path = (
        output_dir
        / "beta_search_summary.csv"
    )
    pd.concat(
        beta_search_summaries,
        ignore_index=True
    ).to_csv(
        beta_summary_path,
        index=False
    )

    # ========================================================
    # MAP BETA NAMES
    # ========================================================

    beta_high = best_betas[
        "Darkest"
    ]

    beta_mid = best_betas[
        "Medium"
    ]

    beta_low = best_betas[
        "Lightest"
    ]

    beta_mapping = {

        "Darkest":
            beta_high,

        "Medium":
            beta_mid,

        "Lightest":
            beta_low,
    }

    print("\nFinal calibrated values:")
    print(
        f"  beta_high = {beta_high}"
    )
    print(
        f"  beta_mid  = {beta_mid}"
    )
    print(
        f"  beta_low  = {beta_low}"
    )

    print(
        f"  beta_global = {beta_global}  (pooled, fixed-beta control)"
    )

    # ========================================================
    # #11 VALIDATION
    # ========================================================

    print(
        "\n[4/6] Creating validation sample..."
    )

    validation_sample = (
        create_validation_sample(
            dataframe,
            config
        )
    )

    qc_dir = (
        output_dir
        / "clahe_samples"
    )

    validation_df = (
        save_qc_samples(
            validation_sample,
            beta_mapping,
            qc_dir,
            config
        )
    )

    # ========================================================
    # SAVE QC SHEET
    # ========================================================

    validation_path = (
        output_dir
        / "mask_clahe_validation.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    print(
        f"QC sheet saved: {validation_path}"
    )

    # ========================================================
    # #12 FROZEN JSON
    # ========================================================

    print(
        "\n[5/6] Creating frozen configuration..."
    )

    frozen_config = (
        create_frozen_configuration(
            beta_high=beta_high,
            beta_mid=beta_mid,
            beta_low=beta_low,
            config=config,
            beta_global=beta_global,
        )
    )

    config_path = (
        output_dir
        / "phase0_calibration.json"
    )

    save_json(
        frozen_config,
        config_path
    )

    print(
        f"Frozen config saved: {config_path}"
    )

    # ========================================================
    # #12 APPLY FROZEN SETTINGS TO ALL TRAINING IMAGES
    # ========================================================

    clahe_all_dir = output_dir / "clahe_all"

    clahe_all_df = export_all_clahe(
        dataframe=dataframe,
        beta_mapping=beta_mapping,
        output_dir=clahe_all_dir,
        config=config,
    )

    clahe_all_manifest_path = output_dir / "clahe_all_manifest.csv"

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n[7/7] Phase 0 complete."
    )

    print("\n" + "=" * 60)
    print("MEMBER 2 OUTPUT")
    print("=" * 60)

    print(
        f"beta_high : {beta_high}"
    )

    print(
        f"beta_mid  : {beta_mid}"
    )

    print(
        f"beta_low  : {beta_low}"
    )

    print(
        "\nFiles:"
    )

    print(
        f"  {beta_results_path}"
    )

    print(
        f"  {validation_path}"
    )

    print(
        f"  {config_path}"
    )

    print(
        f"  {clahe_all_manifest_path}"
    )

    print(
        f"  {clahe_all_dir}"
    )

    print(
        f"  ALL-IMAGE CLAHE: {len(clahe_all_df)} rows"
    )

    print("=" * 60)

    return {
        "beta_high":
            beta_high,

        "beta_mid":
            beta_mid,

        "beta_low":
            beta_low,

        "beta_global":
            beta_global,

        "phase0_calibration":
            str(config_path),

        "beta_search_results":
            str(beta_results_path),

        "validation_sheet":
            str(validation_path),

        "clahe_all_manifest":
            str(clahe_all_manifest_path),

        "clahe_all_output_dir":
            str(clahe_all_dir),

        "clahe_all_count":
            int(len(clahe_all_df)),

        "clahe_all_success":
            int((clahe_all_df["status"] == "SUCCESS").sum()),

        "clahe_all_failed":
            int((clahe_all_df["status"] == "FAILED").sum()),
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Member 2 Phase 0 CLAHE "
            "Calibration and Export"
        )
    )

    parser.add_argument(
        "manifest",
        help=(
            "Path to Member 1's "
            "image_ita_manifest.csv"
        )
    )

    parser.add_argument(
        "--output",
        default="phase0_outputs",
        help=(
            "Output directory "
            "(default: phase0_outputs)"
        )
    )

    args = parser.parse_args()

    run_phase0_calibration(
        manifest_path=args.manifest,
        output_dir=args.output,
    )