"""Reproducible healthy-skin masking and ITA computation.

The module is shared by offline Phase 0 calibration and the inference API.  It
never applies CLAHE or trains a model.  OpenCV LAB values are converted to the
standard CIELAB ranges before clustering or ITA computation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ProcessingStatus(str, Enum):
    MASK_SUCCESS = "MASK_SUCCESS"
    MASK_FALLBACK = "MASK_FALLBACK"
    MASK_FAILED = "MASK_FAILED"
    IMAGE_FAILED = "IMAGE_FAILED"


@dataclass
class MaskingITAConfig:
    k_values: Sequence[int] = (2, 3, 4, 5)
    random_state: int = 42
    n_init: int = 20
    k_selection_max_side: int = 512
    max_k_selection_pixels: int = 10_000
    max_silhouette_samples: int = 5_000

    l_min: float = 20.0
    l_max: float = 90.0
    a_min: float = -5.0
    a_max: float = 35.0
    b_min: float = 0.0
    b_max: float = 50.0
    shadow_l_threshold: float = 20.0
    highlight_l_threshold: float = 95.0
    maximum_shadow_fraction: float = 0.10
    maximum_highlight_fraction: float = 0.10
    minimum_pixel_plausibility_fraction: float = 0.85

    mask_area_min_percent: float = 5.0
    mask_area_max_percent: float = 85.0
    minimum_valid_pixel_count: int = 1_000
    center_crop_width_ratio: float = 0.60
    center_crop_height_ratio: float = 0.60
    minimum_abs_b: float = 0.001
    minimum_short_side: int = 256
    minimum_laplacian_variance: float = 50.0
    black_luma_max: int = 5
    white_luma_min: int = 250
    maximum_clipped_fraction: float = 0.30
    minimum_luminance_span: float = 20.0
    use_standardized_features: bool = False

    def __post_init__(self) -> None:
        if tuple(self.k_values) != (2, 3, 4, 5):
            raise ValueError("k_values must be exactly (2, 3, 4, 5).")
        if self.n_init < 1:
            raise ValueError("n_init must be positive.")
        if not 0 < self.mask_area_min_percent < self.mask_area_max_percent <= 100:
            raise ValueError("Invalid mask area limits.")
        if self.minimum_valid_pixel_count < 1:
            raise ValueError("minimum_valid_pixel_count must be positive.")
        if not 0 < self.center_crop_width_ratio <= 1 or not 0 < self.center_crop_height_ratio <= 1:
            raise ValueError("Crop ratios must be in (0, 1].")
        if self.minimum_abs_b <= 0:
            raise ValueError("minimum_abs_b must be positive.")
        if self.minimum_short_side < 1:
            raise ValueError("minimum_short_side must be positive.")
        if self.use_standardized_features:
            raise ValueError("Standardized LAB features are not permitted by the frozen protocol.")


@dataclass
class ClusterInfo:
    label: int
    pixel_count: int
    area_percent: float
    mean_l: float
    mean_a: float
    mean_b: float
    pixel_plausibility_fraction: float
    shadow_fraction: float
    highlight_fraction: float
    touches_border: bool
    plausible: bool
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class MaskingITAResult:
    filename: str
    status: str
    mask_method: Optional[str] = None
    selected_k: Optional[int] = None
    silhouette_score: Optional[float] = None
    mask_area_pixels: Optional[int] = None
    mask_area_percent: Optional[float] = None
    mean_l: Optional[float] = None
    mean_a: Optional[float] = None
    mean_b: Optional[float] = None
    ita: Optional[float] = None
    bracket: Optional[str] = None
    calibration_eligible: bool = False
    failure_reason: Optional[str] = None
    selection_method: Optional[str] = None
    user_message: Optional[str] = None
    k_scores: dict[str, Optional[float]] = field(default_factory=dict)
    clusters: list[dict] = field(default_factory=list)
    primary_failure_reason: Optional[str] = None
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("mask", None)
        return data


class MaskingITAProcessor:
    def __init__(self, config: Optional[MaskingITAConfig] = None) -> None:
        self.config = config or MaskingITAConfig()

    @staticmethod
    def _pil_to_rgb(source: Image.Image) -> np.ndarray:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("MULTIFRAME_IMAGE")
        source.load()
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
            if alpha.getextrema()[0] < 255:
                raise ValueError("NON_OPAQUE_IMAGE")
        icc = image.info.get("icc_profile")
        if icc:
            try:
                src_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                image = ImageCms.profileToProfile(
                    image.convert("RGB"), src_profile,
                    ImageCms.createProfile("sRGB"), outputMode="RGB"
                )
            except Exception as exc:
                raise ValueError(f"INVALID_ICC_PROFILE: {exc}") from exc
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Decode a single-frame image, apply EXIF orientation, and return sRGB."""
        try:
            with Image.open(Path(path)) as source:
                return MaskingITAProcessor._pil_to_rgb(source)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"UNREADABLE_IMAGE: {exc}") from exc

    @staticmethod
    def load_image_bytes(contents: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(contents)) as source:
                return MaskingITAProcessor._pil_to_rgb(source)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"UNREADABLE_IMAGE: {exc}") from exc

    @staticmethod
    def rgb_to_opencv_lab(image_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] *= 100.0 / 255.0
        lab[:, :, 1] -= 128.0
        lab[:, :, 2] -= 128.0
        return lab

    def _deterministic_sample(self, values: np.ndarray, maximum: int, salt: int = 0) -> np.ndarray:
        if len(values) <= maximum:
            return values
        rng = np.random.default_rng(self.config.random_state + salt)
        return values[rng.choice(len(values), size=maximum, replace=False)]

    def _selection_lab(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        scale = min(1.0, self.config.k_selection_max_side / max(height, width))
        if scale < 1.0:
            image_rgb = cv2.resize(
                image_rgb,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return self.rgb_to_opencv_lab(image_rgb)

    def _select_k(self, image_rgb: np.ndarray) -> tuple[int, Optional[float], str, dict[str, Optional[float]]]:
        features = self._selection_lab(image_rgb).reshape(-1, 3)
        features = self._deterministic_sample(features, self.config.max_k_selection_pixels)
        scores: dict[str, Optional[float]] = {}
        best: Optional[tuple[float, int]] = None
        for k in self.config.k_values:
            key = str(k)
            if len(features) <= k or len(np.unique(features, axis=0)) < k:
                scores[key] = None
                continue
            try:
                model = KMeans(
                    n_clusters=k, random_state=self.config.random_state,
                    n_init=self.config.n_init, algorithm="lloyd",
                )
                labels = model.fit_predict(features)
                sample_count = min(len(features), self.config.max_silhouette_samples)
                if sample_count < len(features):
                    rng = np.random.default_rng(self.config.random_state + k)
                    indices = rng.choice(len(features), sample_count, replace=False)
                    score_features, score_labels = features[indices], labels[indices]
                else:
                    score_features, score_labels = features, labels
                if len(np.unique(score_labels)) < 2:
                    scores[key] = None
                    continue
                score = float(silhouette_score(score_features, score_labels))
                scores[key] = score
                candidate = (score, -k)
                if best is None or candidate > (best[0], -best[1]):
                    best = (score, k)
            except Exception:
                scores[key] = None
        if best is not None:
            return best[1], best[0], "SILHOUETTE", scores
        if len(features) >= 3 and len(np.unique(features, axis=0)) >= 3:
            return 3, None, "K3_FALLBACK", scores
        raise RuntimeError("KMEANS_NO_VALID_K")

    def _cluster_original(self, lab: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        all_features = lab.reshape(-1, 3)
        uniqueness_check = self._deterministic_sample(all_features, min(len(all_features), 10_000), salt=200 + k)
        if len(all_features) < k or len(np.unique(uniqueness_check, axis=0)) < k:
            raise RuntimeError("KMEANS_INSUFFICIENT_UNIQUE_PIXELS")
        model = KMeans(
            n_clusters=k, random_state=self.config.random_state,
            n_init=self.config.n_init, algorithm="lloyd",
        ).fit(all_features)
        return model.labels_, model.cluster_centers_

    def _cluster_info(self, lab: np.ndarray, labels: np.ndarray) -> list[ClusterInfo]:
        flat = lab.reshape(-1, 3)
        height, width = lab.shape[:2]
        output: list[ClusterInfo] = []
        for label in sorted(np.unique(labels)):
            mask = labels.reshape(height, width) == label
            pixels = flat[labels == label]
            count = int(len(pixels))
            area = 100.0 * count / len(flat)
            means = pixels.mean(axis=0)
            pixel_ok = (
                (pixels[:, 0] >= self.config.l_min) & (pixels[:, 0] <= self.config.l_max)
                & (pixels[:, 1] >= self.config.a_min) & (pixels[:, 1] <= self.config.a_max)
                & (pixels[:, 2] >= self.config.b_min) & (pixels[:, 2] <= self.config.b_max)
            )
            plausible_fraction = float(pixel_ok.mean())
            shadow_fraction = float((pixels[:, 0] < self.config.shadow_l_threshold).mean())
            highlight_fraction = float((pixels[:, 0] > self.config.highlight_l_threshold).mean())
            touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
            reasons: list[str] = []
            if not self.config.l_min <= means[0] <= self.config.l_max:
                reasons.append("MEAN_L_OUT_OF_RANGE")
            if not self.config.a_min <= means[1] <= self.config.a_max:
                reasons.append("MEAN_A_OUT_OF_RANGE")
            if not self.config.b_min <= means[2] <= self.config.b_max:
                reasons.append("MEAN_B_OUT_OF_RANGE")
            if not self.config.mask_area_min_percent <= area <= self.config.mask_area_max_percent:
                reasons.append("MASK_AREA_OUT_OF_RANGE")
            if count < self.config.minimum_valid_pixel_count:
                reasons.append("INSUFFICIENT_VALID_PIXELS")
            if plausible_fraction < self.config.minimum_pixel_plausibility_fraction:
                reasons.append("PIXEL_PLAUSIBILITY_TOO_LOW")
            if shadow_fraction > self.config.maximum_shadow_fraction:
                reasons.append("EXCESSIVE_SHADOW")
            if highlight_fraction > self.config.maximum_highlight_fraction:
                reasons.append("EXCESSIVE_HIGHLIGHT")
            output.append(ClusterInfo(
                label=int(label), pixel_count=count, area_percent=area,
                mean_l=float(means[0]), mean_a=float(means[1]), mean_b=float(means[2]),
                pixel_plausibility_fraction=plausible_fraction,
                shadow_fraction=shadow_fraction, highlight_fraction=highlight_fraction,
                touches_border=touches, plausible=not reasons, rejection_reasons=reasons,
            ))
        return output

    @staticmethod
    def assign_bracket(ita: float) -> str:
        if ita < 28.0:
            return "Darkest"
        if ita <= 41.0:
            return "Medium"
        return "Lightest"

    def calculate_ita(self, mean_l: float, mean_b: float) -> float:
        if abs(mean_b) < self.config.minimum_abs_b:
            raise ValueError("ITA_UNSTABLE_B_ZERO")
        return float(np.degrees(np.arctan((mean_l - 50.0) / mean_b)))

    def _image_quality_failure(self, image_rgb: np.ndarray) -> Optional[str]:
        height, width = image_rgb.shape[:2]
        if min(height, width) < self.config.minimum_short_side:
            return "RESOLUTION_TOO_LOW"
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, 1024.0 / max(height, width))
        if scale < 1.0:
            gray = cv2.resize(
                gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.config.minimum_laplacian_variance:
            return "SEVERE_BLUR"
        if float((gray <= self.config.black_luma_max).mean()) > self.config.maximum_clipped_fraction:
            return "EXCESSIVE_BLACK_CLIPPING"
        if float((gray >= self.config.white_luma_min).mean()) > self.config.maximum_clipped_fraction:
            return "EXCESSIVE_WHITE_CLIPPING"
        if float(np.percentile(gray, 95) - np.percentile(gray, 5)) < self.config.minimum_luminance_span:
            return "SEVERE_LOW_CONTRAST"
        return None

    def _attempt(self, image_rgb: np.ndarray, filename: str, method: str) -> MaskingITAResult:
        try:
            selected_k, score, selection_method, scores = self._select_k(image_rgb)
            lab = self.rgb_to_opencv_lab(image_rgb)
            labels, _ = self._cluster_original(lab, selected_k)
            clusters = self._cluster_info(lab, labels)
            audit_clusters = [asdict(cluster) for cluster in clusters]
            plausible = [cluster for cluster in clusters if cluster.plausible]
            if not plausible:
                return MaskingITAResult(
                    filename, ProcessingStatus.MASK_FAILED.value, method,
                    selected_k, score, selection_method=selection_method,
                    failure_reason="NO_PLAUSIBLE_CLUSTER", k_scores=scores,
                    clusters=audit_clusters,
                )
            chosen = max(plausible, key=lambda item: (item.pixel_count, -item.label))
            mask = labels.reshape(lab.shape[:2]) == chosen.label
            pixels = lab[mask]
            mean_l, mean_a, mean_b = (float(value) for value in pixels.mean(axis=0))
            try:
                ita = self.calculate_ita(mean_l, mean_b)
            except ValueError as exc:
                return MaskingITAResult(
                    filename=filename, status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method, selected_k=selected_k, silhouette_score=score,
                    selection_method=selection_method, mask_area_pixels=chosen.pixel_count,
                    mask_area_percent=chosen.area_percent, mean_l=mean_l,
                    mean_a=mean_a, mean_b=mean_b,
                    failure_reason=str(exc), k_scores=scores, clusters=audit_clusters,
                )
            return MaskingITAResult(
                filename=filename,
                status=(ProcessingStatus.MASK_FALLBACK.value if method == "center_weighted_crop" else ProcessingStatus.MASK_SUCCESS.value),
                mask_method=method, selected_k=selected_k, silhouette_score=score,
                selection_method=selection_method, mask_area_pixels=chosen.pixel_count,
                mask_area_percent=chosen.area_percent, mean_l=mean_l, mean_a=mean_a,
                mean_b=mean_b, ita=ita, bracket=self.assign_bracket(ita),
                calibration_eligible=True, k_scores=scores, clusters=audit_clusters, mask=mask,
            )
        except Exception as exc:
            return MaskingITAResult(
                filename, ProcessingStatus.MASK_FAILED.value, method,
                failure_reason=f"PROCESSING_ERROR: {exc}",
            )

    def process_image(self, image_rgb: np.ndarray, filename: str = "<array>") -> MaskingITAResult:
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return MaskingITAResult(
                filename, ProcessingStatus.IMAGE_FAILED.value,
                failure_reason="EXPECTED_RGB_IMAGE", user_message="Invalid image; please retake image.",
            )
        quality_failure = self._image_quality_failure(image_rgb)
        if quality_failure:
            return MaskingITAResult(
                filename, ProcessingStatus.IMAGE_FAILED.value,
                failure_reason=quality_failure,
                user_message="Image quality is insufficient; please retake image.",
            )
        primary = self._attempt(image_rgb, filename, "primary")
        if primary.calibration_eligible:
            return primary
        height, width = image_rgb.shape[:2]
        crop_w = max(1, round(width * self.config.center_crop_width_ratio))
        crop_h = max(1, round(height * self.config.center_crop_height_ratio))
        x1, y1 = (width - crop_w) // 2, (height - crop_h) // 2
        fallback = self._attempt(
            image_rgb[y1:y1 + crop_h, x1:x1 + crop_w], filename, "center_weighted_crop"
        )
        if fallback.calibration_eligible:
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y1:y1 + crop_h, x1:x1 + crop_w] = fallback.mask
            fallback.mask = full_mask
            fallback.mask_area_pixels = int(full_mask.sum())
            fallback.mask_area_percent = 100.0 * float(full_mask.mean())
            fallback.primary_failure_reason = primary.failure_reason
            return fallback
        return MaskingITAResult(
            filename=filename, status=ProcessingStatus.MASK_FAILED.value,
            mask_method="primary_then_center_weighted_crop",
            selected_k=fallback.selected_k, silhouette_score=fallback.silhouette_score,
            selection_method=fallback.selection_method,
            failure_reason=(f"PRIMARY_FAILED:{primary.failure_reason};FALLBACK_FAILED:{fallback.failure_reason}"),
            user_message="No plausible skin detected; please retake image.",
            k_scores=fallback.k_scores, clusters=fallback.clusters,
            primary_failure_reason=primary.failure_reason,
        )

    def process_file(self, path: str | Path) -> MaskingITAResult:
        try:
            image = self.load_image(path)
        except Exception as exc:
            return MaskingITAResult(
                Path(path).name, ProcessingStatus.IMAGE_FAILED.value,
                failure_reason=str(exc), user_message="Invalid image; please retake image.",
            )
        return self.process_image(image, Path(path).name)

    def process_directory(self, directory: str | Path, extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")) -> list[MaskingITAResult]:
        allowed = {extension.lower() for extension in extensions}
        paths = sorted(path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in allowed)
        return [self.process_file(path) for path in paths]


def result_to_dict(result: MaskingITAResult) -> dict:
    return result.to_dict()


def results_to_rows(results) -> list[dict]:
    return [result.to_dict() for result in results]


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Run masking and ITA on one image.")
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(MaskingITAProcessor().process_file(args.image).to_dict(), indent=2))
