"""
masking_ita.py
==============

Reusable image masking and ITA (Individual Typology Angle) processing module.

Purpose
-------
This module provides a documented, configurable pipeline for:

1. Converting RGB images to OpenCV CIELAB and scaling it to:
       L* = Lcv * 100 / 255
       a* = acv - 128
       b* = bcv - 128

2. Running K-Means clustering for k = 2, 3, 4, 5.

3. Selecting the k with the highest Silhouette Score.

4. Identifying the largest cluster that satisfies configurable,
   biologically-plausible skin constraints.

5. Validating the resulting mask using luminance and mask-area rules.

6. Falling back to a center-weighted crop when the primary mask fails.

7. Calculating ITA from the mean L* and b* values of the successful mask:
       ITA = atan((L* - 50) / b*) * 180 / pi

8. Assigning the image to:
       Darkest : ITA < 28
       Medium  : 28 <= ITA <= 41
       Lightest: ITA > 41

9. Returning structured processing results suitable for use by other
   applications, scripts, notebooks, or calibration pipelines.

The module intentionally keeps thresholds configurable because the
definition of "biologically plausible" and acceptable mask area may
depend on the dataset and study protocol.

Dependencies
------------
    numpy
    opencv-python
    scikit-learn

Example
-------
    from masking_ita import MaskingITAProcessor, MaskingITAConfig

    config = MaskingITAConfig()

    processor = MaskingITAProcessor(config)

    result = processor.process_file("image_001.jpg")

    print(result.status)
    print(result.ita)
    print(result.bracket)

Batch example
-------------
    from pathlib import Path

    results = processor.process_directory(Path("images"))

    for result in results:
        print(result.filename, result.status, result.ita, result.bracket)

Notes
-----
- Input images are converted from OpenCV's BGR representation to RGB
  before applying the requested OpenCV RGB -> CIELAB conversion.
- K-Means operates on Lab pixel features by default.
- Silhouette Score can be expensive for very large images. The module
  therefore supports a configurable maximum number of pixels used for
  scoring.
- Failed images are returned as structured results with
  calibration_eligible=False rather than raising an exception for normal
  image-processing failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Status and result types
# ---------------------------------------------------------------------------

class ProcessingStatus(str, Enum):
    """Processing state returned for each image."""

    MASK_SUCCESS = "MASK_SUCCESS"
    MASK_FALLBACK = "MASK_FALLBACK"
    MASK_FAILED = "MASK_FAILED"
    IMAGE_FAILED = "IMAGE_FAILED"


@dataclass
class MaskingITAConfig:
    """
    Configuration for the masking and ITA pipeline.

    The default Lab limits are intentionally conservative starting points.
    They should be reviewed and calibrated against the actual image dataset.

    Attributes
    ----------
    k_values:
        K values tested by K-Means.

    random_state:
        Random seed for reproducible K-Means results.

    n_init:
        Number of K-Means initializations.

    max_silhouette_samples:
        Maximum number of pixels sampled when calculating the Silhouette Score.
        This controls memory/runtime for large images.

    # Skin plausibility limits
    l_min, l_max:
        Mean L* limits for a candidate skin cluster.

    a_min, a_max:
        Mean a* limits for a candidate skin cluster.

    b_min, b_max:
        Mean b* limits for a candidate skin cluster.

    # Mask area limits
    mask_area_min_percent:
        Minimum percentage of the image that the selected mask must occupy.

    mask_area_max_percent:
        Maximum percentage of the image that the selected mask may occupy.

    # Center-weighted fallback
    center_crop_width_ratio:
        Width of fallback crop relative to the original image.

    center_crop_height_ratio:
        Height of fallback crop relative to the original image.

    # ITA
    minimum_abs_b:
        Prevents unstable ITA calculation when mean b* is too close to zero.

    # Feature scaling
    use_standardized_features:
        If True, standardize L*, a*, and b* before K-Means. If False,
        K-Means uses the raw OpenCV-scaled Lab values.

    Notes
    -----
    Changing `use_standardized_features` changes the geometry used by
    K-Means. Keep this setting consistent across a calibration dataset.
    """

    k_values: Sequence[int] = (2, 3, 4, 5)
    random_state: int = 42
    n_init: int = 10
    max_silhouette_samples: int = 10_000

    l_min: float = 20.0
    l_max: float = 90.0

    a_min: float = -5.0
    a_max: float = 35.0

    b_min: float = 0.0
    b_max: float = 50.0

    mask_area_min_percent: float = 2.0
    mask_area_max_percent: float = 70.0

    center_crop_width_ratio: float = 0.60
    center_crop_height_ratio: float = 0.60

    minimum_abs_b: float = 1e-6

    use_standardized_features: bool = False


@dataclass
class ClusterInfo:
    """Summary information for one K-Means cluster."""

    label: int
    pixel_count: int
    area_percent: float
    mean_l: float
    mean_a: float
    mean_b: float
    plausible: bool


@dataclass
class MaskingITAResult:
    """
    Structured result for one processed image.

    All fields are intentionally serializable so results can easily be
    converted to dictionaries, JSON, CSV, or database records.
    """

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

    def to_dict(self) -> dict:
        """Return the result as a standard Python dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class MaskingITAProcessor:
    """
    Image masking and ITA processor.

    Parameters
    ----------
    config:
        Optional MaskingITAConfig. If omitted, default configuration is used.

    Example
    -------
    >>> processor = MaskingITAProcessor()
    >>> result = processor.process_file("image.jpg")
    >>> print(result.ita)
    """

    def __init__(self, config: Optional[MaskingITAConfig] = None) -> None:
        self.config = config or MaskingITAConfig()

        if not self.config.k_values:
            raise ValueError("k_values must contain at least one value.")

        if any(k < 2 for k in self.config.k_values):
            raise ValueError("Every K-Means k value must be >= 2.")

        if self.config.mask_area_min_percent < 0:
            raise ValueError("mask_area_min_percent cannot be negative.")

        if self.config.mask_area_max_percent <= self.config.mask_area_min_percent:
            raise ValueError(
                "mask_area_max_percent must be greater than "
                "mask_area_min_percent."
            )

        if not 0 < self.config.center_crop_width_ratio <= 1:
            raise ValueError("center_crop_width_ratio must be in (0, 1].")

        if not 0 < self.config.center_crop_height_ratio <= 1:
            raise ValueError("center_crop_height_ratio must be in (0, 1].")

    # ------------------------------------------------------------------
    # Image loading / color conversion
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """
        Load an image using OpenCV and return it in RGB format.

        Raises
        ------
        FileNotFoundError
            If OpenCV cannot read the image.
        """
        path = Path(path)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {path}")

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_opencv_lab(image_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to the requested OpenCV-scaled CIELAB representation.

        Exact requested correction:
            L* = Lcv * 100 / 255
            a* = acv - 128
            b* = bcv - 128

        Returns
        -------
        np.ndarray
            Float32 Lab image with channels [L*, a*, b*].
        """
        lab_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        lab = lab_cv.astype(np.float32)

        lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
        lab[:, :, 1] = lab[:, :, 1] - 128.0
        lab[:, :, 2] = lab[:, :, 2] - 128.0

        return lab

    # ------------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------------

    def _prepare_features(self, lab: np.ndarray) -> np.ndarray:
        """Convert the Lab image into a 2-D feature matrix."""
        features = lab.reshape(-1, 3).astype(np.float32)

        if not self.config.use_standardized_features:
            return features

        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0

        return (features - mean) / std

    def _sample_for_silhouette(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Deterministically sample pixels for Silhouette Score calculation.
        """
        maximum = self.config.max_silhouette_samples

        if len(features) <= maximum:
            return features, labels

        rng = np.random.default_rng(self.config.random_state)
        indices = rng.choice(len(features), size=maximum, replace=False)

        return features[indices], labels[indices]

    def _run_kmeans(
        self,
        lab: np.ndarray,
    ) -> tuple[int, float, np.ndarray, list[ClusterInfo]]:
        """
        Test all configured k values and return the best K-Means solution.

        Selection criterion:
            highest Silhouette Score.

        Returns
        -------
        tuple
            (selected_k, silhouette_score, labels, cluster_info)
        """
        features = self._prepare_features(lab)

        best_k: Optional[int] = None
        best_score = -np.inf
        best_labels: Optional[np.ndarray] = None
        best_clusters: Optional[list[ClusterInfo]] = None

        total_pixels = len(features)

        for k in self.config.k_values:
            if k >= total_pixels:
                continue

            model = KMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
            )

            labels = model.fit_predict(features)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            score_features, score_labels = self._sample_for_silhouette(
                features,
                labels,
            )

            if len(np.unique(score_labels)) < 2:
                continue

            score = float(
                silhouette_score(
                    score_features,
                    score_labels,
                    metric="euclidean",
                )
            )

            clusters = self._build_cluster_info(lab, labels)

            if score > best_score:
                best_k = k
                best_score = score
                best_labels = labels
                best_clusters = clusters

        if best_k is None or best_labels is None or best_clusters is None:
            raise RuntimeError("K-Means failed for all configured k values.")

        return best_k, best_score, best_labels, best_clusters

    # ------------------------------------------------------------------
    # Cluster validation
    # ------------------------------------------------------------------

    def _build_cluster_info(
        self,
        lab: np.ndarray,
        labels: np.ndarray,
    ) -> list[ClusterInfo]:
        """Calculate statistics and plausibility for each cluster."""
        flattened_lab = lab.reshape(-1, 3)
        total_pixels = len(labels)

        clusters: list[ClusterInfo] = []

        for label in np.unique(labels):
            pixels = flattened_lab[labels == label]

            mean_l = float(np.mean(pixels[:, 0]))
            mean_a = float(np.mean(pixels[:, 1]))
            mean_b = float(np.mean(pixels[:, 2]))

            pixel_count = len(pixels)
            area_percent = pixel_count / total_pixels * 100.0

            plausible = (
                self.config.l_min <= mean_l <= self.config.l_max
                and self.config.a_min <= mean_a <= self.config.a_max
                and self.config.b_min <= mean_b <= self.config.b_max
                and self.config.mask_area_min_percent
                <= area_percent
                <= self.config.mask_area_max_percent
            )

            clusters.append(
                ClusterInfo(
                    label=int(label),
                    pixel_count=pixel_count,
                    area_percent=area_percent,
                    mean_l=mean_l,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    plausible=plausible,
                )
            )

        return clusters

    @staticmethod
    def _largest_plausible_cluster(
        clusters: Iterable[ClusterInfo],
    ) -> Optional[ClusterInfo]:
        """Return the largest cluster marked as biologically plausible."""
        plausible = [cluster for cluster in clusters if cluster.plausible]

        if not plausible:
            return None

        return max(plausible, key=lambda cluster: cluster.pixel_count)

    def _create_mask(
        self,
        labels: np.ndarray,
        selected_cluster: ClusterInfo,
        image_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Convert a selected K-Means cluster into a boolean image mask."""
        height, width = image_shape[:2]

        return (
            labels.reshape(height, width) == selected_cluster.label
        )

    # ------------------------------------------------------------------
    # Fallback crop
    # ------------------------------------------------------------------

    def center_weighted_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Return a center-weighted crop.

        The crop is centered on the image and uses configurable width and
        height ratios. This is intended as a fallback when the full-image
        K-Means mask fails validation.
        """
        height, width = image_rgb.shape[:2]

        crop_width = max(
            1,
            int(width * self.config.center_crop_width_ratio),
        )
        crop_height = max(
            1,
            int(height * self.config.center_crop_height_ratio),
        )

        x1 = max(0, (width - crop_width) // 2)
        y1 = max(0, (height - crop_height) // 2)

        x2 = min(width, x1 + crop_width)
        y2 = min(height, y1 + crop_height)

        return image_rgb[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # ITA
    # ------------------------------------------------------------------

    def calculate_ita(
        self,
        mean_l: float,
        mean_b: float,
    ) -> float:
        """
        Calculate Individual Typology Angle (ITA) in degrees.

        Formula:
            ITA = atan((L* - 50) / b*) * 180 / pi

        Raises
        ------
        ValueError
            If b* is too close to zero for a stable calculation.
        """
        if abs(mean_b) < self.config.minimum_abs_b:
            raise ValueError(
                f"Mean b* ({mean_b}) is too close to zero for ITA."
            )

        return float(
            np.degrees(
                np.arctan((mean_l - 50.0) / mean_b)
            )
        )

    @staticmethod
    def assign_bracket(ita: float) -> str:
        """
        Assign an ITA calibration bracket.

        Darkest:
            ITA < 28

        Medium:
            28 <= ITA <= 41

        Lightest:
            ITA > 41
        """
        if ita < 28.0:
            return "Darkest"

        if ita <= 41.0:
            return "Medium"

        return "Lightest"

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------

    def _process_lab_image(
        self,
        lab: np.ndarray,
        filename: str,
        method: str,
    ) -> MaskingITAResult:
        """Run K-Means, mask validation, and ITA calculation on a Lab image."""
        try:
            (
                selected_k,
                score,
                labels,
                clusters,
            ) = self._run_kmeans(lab)

            selected_cluster = self._largest_plausible_cluster(clusters)

            if selected_cluster is None:
                return MaskingITAResult(
                    filename=filename,
                    status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method,
                    selected_k=selected_k,
                    silhouette_score=score,
                    calibration_eligible=False,
                    failure_reason="NO_PLAUSIBLE_CLUSTER",
                )

            mask = self._create_mask(
                labels,
                selected_cluster,
                lab.shape,
            )

            masked_pixels = lab[mask]

            if len(masked_pixels) == 0:
                return MaskingITAResult(
                    filename=filename,
                    status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method,
                    selected_k=selected_k,
                    silhouette_score=score,
                    calibration_eligible=False,
                    failure_reason="EMPTY_MASK",
                )

            mean_l = float(np.mean(masked_pixels[:, 0]))
            mean_a = float(np.mean(masked_pixels[:, 1]))
            mean_b = float(np.mean(masked_pixels[:, 2]))

            mask_area_pixels = int(np.count_nonzero(mask))
            mask_area_percent = (
                mask_area_pixels / mask.size * 100.0
            )

            if not (
                self.config.mask_area_min_percent
                <= mask_area_percent
                <= self.config.mask_area_max_percent
            ):
                return MaskingITAResult(
                    filename=filename,
                    status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method,
                    selected_k=selected_k,
                    silhouette_score=score,
                    mask_area_pixels=mask_area_pixels,
                    mask_area_percent=mask_area_percent,
                    mean_l=mean_l,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    calibration_eligible=False,
                    failure_reason="MASK_AREA_OUT_OF_RANGE",
                )

            if not self.config.l_min <= mean_l <= self.config.l_max:
                return MaskingITAResult(
                    filename=filename,
                    status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method,
                    selected_k=selected_k,
                    silhouette_score=score,
                    mask_area_pixels=mask_area_pixels,
                    mask_area_percent=mask_area_percent,
                    mean_l=mean_l,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    calibration_eligible=False,
                    failure_reason="LUMINANCE_OUT_OF_RANGE",
                )

            try:
                ita = self.calculate_ita(mean_l, mean_b)
            except ValueError as exc:
                return MaskingITAResult(
                    filename=filename,
                    status=ProcessingStatus.MASK_FAILED.value,
                    mask_method=method,
                    selected_k=selected_k,
                    silhouette_score=score,
                    mask_area_pixels=mask_area_pixels,
                    mask_area_percent=mask_area_percent,
                    mean_l=mean_l,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    calibration_eligible=False,
                    failure_reason=f"ITA_FAILED: {exc}",
                )

            bracket = self.assign_bracket(ita)

            status = (
                ProcessingStatus.MASK_FALLBACK.value
                if method == "center_weighted_crop"
                else ProcessingStatus.MASK_SUCCESS.value
            )

            return MaskingITAResult(
                filename=filename,
                status=status,
                mask_method=method,
                selected_k=selected_k,
                silhouette_score=score,
                mask_area_pixels=mask_area_pixels,
                mask_area_percent=mask_area_percent,
                mean_l=mean_l,
                mean_a=mean_a,
                mean_b=mean_b,
                ita=ita,
                bracket=bracket,
                calibration_eligible=True,
            )

        except Exception as exc:
            return MaskingITAResult(
                filename=filename,
                status=ProcessingStatus.MASK_FAILED.value,
                mask_method=method,
                calibration_eligible=False,
                failure_reason=f"PROCESSING_ERROR: {exc}",
            )

    def process_image(
        self,
        image_rgb: np.ndarray,
        filename: str = "<array>",
    ) -> MaskingITAResult:
        """
        Process one RGB image through the complete masking/ITA pipeline.

        Primary approach:
            1. Process the complete image.
            2. Run dynamic K-Means.
            3. Select the largest plausible skin cluster.
            4. Validate mask.
            5. Calculate ITA.

        Fallback:
            If the primary approach fails, use the center-weighted crop and
            repeat the same K-Means/masking/ITA process.

        Final failure:
            Return MASK_FAILED and calibration_eligible=False.
        """
        if image_rgb is None:
            return MaskingITAResult(
                filename=filename,
                status=ProcessingStatus.IMAGE_FAILED.value,
                calibration_eligible=False,
                failure_reason="IMAGE_IS_NONE",
            )

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return MaskingITAResult(
                filename=filename,
                status=ProcessingStatus.IMAGE_FAILED.value,
                calibration_eligible=False,
                failure_reason="EXPECTED_RGB_IMAGE",
            )

        # Primary processing
        primary_lab = self.rgb_to_opencv_lab(image_rgb)
        primary_result = self._process_lab_image(
            primary_lab,
            filename,
            method="primary",
        )

        if primary_result.calibration_eligible:
            return primary_result

        # Fallback processing
        fallback_image = self.center_weighted_crop(image_rgb)
        fallback_lab = self.rgb_to_opencv_lab(fallback_image)

        fallback_result = self._process_lab_image(
            fallback_lab,
            filename,
            method="center_weighted_crop",
        )

        if fallback_result.calibration_eligible:
            return fallback_result

        # Preserve the fact that both attempts failed.
        return MaskingITAResult(
            filename=filename,
            status=ProcessingStatus.MASK_FAILED.value,
            mask_method="primary_then_center_weighted_crop",
            selected_k=fallback_result.selected_k,
            silhouette_score=fallback_result.silhouette_score,
            mask_area_pixels=fallback_result.mask_area_pixels,
            mask_area_percent=fallback_result.mask_area_percent,
            mean_l=fallback_result.mean_l,
            mean_a=fallback_result.mean_a,
            mean_b=fallback_result.mean_b,
            calibration_eligible=False,
            failure_reason=(
                "PRIMARY_FAILED; "
                f"PRIMARY_REASON={primary_result.failure_reason}; "
                f"FALLBACK_REASON={fallback_result.failure_reason}"
            ),
        )

    def process_file(self, path: str | Path) -> MaskingITAResult:
        """
        Process one image file.

        Normal image-processing failures are returned as IMAGE_FAILED or
        MASK_FAILED rather than being raised.
        """
        path = Path(path)

        try:
            image_rgb = self.load_image(path)
        except Exception as exc:
            return MaskingITAResult(
                filename=path.name,
                status=ProcessingStatus.IMAGE_FAILED.value,
                calibration_eligible=False,
                failure_reason=str(exc),
            )

        return self.process_image(
            image_rgb,
            filename=path.name,
        )

    def process_directory(
        self,
        directory: str | Path,
        extensions: Sequence[str] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ),
    ) -> list[MaskingITAResult]:
        """
        Process all supported image files in a directory.

        Files are processed in sorted filename order.

        Parameters
        ----------
        directory:
            Directory containing images.

        extensions:
            File extensions to process.

        Returns
        -------
        list[MaskingITAResult]
            One result for each image.
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Directory does not exist: {directory}"
            )

        allowed = {ext.lower() for ext in extensions}

        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        )

        return [self.process_file(path) for path in files]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def result_to_dict(result: MaskingITAResult) -> dict:
    """Convert a result object into a dictionary."""
    return result.to_dict()


def results_to_rows(
    results: Iterable[MaskingITAResult],
) -> list[dict]:
    """Convert multiple results into CSV/DataFrame-friendly rows."""
    return [result.to_dict() for result in results]


if __name__ == "__main__":
    # Simple command-line demonstration.
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run masking and ITA processing on one image."
    )
    parser.add_argument(
        "image",
        help="Path to the image to process.",
    )

    args = parser.parse_args()

    processor = MaskingITAProcessor()
    result = processor.process_file(args.image)

    print(json.dumps(result.to_dict(), indent=2))
