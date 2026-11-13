"""Robust boundary metrics for binary segmentation masks."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


BOUNDARY_STRUCTURE = np.ones((3, 3), dtype=bool)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    return mask.astype(bool, copy=False)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel, 8-connected inner boundary of a binary mask."""
    mask = _as_binary_mask(mask)
    eroded = binary_erosion(
        mask,
        structure=BOUNDARY_STRUCTURE,
        border_value=0,
    )
    return np.logical_xor(mask, eroded)


def robust_boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: float = 2.0,
) -> dict[str, float]:
    """Compute HD95, ASD, and tolerance-based Boundary F1.

    Distances are measured in pixels on the evaluation grid. HD95 is the
    95th percentile and ASD is the mean of the concatenated bidirectional
    surface distances. Boundary F1 matches boundary pixels within
    ``tolerance_px``. If both masks are empty, the result is a perfect match;
    if exactly one mask is empty, distances equal the image diagonal and
    Boundary F1 is zero.
    """
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")

    prediction = _as_binary_mask(prediction)
    target = _as_binary_mask(target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs "
            f"{target.shape}"
        )

    prediction_nonempty = bool(np.any(prediction))
    target_nonempty = bool(np.any(target))
    if not prediction_nonempty and not target_nonempty:
        return {"HD95": 0.0, "ASD": 0.0, "BoundaryF1": 1.0}

    if prediction_nonempty != target_nonempty:
        height, width = prediction.shape
        diagonal = math.hypot(max(height - 1, 0), max(width - 1, 0))
        return {
            "HD95": float(diagonal),
            "ASD": float(diagonal),
            "BoundaryF1": 0.0,
        }

    prediction_boundary = extract_boundary(prediction)
    target_boundary = extract_boundary(target)

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)
    prediction_to_target = distance_to_target[prediction_boundary]
    target_to_prediction = distance_to_prediction[target_boundary]
    bidirectional = np.concatenate(
        (prediction_to_target, target_to_prediction)
    ).astype(np.float64, copy=False)

    hd95 = float(np.percentile(bidirectional, 95))
    asd = float(np.mean(bidirectional))

    boundary_precision = float(
        np.mean(prediction_to_target <= tolerance_px)
    )
    boundary_recall = float(np.mean(target_to_prediction <= tolerance_px))
    denominator = boundary_precision + boundary_recall
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall / denominator
        if denominator > 0
        else 0.0
    )

    return {
        "HD95": hd95,
        "ASD": asd,
        "BoundaryF1": float(boundary_f1),
    }
