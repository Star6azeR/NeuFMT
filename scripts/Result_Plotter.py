from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _display_slice(volume: np.ndarray) -> np.ndarray:
    volume = np.squeeze(volume)
    if volume.ndim == 2:
        return volume
    if volume.ndim == 3:
        return volume[:, :, volume.shape[2] // 2]
    raise ValueError(f"Expected a 2-D or 3-D reconstruction, got shape {volume.shape}")


def plot_neuflim_results(
    yield_grid: np.ndarray,
    tau_grid: np.ndarray,
    measurement_pred: np.ndarray,
    measurement_true: np.ndarray,
    loss_history: list[float],
    dt: float,
    save_path: str,
    source_index: int = 0,
    detector_stride: int = 8,
) -> None:
    """Save yield, lifetime, TPSF comparison, and loss in one figure."""

    yield_image = _display_slice(yield_grid)
    tau_image = _display_slice(tau_grid)

    pred = np.asarray(measurement_pred)
    true = np.asarray(measurement_true)
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError("Measurements must have shape [time, detector, source]")

    source_index = int(np.clip(source_index, 0, pred.shape[2] - 1))
    detector_ids = np.arange(0, pred.shape[1], max(1, detector_stride))
    time = np.arange(pred.shape[0]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im0 = axes[0, 0].imshow(yield_image, origin="lower")
    axes[0, 0].set_title("Reconstructed fluorescence yield")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(tau_image, origin="lower")
    axes[0, 1].set_title("Reconstructed lifetime")
    fig.colorbar(im1, ax=axes[0, 1])

    for detector in detector_ids:
        axes[1, 0].plot(time, true[:, detector, source_index], linestyle="--", alpha=0.7)
        axes[1, 0].plot(time, pred[:, detector, source_index], alpha=0.8)
    axes[1, 0].set_title(f"Emission TPSFs, source {source_index}")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Intensity")

    axes[1, 1].plot(np.arange(1, len(loss_history) + 1), loss_history)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Training loss")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
