from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _central_slice(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim == 2:
        return volume
    if volume.ndim != 3:
        raise ValueError("Parameter map must be 2-D or 3-D")
    return volume[:, :, volume.shape[2] // 2]


def save_neuflim_figure(
    yield_grid: np.ndarray,
    lifetime_grid: np.ndarray,
    meas_pred: np.ndarray,
    meas_true: np.ndarray,
    loss_history: list[float],
    dt: float,
    save_path: str,
    source_idx: int = 0,
    detector_idx: int = 0,
    log_tpsf: bool = False,
) -> None:
    """Save yield/lifetime maps, one TPSF fit, and the training curve."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_slice = _central_slice(yield_grid)
    tau_slice = _central_slice(lifetime_grid)

    meas_pred = np.asarray(meas_pred)
    meas_true = np.asarray(meas_true)
    if meas_pred.shape != meas_true.shape or meas_pred.ndim != 3:
        raise ValueError("Measurements must both have shape [T, Nd, Ns]")

    source_idx = int(np.clip(source_idx, 0, meas_pred.shape[2] - 1))
    detector_idx = int(np.clip(detector_idx, 0, meas_pred.shape[1] - 1))
    times = np.arange(meas_pred.shape[0]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(y_slice.T, origin="lower", aspect="equal")
    axes[0, 0].set_title("Reconstructed fluorescence yield")
    axes[0, 0].set_xlabel("x index")
    axes[0, 0].set_ylabel("y index")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(tau_slice.T, origin="lower", aspect="equal")
    axes[0, 1].set_title("Reconstructed lifetime")
    axes[0, 1].set_xlabel("x index")
    axes[0, 1].set_ylabel("y index")
    fig.colorbar(im1, ax=axes[0, 1])

    pred_curve = meas_pred[:, detector_idx, source_idx]
    true_curve = meas_true[:, detector_idx, source_idx]
    if log_tpsf:
        eps = np.finfo(np.float32).tiny
        axes[1, 0].semilogy(times, np.maximum(true_curve, eps), label="measurement")
        axes[1, 0].semilogy(times, np.maximum(pred_curve, eps), label="prediction")
    else:
        axes[1, 0].plot(times, true_curve, label="measurement")
        axes[1, 0].plot(times, pred_curve, label="prediction")
    axes[1, 0].set_title(f"Emission TPSF: source {source_idx}, detector {detector_idx}")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("intensity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(np.arange(1, len(loss_history) + 1), loss_history)
    axes[1, 1].set_title("Training loss")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("loss")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=180)
    plt.close(fig)
