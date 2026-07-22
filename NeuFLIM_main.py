import os
from typing import Any

import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from models.Encoder import positional_encoding_altz, positional_encoding_default
from models.NeFLIM import NeFLIM
from scripts.Config_Parser_NeuFLIM import config_parser_neuflim
from scripts.Path_Setter import set_io_path_v4
from scripts.Result_Plotter import plot_neuflim_results


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_mat(path: str, key: str) -> Any:
    data = hdf5storage.loadmat(path)
    if key not in data:
        raise KeyError(f"{key!r} not found in {path}. Available keys: {list(data)}")
    return data[key]


def _dense_real(array: Any) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array).real


def _prepare_nodes(node: np.ndarray, phantom_size: tuple[float, float, float]) -> np.ndarray:
    if node.ndim != 2 or node.shape[1] not in (2, 3):
        raise ValueError(f"node must have shape [N,2] or [N,3], got {node.shape}")
    if node.shape[1] == 2:
        node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=node.dtype)), axis=1)
    return node / np.asarray(phantom_size, dtype=node.dtype)[None, :]


def _as_time_node_source(array: np.ndarray, n_nodes: int) -> np.ndarray:
    """Accept [T,N,S], [N,S,T], or [N,T,S], return [T,N,S]."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D excitation field, got {array.shape}")
    if array.shape[1] == n_nodes:
        return array
    if array.shape[0] == n_nodes:
        if array.shape[1] <= array.shape[2]:
            return np.transpose(array, (2, 0, 1))
        return np.transpose(array, (1, 0, 2))
    if array.shape[2] == n_nodes:
        return np.transpose(array, (0, 2, 1))
    raise ValueError(f"Cannot identify node axis in excitation field {array.shape}")


def _as_time_detector_source(array: np.ndarray, n_detectors: int) -> np.ndarray:
    """Accept [T,D,S] or [D,S,T], return [T,D,S]."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D measurement array, got {array.shape}")
    if array.shape[1] == n_detectors:
        return array
    if array.shape[0] == n_detectors:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"Cannot identify detector axis in measurement {array.shape}")


def _window_measurement(measurement: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 1:
        return measurement
    n_windows = measurement.shape[0] // window_size
    if n_windows == 0:
        raise ValueError("time_window is larger than the number of time steps")
    measurement = measurement[: n_windows * window_size]
    return measurement.reshape(
        n_windows, window_size, measurement.shape[1], measurement.shape[2]
    ).sum(dim=1)


def time_domain_forward(
    yield_pred: torch.Tensor,
    tau_pred: torch.Tensor,
    tphi_ex: torch.Tensor,
    A_em: torch.Tensor,
    B_em: torch.Tensor,
    detector_projection: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Differentiable lifetime-FMT model returning [time, detector, source]."""
    decay = torch.exp(-dt / tau_pred)
    gain = 1.0 - decay

    n_nodes, n_sources = tphi_ex.shape[1:]
    lifetime_state = torch.zeros(
        (n_nodes, n_sources), dtype=tphi_ex.dtype, device=tphi_ex.device
    )
    emission_field = torch.zeros_like(lifetime_state)
    measurement = []

    for excitation_field in tphi_ex:
        lifetime_state = decay * lifetime_state + gain * excitation_field
        emission_source = yield_pred * lifetime_state
        emission_field = A_em @ emission_field + B_em @ emission_source
        measurement.append(detector_projection @ emission_field)

    return torch.stack(measurement, dim=0)


def _encode(points: torch.Tensor, args):
    if args.enc_type == "PE":
        return positional_encoding_default(points, args.enc_level), 6 * args.enc_level
    if args.enc_type == "PE_altz":
        return (
            positional_encoding_altz(points, args.enc_level, args.enc_level_z),
            4 * args.enc_level + 2 * args.enc_level_z,
        )
    raise NotImplementedError(args.enc_type)


def train() -> None:
    args = config_parser_neuflim().parse_args()
    data_dir, save_dir, save_mod_dir, save_img_dir, save_mat_dir = set_io_path_v4(
        args.exp_type,
        args.exp_no,
        args.data_dir_rel,
        args.save_dir_rel,
        args.tar_shape,
        args.ray_type,
        args.data_name,
        args.noise_level,
    )
    print(f"Using {DEVICE}")

    phantom_size = (args.phantom_x, args.phantom_y, args.phantom_z)
    node_np = _dense_real(_load_mat(os.path.join(data_dir, args.node_file), args.node_key))
    node_np = _prepare_nodes(node_np, phantom_size)
    node = torch.as_tensor(node_np, dtype=torch.float32, device=DEVICE)
    n_nodes = node.shape[0]

    mvec_np = _dense_real(_load_mat(os.path.join(data_dir, args.mvec_file), args.mvec_key))
    detector_projection = torch.as_tensor(mvec_np.T, dtype=torch.float32, device=DEVICE)
    n_detectors = detector_projection.shape[0]

    A_em = torch.as_tensor(
        _dense_real(_load_mat(os.path.join(data_dir, args.a_em_file), args.a_em_key)),
        dtype=torch.float32,
        device=DEVICE,
    )
    B_em = torch.as_tensor(
        _dense_real(_load_mat(os.path.join(data_dir, args.b_em_file), args.b_em_key)),
        dtype=torch.float32,
        device=DEVICE,
    )

    tphi_ex_np = _dense_real(
        _load_mat(os.path.join(data_dir, args.tphi_ex_file), args.tphi_ex_key)
    )
    tphi_ex_np = _as_time_node_source(tphi_ex_np, n_nodes)
    tphi_ex = torch.as_tensor(tphi_ex_np, dtype=torch.float32, device=DEVICE)

    measurement_np = _dense_real(
        _load_mat(os.path.join(data_dir, args.measurement_file), args.measurement_key)
    )
    measurement_np = _as_time_detector_source(measurement_np, n_detectors)
    measurement_full = torch.as_tensor(measurement_np, dtype=torch.float32, device=DEVICE)
    measurement = _window_measurement(measurement_full, args.time_window)

    node_enc, input_dim = _encode(node, args)

    nx, ny, nz = (int(args.phantom_x), int(args.phantom_y), int(args.phantom_z))
    grid = torch.cartesian_prod(
        torch.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx),
        torch.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny),
        torch.linspace(0.5 / nz, 1.0 - 0.5 / nz, nz),
    ).to(DEVICE)
    grid_enc, _ = _encode(grid, args)

    net = NeFLIM(
        input_dim=input_dim,
        layer_dim=args.layer_dim,
        activation_fn=args.activation_fn,
        yield_min=args.yield_min,
        yield_max=args.yield_max,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_start)
    error_fn = nn.MSELoss() if args.loss_type == "MSE" else nn.L1Loss()
    writer = SummaryWriter(save_dir)
    loss_history = []

    progress = trange(args.n_iter)
    for iteration in progress:
        yield_pred, tau_pred = net(node_enc)
        measurement_pred_full = time_domain_forward(
            yield_pred,
            tau_pred,
            tphi_ex,
            A_em,
            B_em,
            detector_projection,
            args.dt,
        )
        measurement_pred = _window_measurement(measurement_pred_full, args.time_window)

        data_loss = error_fn(
            args.loss_scale * args.measurement_scale * measurement_pred,
            args.loss_scale * args.measurement_scale * measurement,
        )
        yield_reg = args.yield_reg_scale * yield_pred.abs().mean()
        tau_reg = args.tau_reg_scale * (
            yield_pred.detach() * (tau_pred - args.tau_background).abs()
        ).mean()
        loss = data_loss + yield_reg + tau_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach())
        loss_history.append(loss_value)
        progress.set_postfix(loss=loss_value)
        writer.add_scalar("loss/total", loss_value, iteration + 1)
        writer.add_scalar("loss/data", float(data_loss.detach()), iteration + 1)
        writer.add_scalar("parameter/yield_mean", float(yield_pred.mean().detach()), iteration + 1)
        writer.add_scalar("parameter/tau_mean", float(tau_pred.mean().detach()), iteration + 1)

        if (iteration + 1) % args.lr_decay_step == 0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay_rate

        if (iteration + 1) % args.iter_save == 0 or iteration + 1 == args.n_iter:
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_value,
                },
                os.path.join(save_mod_dir, f"{iteration + 1}.pyt"),
            )

            with torch.no_grad():
                yield_grid, tau_grid = net(grid_enc)
                yield_grid = yield_grid.cpu().numpy().reshape(nx, ny, nz)
                tau_grid = tau_grid.cpu().numpy().reshape(nx, ny, nz)
                pred_np = measurement_pred_full.cpu().numpy()

            hdf5storage.savemat(
                os.path.join(save_mat_dir, f"{iteration + 1}.mat"),
                {
                    "yield_pred": yield_grid.astype(np.float64),
                    "tau_pred": tau_grid.astype(np.float64),
                    "measurement_pred": pred_np.astype(np.float64),
                },
                format="7.3",
            )
            plot_neuflim_results(
                yield_grid,
                tau_grid,
                pred_np,
                measurement_np,
                loss_history,
                args.dt,
                os.path.join(save_img_dir, f"{iteration + 1}.png"),
                args.plot_source,
                args.plot_detector_stride,
            )

    writer.close()
    print("NeuFLIM training done.")


if __name__ == "__main__":
    train()
