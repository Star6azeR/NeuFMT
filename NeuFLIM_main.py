import os

import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from models.Encoder import positional_encoding_altz, positional_encoding_default
from models.NeFLIM import NeFLIM
from scripts.Config_Parser import config_parser_neuflim
from scripts.NeuFLIM_Forward import TimeDomainFLIMForward
from scripts.NeuFLIM_Plot import save_neuflim_figure
from scripts.Path_Setter import set_io_path_v4


def _load_mat_array(data_dir: str, filename: str, key: str) -> np.ndarray:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required NeuFLIM input does not exist: {path}")
    data = hdf5storage.loadmat(path)
    if key not in data:
        raise KeyError(f"Key '{key}' is missing from {path}; available keys: {list(data)}")
    value = data[key]
    if sparse.issparse(value):
        value = value.toarray()
    value = np.asarray(value)
    if np.iscomplexobj(value):
        value = value.real
    return value


def _move_time_axis(array: np.ndarray, time_axis: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D time-domain array, got shape {array.shape}")
    return np.moveaxis(array, time_axis, 0)


def _ensure_tnhs(array: np.ndarray, nh: int, time_axis: int) -> np.ndarray:
    array = _move_time_axis(array, time_axis)
    if array.shape[1] == nh:
        return array
    if array.shape[2] == nh:
        return array.transpose(0, 2, 1)
    raise ValueError(f"Cannot identify node axis in excitation shape {array.shape}; Nh={nh}")


def _ensure_tdns(array: np.ndarray, nd: int, time_axis: int) -> np.ndarray:
    array = _move_time_axis(array, time_axis)
    if array.shape[1] == nd:
        return array
    if array.shape[2] == nd:
        return array.transpose(0, 2, 1)
    raise ValueError(f"Cannot identify detector axis in measurement shape {array.shape}; Nd={nd}")


def _measurement_for_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalization: str,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalization == "none":
        return pred, target
    if normalization == "global_max":
        scale = target.abs().amax().clamp_min(epsilon)
    elif normalization == "global_l2":
        scale = torch.linalg.vector_norm(target).clamp_min(epsilon)
    elif normalization == "per_tpsf_max":
        scale = target.abs().amax(dim=0, keepdim=True).clamp_min(epsilon)
    else:
        raise NotImplementedError(f"Unknown measurement normalization: {normalization}")
    return pred / scale, target / scale


def _make_query_grid(args, device: torch.device) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if args.res_res == 1.0:
        nx, ny, nz = int(args.phantom_x), int(args.phantom_y), int(args.phantom_z)
        x_cor = torch.linspace(0.5, args.phantom_x - 0.5, nx) / args.phantom_x
        y_cor = torch.linspace(0.5, args.phantom_y - 0.5, ny) / args.phantom_y
        z_cor = torch.linspace(0.5, args.phantom_z - 0.5, nz) / args.phantom_z
    elif args.res_res == 1.1:
        nx, ny, nz = int(args.phantom_x) + 1, int(args.phantom_y) + 1, int(args.phantom_z) + 1
        x_cor = torch.linspace(0, args.phantom_x, nx) / args.phantom_x
        y_cor = torch.linspace(0, args.phantom_y, ny) / args.phantom_y
        z_cor = torch.linspace(0, args.phantom_z, nz) / args.phantom_z
    elif args.res_res == 2.0:
        nx, ny, nz = int(2 * args.phantom_x), int(2 * args.phantom_y), int(2 * args.phantom_z)
        x_cor = torch.linspace(0.25, args.phantom_x - 0.25, nx) / args.phantom_x
        y_cor = torch.linspace(0.25, args.phantom_y - 0.25, ny) / args.phantom_y
        z_cor = torch.linspace(0.25, args.phantom_z - 0.25, nz) / args.phantom_z
    else:
        raise NotImplementedError(f"Unsupported res_res={args.res_res}")

    return torch.cartesian_prod(x_cor, y_cor, z_cor).to(device), (nx, ny, nz)


def _encode(points: torch.Tensor, args) -> tuple[torch.Tensor, int]:
    if args.enc_type == "PE":
        return positional_encoding_default(points, args.enc_level), 6 * args.enc_level
    if args.enc_type == "PE_altz":
        encoded = positional_encoding_altz(points, args.enc_level, args.enc_level_z)
        return encoded, 4 * args.enc_level + 2 * args.enc_level_z
    raise NotImplementedError(f"Unsupported encoding: {args.enc_type}")


def train() -> None:
    parser = config_parser_neuflim()
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using {device} device")

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

    node_np = _load_mat_array(data_dir, args.node_file, args.node_key)
    node = torch.as_tensor(node_np, dtype=torch.float32, device=device)
    node = node * torch.tensor(
        [1.0 / args.phantom_x, 1.0 / args.phantom_y, 1.0 / args.phantom_z],
        dtype=node.dtype,
        device=device,
    )
    nh = node.shape[0]

    mvec_np = _load_mat_array(data_dir, args.mvec_file, args.mvec_key)
    if mvec_np.shape[0] != nh and mvec_np.shape[1] == nh:
        mvec_np = mvec_np.T
    if mvec_np.shape[0] != nh:
        raise ValueError(f"mvec must have shape [Nh, Nd], got {mvec_np.shape}")
    nd = mvec_np.shape[1]

    tphi_ex_np = _load_mat_array(data_dir, args.tphi_ex_file, args.tphi_ex_key)
    tphi_ex_np = _ensure_tnhs(tphi_ex_np, nh, args.tphi_time_axis)

    meas_em_np = _load_mat_array(data_dir, args.meas_em_file, args.meas_em_key)
    meas_em_np = _ensure_tdns(meas_em_np, nd, args.meas_time_axis)

    if args.nstep > 0:
        tphi_ex_np = tphi_ex_np[: args.nstep]
        meas_em_np = meas_em_np[: args.nstep]
    if tphi_ex_np.shape[0] != meas_em_np.shape[0]:
        raise ValueError("Excitation and measurement must have the same number of time bins")
    if tphi_ex_np.shape[2] != meas_em_np.shape[2]:
        raise ValueError("Excitation and measurement must have the same number of sources")

    A_em_np = _load_mat_array(data_dir, args.a_em_file, args.a_em_key)
    B_em_np = _load_mat_array(data_dir, args.b_em_file, args.b_em_key)
    if A_em_np.shape != (nh, nh) or B_em_np.shape != (nh, nh):
        raise ValueError(
            f"A_em and B_em must both be [Nh, Nh]; got {A_em_np.shape}, {B_em_np.shape}"
        )

    tphi_ex = torch.as_tensor(tphi_ex_np, dtype=torch.float32, device=device)
    meas_em = torch.as_tensor(meas_em_np, dtype=torch.float32, device=device) * args.em_scale_gap
    A_em = torch.as_tensor(A_em_np, dtype=torch.float32, device=device)
    B_em = torch.as_tensor(B_em_np, dtype=torch.float32, device=device)
    P = torch.as_tensor(mvec_np.T, dtype=torch.float32, device=device)

    measurement_mask = None
    if args.mask_file:
        mask_np = _load_mat_array(data_dir, args.mask_file, args.mask_key).astype(bool)
        mask_np = _ensure_tdns(mask_np, nd, args.mask_time_axis)
        if args.nstep > 0:
            mask_np = mask_np[: args.nstep]
        if mask_np.shape != meas_em_np.shape:
            raise ValueError(f"Mask shape {mask_np.shape} does not match measurement {meas_em_np.shape}")
        measurement_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    grid, grid_shape = _make_query_grid(args, device)
    node_enc, input_dim = _encode(node, args)
    grid_enc, _ = _encode(grid, args)

    net = NeFLIM(
        input_dim=input_dim,
        layer_dim=args.layer_dim,
        activation_fn=args.activation_fn,
        yield_min=args.yield_min,
        yield_max=args.yield_max,
        lifetime_min=args.lifetime_min,
        lifetime_max=args.lifetime_max,
    ).to(device)

    forward_model = TimeDomainFLIMForward(
        excitation=tphi_ex,
        A_em=A_em,
        B_em=B_em,
        P=P,
        dt=args.dt,
        lifetime_scheme=args.lifetime_scheme,
        tau_epsilon=args.tau_epsilon,
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr_start, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )

    if args.loss_type in {"MSE", "L2"}:
        error_fn = nn.MSELoss(reduction="mean")
    elif args.loss_type in {"MAE", "L1"}:
        error_fn = nn.L1Loss(reduction="mean")
    else:
        raise NotImplementedError(f"Unsupported loss: {args.loss_type}")

    writer = SummaryWriter(save_dir)
    loss_history: list[float] = []
    tqd = trange(args.n_iter)

    for iteration in tqd:
        optimizer.zero_grad(set_to_none=True)

        yield_pred, lifetime_pred = net(node_enc)
        yield_pred = yield_pred[:, 0]
        lifetime_pred = lifetime_pred[:, 0]

        meas_em_pred = forward_model(yield_pred, lifetime_pred)
        pred_for_loss, target_for_loss = _measurement_for_loss(
            meas_em_pred,
            meas_em,
            args.measurement_normalization,
            args.normalization_epsilon,
        )
        if measurement_mask is not None:
            pred_for_loss = pred_for_loss[measurement_mask]
            target_for_loss = target_for_loss[measurement_mask]

        data_loss = args.loss_scale * error_fn(pred_for_loss, target_for_loss)
        yield_reg = args.yield_regu_scale * torch.mean(torch.abs(yield_pred))
        lifetime_reg = args.lifetime_regu_scale * torch.mean(
            (lifetime_pred - args.lifetime_prior) ** 2
        )
        loss = data_loss + yield_reg + lifetime_reg

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        tqd.set_postfix(loss=loss_value, data=float(data_loss.detach().cpu()))
        writer.add_scalar("loss/total", loss_value, iteration + 1)
        writer.add_scalar("loss/data", float(data_loss.detach().cpu()), iteration + 1)
        writer.add_scalar("loss/yield_regularization", float(yield_reg.detach().cpu()), iteration + 1)
        writer.add_scalar("loss/lifetime_regularization", float(lifetime_reg.detach().cpu()), iteration + 1)
        writer.add_scalar("parameter/yield_mean", float(yield_pred.mean().detach().cpu()), iteration + 1)
        writer.add_scalar("parameter/lifetime_mean", float(lifetime_pred.mean().detach().cpu()), iteration + 1)

        if (iteration + 1) % args.lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.lr_decay_rate

        should_save = (iteration + 1) % args.iter_save == 0 or iteration + 1 == args.n_iter
        if should_save:
            step = iteration + 1
            checkpoint_path = os.path.join(save_mod_dir, f"{step}.pyt")
            mat_path = os.path.join(save_mat_dir, f"{step}.mat")
            image_path = os.path.join(save_img_dir, f"{step}.png")

            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_value,
                    "config": vars(args),
                },
                checkpoint_path,
            )

            net.eval()
            with torch.no_grad():
                yield_grid, lifetime_grid = net(grid_enc)
                yield_grid_np = yield_grid[:, 0].cpu().numpy().reshape(grid_shape)
                lifetime_grid_np = lifetime_grid[:, 0].cpu().numpy().reshape(grid_shape)
                meas_pred_np = meas_em_pred.detach().cpu().numpy()

            output_mat = {
                "yield_pred_mesh": yield_pred.detach().cpu().numpy(),
                "lifetime_pred_mesh": lifetime_pred.detach().cpu().numpy(),
                "yield_pred_grid": yield_grid_np,
                "lifetime_pred_grid": lifetime_grid_np,
                "loss_history": np.asarray(loss_history),
            }
            if args.save_measurement:
                output_mat["meas_em_pred"] = meas_pred_np
            hdf5storage.savemat(mat_path, output_mat, format="7.3")

            save_neuflim_figure(
                yield_grid_np,
                lifetime_grid_np,
                meas_pred_np,
                meas_em.detach().cpu().numpy(),
                loss_history,
                args.dt,
                image_path,
                source_idx=args.plot_source_idx,
                detector_idx=args.plot_detector_idx,
                log_tpsf=bool(args.plot_log_tpsf),
            )
            net.train()

    writer.close()
    print("NeuFLIM training done.")


if __name__ == "__main__":
    train()
