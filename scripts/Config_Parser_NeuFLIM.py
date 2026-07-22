import configargparse


def config_parser_neuflim():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, default="./configs/NeuFLIM_example.txt")

    # Experiment and paths
    parser.add_argument("--exp_type", type=str, default="simu")
    parser.add_argument("--exp_no", type=str, default="0")
    parser.add_argument("--data_dir_rel", type=str, default="./data/")
    parser.add_argument("--save_dir_rel", type=str, default="./logs/")
    parser.add_argument("--tar_shape", type=str, default="2bar")
    parser.add_argument("--ray_type", type=str, default="transmissive")
    parser.add_argument("--data_name", type=str, default="neuflim")
    parser.add_argument("--noise_level", type=float, default=0.0)

    # Domain and encoding
    parser.add_argument("--phantom_x", type=float, default=64)
    parser.add_argument("--phantom_y", type=float, default=64)
    parser.add_argument("--phantom_z", type=float, default=1)
    parser.add_argument("--enc_type", type=str, default="PE")
    parser.add_argument("--enc_level", type=int, default=6)
    parser.add_argument("--enc_level_z", type=int, default=2)

    # Time-domain forward model
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--time_window", type=int, default=1)
    parser.add_argument("--measurement_scale", type=float, default=1.0)

    # MAT file names and variable keys
    parser.add_argument("--node_file", type=str, default="node.mat")
    parser.add_argument("--node_key", type=str, default="node")
    parser.add_argument("--mvec_file", type=str, default="mvec.mat")
    parser.add_argument("--mvec_key", type=str, default="mvec")
    parser.add_argument("--a_em_file", type=str, default="A_em.mat")
    parser.add_argument("--a_em_key", type=str, default="A_em")
    parser.add_argument("--b_em_file", type=str, default="B_em.mat")
    parser.add_argument("--b_em_key", type=str, default="B_em")
    parser.add_argument("--tphi_ex_file", type=str, default="tphi_ex.mat")
    parser.add_argument("--tphi_ex_key", type=str, default="tphi_ex")
    parser.add_argument("--measurement_file", type=str, default="meas_em_td.mat")
    parser.add_argument("--measurement_key", type=str, default="meas_em")

    # Dual-output INR
    parser.add_argument("--layer_dim", type=int, default=256)
    parser.add_argument("--activation_fn", type=str, default="relu")
    parser.add_argument("--yield_min", type=float, default=0.0)
    parser.add_argument("--yield_max", type=float, default=1.2)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=3.0)
    parser.add_argument("--tau_background", type=float, default=0.05)

    # Optimization
    parser.add_argument("--n_iter", type=int, default=16000)
    parser.add_argument("--loss_type", choices=["MSE", "MAE"], default="MSE")
    parser.add_argument("--loss_scale", type=float, default=1.0)
    parser.add_argument("--yield_reg_scale", type=float, default=0.0)
    parser.add_argument("--tau_reg_scale", type=float, default=0.0)
    parser.add_argument("--lr_start", type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate", type=float, default=0.98)
    parser.add_argument("--lr_decay_step", type=int, default=160)

    # Saving and plotting
    parser.add_argument("--iter_save", type=int, default=2000)
    parser.add_argument("--plot_source", type=int, default=0)
    parser.add_argument("--plot_detector_stride", type=int, default=8)
    return parser
