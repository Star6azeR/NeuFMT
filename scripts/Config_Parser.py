import configargparse


def config_parser_v4():

    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, default='./configs/example.txt')
    print('Loaded config file successfully.')

    # Experiment settings
    parser.add_argument("--exp_type",       type=str)
    parser.add_argument("--exp_no",         type=str)
    parser.add_argument("--auto",           type=int,   default=0)
    parser.add_argument("--data_dir_rel",   type=str,   default='./data/')
    parser.add_argument("--save_dir_rel",   type=str,   default='./logs/')

    # Phantom and optical settings
    parser.add_argument("--phantom_x",      type=float, default=50)
    parser.add_argument("--phantom_y",      type=float, default=50)
    parser.add_argument("--phantom_z",      type=float, default=16)
    parser.add_argument("--te_maxvol",      type=float, default=0.6)
    parser.add_argument("--node_size",      type=float, default=0.1)
    parser.add_argument("--tar_shape",      type=str,   default='2bar')
    parser.add_argument("--ray_type",       type=str,   default='reflective')
    parser.add_argument("--data_name",      type=str)
    parser.add_argument("--mua",            type=float, default=0.1)
    parser.add_argument("--mus",            type=float, default=1.0)
    parser.add_argument("--ref",            type=float, default=1.4)
    parser.add_argument("--q_dim",          type=int,   default=11)
    parser.add_argument("--q_gap",          type=int,   default=4)
    parser.add_argument("--q_offset",       type=int,   default=5)
    parser.add_argument("--m_dim",          type=int,   default=49)
    parser.add_argument("--m_gap",          type=int,   default=1)
    parser.add_argument("--m_offset",       type=int,   default=1)
    parser.add_argument("--noise_level",    type=float, default=0)
    parser.add_argument("--em_scale_gap",   type=float, default=1)

    # NN build and train settings
    parser.add_argument("--enc_type",       type=str,   default='PE')
    parser.add_argument("--enc_level",      type=int,   default=4)
    parser.add_argument("--enc_level_z",    type=int,   default=4)
    parser.add_argument("--layer_dim",      type=int,   default=256)
    parser.add_argument("--activation_fn",  type=str,   default='relu')
    parser.add_argument("--n_iter",         type=int,   default=16000)
    parser.add_argument("--loss_type",      type=str,   default='MSE')
    parser.add_argument("--loss_scale",     type=float, default=1e6)
    parser.add_argument("--n_regu_term",    type=int,   default=0)
    parser.add_argument("--regu_type",      type=str,   default=[], nargs='*')
    parser.add_argument("--regu_scale",     type=float, default=[], nargs='*')
    parser.add_argument("--lr_start",       type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate",  type=float, default=0.98)
    parser.add_argument("--lr_decay_step",  type=int,   default=160)

    # Saving Settings
    parser.add_argument("--iter_save",      type=int,   default=2000)
    parser.add_argument("--res_res",        type=float, default=1.1)
    parser.add_argument("--query_type",     type=str,   default='grid')

    print('Loaded all arguments successfully.')
    return parser


def config_parser_ada_mus():

    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, default='./configs/example.txt')
    print('Loaded config file successfully.')

    # Experiment settings
    parser.add_argument("--exp_type",       type=str)
    parser.add_argument("--exp_no",         type=str)
    parser.add_argument("--auto",           type=int,   default=0)
    parser.add_argument("--data_dir_rel",   type=str,   default='./data/')
    parser.add_argument("--save_dir_rel",   type=str,   default='./logs/')

    # Phantom and optical settings
    parser.add_argument("--phantom_x",      type=float, default=50)
    parser.add_argument("--phantom_y",      type=float, default=50)
    parser.add_argument("--phantom_z",      type=float, default=16)
    parser.add_argument("--te_maxvol",      type=float, default=0.6)
    parser.add_argument("--node_size",      type=float, default=0.1)
    parser.add_argument("--tar_shape",      type=str,   default='2bar')
    parser.add_argument("--ray_type",       type=str,   default='reflective')
    parser.add_argument("--data_name",      type=str)
    parser.add_argument("--mua",            type=float, default=0.1)
    parser.add_argument("--mus_init",       type=float, default=1.0)
    parser.add_argument("--ref",            type=float, default=1.4)
    parser.add_argument("--q_dim",          type=int,   default=11)
    parser.add_argument("--q_gap",          type=int,   default=4)
    parser.add_argument("--q_offset",       type=int,   default=5)
    parser.add_argument("--m_dim",          type=int,   default=49)
    parser.add_argument("--m_gap",          type=int,   default=1)
    parser.add_argument("--m_offset",       type=int,   default=1)
    parser.add_argument("--noise_level",    type=float, default=0)
    parser.add_argument("--em_scale_gap",   type=float, default=1)

    # NN build and train settings
    parser.add_argument("--enc_type",       type=str,   default='PE')
    parser.add_argument("--enc_level",      type=int,   default=4)
    parser.add_argument("--enc_level_z",    type=int,   default=4)
    parser.add_argument("--layer_dim",      type=int,   default=256)
    parser.add_argument("--activation_fn",  type=str,   default='leaky_relu')
    parser.add_argument("--n_iter",         type=int,   default=16000)
    parser.add_argument("--loss_type",      type=str,   default='MSE')
    parser.add_argument("--loss_scale",     type=float, default=1e6)
    parser.add_argument("--n_regu_term",    type=int,   default=0)
    parser.add_argument("--regu_type",      type=str,   default=[], nargs='*')
    parser.add_argument("--regu_scale",     type=float, default=[], nargs='*')
    parser.add_argument("--lr_start",       type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate",  type=float, default=0.99)
    parser.add_argument("--lr_decay_step",  type=int,   default=160)
    parser.add_argument("--lr_mu",          type=float, default=2e-3)
    parser.add_argument("--T_mu",           type=float, default=80)

    # Saving Settings
    parser.add_argument("--iter_save",      type=int,   default=2000)
    parser.add_argument("--res_res",        type=float, default=1.1)
    parser.add_argument("--query_type",     type=str,   default='grid')

    print('Loaded all arguments successfully.')
    return parser


def config_parser_neuflim():
    """Configuration parser for INR-based time-domain yield/lifetime imaging."""

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '--config', is_config_file=True, default='./configs/NeuFLIM_peanut_example.txt'
    )

    # Experiment and paths
    parser.add_argument("--exp_type", type=str, required=True)
    parser.add_argument("--exp_no", type=str, required=True)
    parser.add_argument("--data_dir_rel", type=str, default='./data/')
    parser.add_argument("--save_dir_rel", type=str, default='./logs/')
    parser.add_argument("--tar_shape", type=str, default='2bar')
    parser.add_argument("--ray_type", type=str, default='reflective')
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--noise_level", type=float, default=0)
    parser.add_argument("--em_scale_gap", type=float, default=1)
    parser.add_argument("--device", type=str, default='')

    # Phantom geometry
    parser.add_argument("--phantom_x", type=float, default=50)
    parser.add_argument("--phantom_y", type=float, default=50)
    parser.add_argument("--phantom_z", type=float, default=16)
    parser.add_argument("--te_maxvol", type=float, default=0.6)
    parser.add_argument("--node_size", type=float, default=0.1)

    # Time-domain input files
    parser.add_argument("--node_file", type=str, default='node.mat')
    parser.add_argument("--node_key", type=str, default='node')
    parser.add_argument("--mvec_file", type=str, default='mvec.mat')
    parser.add_argument("--mvec_key", type=str, default='mvec')
    parser.add_argument("--tphi_ex_file", type=str, default='tphi_ex.mat')
    parser.add_argument("--tphi_ex_key", type=str, default='tphi_ex')
    parser.add_argument("--a_em_file", type=str, default='A_em.mat')
    parser.add_argument("--a_em_key", type=str, default='A_em')
    parser.add_argument("--b_em_file", type=str, default='B_em.mat')
    parser.add_argument("--b_em_key", type=str, default='B_em')
    parser.add_argument("--meas_em_file", type=str, default='meas_em_td.mat')
    parser.add_argument("--meas_em_key", type=str, default='meas_em_td')
    parser.add_argument("--mask_file", type=str, default='')
    parser.add_argument("--mask_key", type=str, default='mask')
    parser.add_argument("--tphi_time_axis", type=int, default=0)
    parser.add_argument("--meas_time_axis", type=int, default=0)
    parser.add_argument("--mask_time_axis", type=int, default=0)

    # Time discretization. dt and lifetime limits must use the same unit.
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--nstep", type=int, default=0)
    parser.add_argument("--lifetime_scheme", type=str, default='exact', choices=['exact', 'euler'])
    parser.add_argument("--tau_epsilon", type=float, default=1e-8)

    # INR architecture and physical output ranges
    parser.add_argument("--enc_type", type=str, default='PE')
    parser.add_argument("--enc_level", type=int, default=4)
    parser.add_argument("--enc_level_z", type=int, default=4)
    parser.add_argument("--layer_dim", type=int, default=256)
    parser.add_argument("--activation_fn", type=str, default='relu')
    parser.add_argument("--yield_min", type=float, default=0.0)
    parser.add_argument("--yield_max", type=float, default=1.5)
    parser.add_argument("--lifetime_min", type=float, default=0.01)
    parser.add_argument("--lifetime_max", type=float, default=3.0)

    # Optimization and losses
    parser.add_argument("--n_iter", type=int, default=16000)
    parser.add_argument("--loss_type", type=str, default='MSE')
    parser.add_argument("--loss_scale", type=float, default=1.0)
    parser.add_argument(
        "--measurement_normalization",
        type=str,
        default='global_max',
        choices=['none', 'global_max', 'global_l2', 'per_tpsf_max'],
    )
    parser.add_argument("--normalization_epsilon", type=float, default=1e-12)
    parser.add_argument("--yield_regu_scale", type=float, default=0.0)
    parser.add_argument("--lifetime_regu_scale", type=float, default=0.0)
    parser.add_argument("--lifetime_prior", type=float, default=0.5)
    parser.add_argument("--lr_start", type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate", type=float, default=0.98)
    parser.add_argument("--lr_decay_step", type=int, default=160)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # Saving and plotting
    parser.add_argument("--iter_save", type=int, default=2000)
    parser.add_argument("--res_res", type=float, default=1.1)
    parser.add_argument("--save_measurement", type=int, default=0)
    parser.add_argument("--plot_source_idx", type=int, default=0)
    parser.add_argument("--plot_detector_idx", type=int, default=0)
    parser.add_argument("--plot_log_tpsf", type=int, default=0)

    return parser
