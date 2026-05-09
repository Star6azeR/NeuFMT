import os
import numpy as np
import torch
import torch.nn as nn
import hdf5storage
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
from tqdm import trange
import time

import matlab.engine

from torch.utils.tensorboard import SummaryWriter

from models.NeRF import *
from models.Encoder import *
from models.SysmatCom import *
from scripts.Config_Parser import config_parser_ada_mus as config_parser
from scripts.Path_Setter import set_io_path_v4 as set_io_path


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


def train():
    
    parser = config_parser()
    args = parser.parse_args()

    if len(args.regu_type) != args.n_regu_term or len(args.regu_scale) != args.n_regu_term:
        raise TypeError('Unmatched number of regularization terms')
    
    [data_dir, save_dir, save_mod_dir, save_img_dir, save_mat_dir] = \
        set_io_path(args.exp_type, args.exp_no, args.data_dir_rel, args.save_dir_rel,\
                    args.tar_shape, args.ray_type, args.data_name, args.noise_level)
    
    eng = matlab.engine.start_matlab()
    eng.addpath('./scripts')
    

    # Load data
    if os.path.isdir(data_dir):

        node = hdf5storage.loadmat(os.path.join(data_dir, "node.mat"))['node']
        node = torch.tensor(node, device=device).float()
        node = node * torch.tensor([1/args.phantom_x, 1/args.phantom_y, 1/args.phantom_z]).to(device)
        print('node:', node.shape)

        mvec = hdf5storage.loadmat(os.path.join(data_dir, "mvec.mat"))['mvec']
        mvec = torch.tensor(mvec.toarray().real).float()    # No .to(device) to save GPU memory
        print('mvec:', mvec.shape)
        mvec_trans = torch.transpose(mvec,0,1).to(device)

        qvec = hdf5storage.loadmat(os.path.join(data_dir, "qvec.mat"))['qvec']
        qvec = torch.tensor(qvec.toarray().real, device=device).float()
        print('qvec:', qvec.shape)

        S1_one = hdf5storage.loadmat(os.path.join(data_dir, "S1_one.mat"))['S1_one_full']
        S1_one = torch.tensor(S1_one, device=device).float()
        S2_one = hdf5storage.loadmat(os.path.join(data_dir, "S2_one.mat"))['S2_one_full']
        S2_one = torch.tensor(S2_one, device=device).float()
        S3_one = hdf5storage.loadmat(os.path.join(data_dir, "S3_one.mat"))['S3_one_full']
        S3_one = torch.tensor(S3_one, device=device).float()
        print('S:', S1_one.shape)

        if args.exp_type == 'simu':
            meas_em = hdf5storage.loadmat(os.path.join(data_dir, "meas_em_simu.mat"))['meas_em_simu']
            meas_em = torch.tensor(meas_em, device=device).float()
        else:
            meas_em = hdf5storage.loadmat(os.path.join(data_dir, "meas_em.mat"))['meas_em']
            meas_em = np.float16(meas_em)
            meas_em = torch.tensor(meas_em, device=device).float()
            meas_em = args.em_scale_gap * meas_em
        print('meas_em:', meas_em.shape)

        print("torch.cuda.memory_allocated: %.2fGB"
              %(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %.2fGB"
              %(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %.2fGB"
              %(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    else:
        raise NotADirectoryError
    

    # Solution basis is grid basis
    if args.res_res == 1.0:
        x_cor = torch.linspace(start=0.5, end=args.phantom_x-0.5, steps=int(args.phantom_x)) / args.phantom_x
        y_cor = torch.linspace(start=0.5, end=args.phantom_y-0.5, steps=int(args.phantom_y)) / args.phantom_y
        z_cor = torch.linspace(start=0.5, end=args.phantom_z-0.5, steps=int(args.phantom_z)) / args.phantom_z
    elif args.res_res == 1.1:
        x_cor = torch.linspace(start=0, end=args.phantom_x, steps=int(args.phantom_x)+1) / args.phantom_x
        y_cor = torch.linspace(start=0, end=args.phantom_y, steps=int(args.phantom_y)+1) / args.phantom_y
        z_cor = torch.linspace(start=0, end=args.phantom_z, steps=int(args.phantom_z)+1) / args.phantom_z
    elif args.res_res == 2.0:
        x_cor = torch.linspace(start=0.25, end=args.phantom_x-0.25, steps=int(2*args.phantom_x)) / args.phantom_x
        y_cor = torch.linspace(start=0.25, end=args.phantom_y-0.25, steps=int(2*args.phantom_y)) / args.phantom_y
        z_cor = torch.linspace(start=0.25, end=args.phantom_z-0.25, steps=int(2*args.phantom_z)) / args.phantom_z
    grid = torch.cartesian_prod(x_cor, y_cor, z_cor)
    grid = grid.to(device)


    # Input encoding
    if args.enc_type == 'PE':
        node_enc = positional_encoding_default(node, args.enc_level)
        grid_enc = positional_encoding_default(grid, args.enc_level)
        input_dim = 6 * args.enc_level
    elif args.enc_type == 'PE_altz':
        node_enc = positional_encoding_altz(node, args.enc_level, args.enc_level_z)
        grid_enc = positional_encoding_altz(grid, args.enc_level, args.enc_level_z)
        input_dim = 4 * args.enc_level + 2 * args.enc_level_z
    else:
        raise NotImplementedError
    

    # Instantiation
    net = NeRF(input_dim, args.layer_dim, args.activation_fn).to(device)
    mus = nn.Parameter(torch.tensor([args.mus_init], device=device, requires_grad=True))

    optimizer_nn = torch.optim.Adam(net.parameters(),
                                 lr=args.lr_start, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
    optimizer_nn.zero_grad()
    optimizer_c = torch.optim.SGD([mus], lr=args.lr_mu)
    optimizer_c.zero_grad()


    # Loss function - error part
    if args.loss_type in {"MSE", "L2"}:
        error_fn = nn.MSELoss(reduction='mean')
    elif args.loss_type in {"MAE", "L1"}:
        error_fn = nn.L1Loss(reduction='mean')
    else:
        raise NotImplementedError


    # Loss function - regularization part
    regu_fn = []
    for i in range(args.n_regu_term):
        curr_regu_type = args.regu_type[i]
        if curr_regu_type in {"MSE", "L2"}:
            regu_fn.append(nn.MSELoss(reduction='mean'))
        elif curr_regu_type in {"MAE", "L1"}:
            regu_fn.append(nn.L1Loss(reduction='mean'))
        else:
            raise NotImplementedError
        

    # S initialization
    mua = args.mua
    ref = args.ref
    S = SysmatCom(S1_one, S2_one, S3_one, mua, mus, ref)
    S_inv = torch.linalg.inv(S)


    writer = SummaryWriter(save_dir)

    # Training loop begin
    tqd = trange(args.n_iter)
    for iter in tqd:

        if mus > 10.0 or mus < 0.0:
            break

        update_mu = (iter + 1 > 1000) and ((iter + 1) % args.T_mu == 0)
        if update_mu:
                S = SysmatCom(S1_one, S2_one, S3_one, mua, mus, ref)
                S_inv = torch.linalg.inv(S)
        else:
            S_inv = S_inv.detach()
        
        # Inter-iteration FEM forward simulation
        flr_pred = net(node_enc)
        phi_ex = torch.mm(S_inv, qvec)
        phi_em_pred = torch.mm(S_inv, phi_ex.mul(flr_pred))
        meas_em_pred = torch.mm(mvec_trans, phi_em_pred)

        loss = error_fn(args.loss_scale * meas_em_pred, args.loss_scale * meas_em)

        regu_term = 0.0 * loss

        if (not update_mu) and args.n_regu_term:
            for i in range(args.n_regu_term):
                regu_term += args.regu_scale[i] * regu_fn[i](flr_pred, torch.zeros_like(flr_pred))
            loss += regu_term

        loss.backward()

        optimizer_nn.step()
        optimizer_nn.zero_grad()

        if update_mu:
            optimizer_c.step()
            optimizer_c.zero_grad()
            S = SysmatCom(S1_one, S2_one, S3_one, mua, mus, ref)
            S_inv = torch.linalg.inv(S)
        else:
            mus.grad = None
        
        if update_mu:
            writer.add_scalar('mus', mus.item(), global_step=iter+1)
            writer.add_scalar('lossi', loss.item(), global_step=iter+1)
            writer.add_scalar('flr_pred_max', flr_pred.max().item(), global_step=iter+1)

        # Per-iteration loss record
        tqd.set_postfix(loss=loss.item(), mus=mus.item(),
                        flr_pred_max=flr_pred.max().item())

        # Manual learning rate decay
        if (iter+1) % args.lr_decay_step == 0:
            for param_group in optimizer_nn.param_groups:
                param_group['lr'] = param_group['lr'] * args.lr_decay_rate

        # Save results
        if (iter+1) % args.iter_save == 0:

            res_mod_fpath = os.path.join(save_mod_dir, str(iter+1) + '.pyt')
            res_img_fpath = os.path.join(save_img_dir, str(iter+1) + '.png')
            res_mat_fpath = os.path.join(save_mat_dir, str(iter+1) + '.mat')

            torch.save({
                'iteration': iter,
                'model_state_dict': net.state_dict(),
                'optimizer_nn_state_dict': optimizer_nn.state_dict(),
                'optimizer_c_state_dict': optimizer_c.state_dict(),
                'loss': loss,
                }, res_mod_fpath)

            if args.query_type == 'mesh':
                flr_pred_mat = flr_pred.cpu().detach().numpy().astype('double')
                hdf5storage.savemat(res_mat_fpath, {'flr_pred': flr_pred_mat}, format='7.3')
                eng.mat_plot_mesh(args.phantom_x, args.phantom_y, args.phantom_z,
                                  args.te_maxvol, args.node_size, res_mat_fpath, res_img_fpath,
                                  nargout=0)

            elif args.query_type == 'grid':
                flr_pred_grid = net(grid_enc)
                flr_pred_grid_mat = flr_pred_grid.cpu().detach().numpy().astype('double')
                if args.res_res == 1.0:
                    flr_pred_grid_mat = np.reshape(flr_pred_grid_mat, (int(args.phantom_x), int(args.phantom_y), int(args.phantom_z)))
                elif args.res_res == 1.1:
                    flr_pred_grid_mat = np.reshape(flr_pred_grid_mat, (int(args.phantom_x)+1, int(args.phantom_y)+1, int(args.phantom_z)+1))
                else:
                    raise NotImplementedError
                hdf5storage.savemat(res_mat_fpath, {'flr_pred_grid': flr_pred_grid_mat}, format='7.3')
                eng.mat_plot_grid(args.phantom_x, args.phantom_y, args.phantom_z, args.res_res,
                                res_mat_fpath, res_img_fpath, nargout=0)
    
    eng.quit()

    print('Train done.')


if __name__=='__main__':
    # python3 NeuFMT-Ada_main.py --config configs/your_config.txt
    train()