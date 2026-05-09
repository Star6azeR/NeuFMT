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
from scripts.Config_Parser import config_parser_v4 as config_parser
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

        S = hdf5storage.loadmat(os.path.join(data_dir, "S.mat"))['S_full']
        S = torch.tensor(S, device=device).float()          # trade GPU memory space for speed
        print('S:', S.shape)
        S_inv = torch.linalg.inv(S)

        phi_ex = hdf5storage.loadmat(os.path.join(data_dir, "phi_ex.mat"))['phi_ex']
        phi_ex = torch.tensor(phi_ex, device=device).float()
        print('phi_ex:', phi_ex.shape)

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
    

    # NN instantiation
    # net = NeFluor(input_dim, args.layer_dim, args.activation_fn).to(device)
    net = NeRF(input_dim, args.layer_dim, args.activation_fn).to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr_start, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
    optimizer.zero_grad()


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

    writer = SummaryWriter(save_dir)


    # Training loop begin
    tqd = trange(args.n_iter)
    for iter in tqd:
        
        # Inter-iteration FEM forward simulation
        flr_pred = net(node_enc)
        phi_em_pred = torch.mm(S_inv, phi_ex.mul(flr_pred))
        meas_em_pred = torch.mm(mvec_trans, phi_em_pred)

        loss = error_fn(args.loss_scale * meas_em_pred, args.loss_scale * meas_em)

        regu_term = 0.0 * loss

        if args.n_regu_term:
            for i in range(args.n_regu_term):
                regu_term += args.regu_scale[i] * regu_fn[i](flr_pred, torch.zeros_like(flr_pred))
            loss += regu_term

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Per-iteration loss record
        tqd.set_postfix(loss=loss.item())
        writer.add_scalar('lossi', loss.item(), global_step=iter+1)

        # Manual learning rate decay
        if (iter+1) % args.lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.lr_decay_rate

        # Save results
        if (iter+1) % args.iter_save == 0:

            res_mod_fpath = os.path.join(save_mod_dir, str(iter+1) + '.pyt')
            res_img_fpath = os.path.join(save_img_dir, str(iter+1) + '.png')
            res_mat_fpath = os.path.join(save_mat_dir, str(iter+1) + '.mat')

            torch.save({
                'iteration': iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
    # python3 main_v4.py --config configs/your_config.txt
    train()