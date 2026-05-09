import os
import sys

def set_io_path(exp_type: str, exp_no: int, data_dir: str, save_dir: str,
                tar_shape: str, ray_type: str, n_node: int, noise_level: float = 0, phantom_rev: int = 0) \
               -> list[str]:
   
    if not phantom_rev:
        if not noise_level:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], str(n_node)))
        else:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], str(n_node), 'noise-'+noise_level))
    else:
        if not noise_level:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], str(n_node), 'alt'+phantom_rev))
        else:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], str(n_node), 'noise-'+noise_level, 'alt'+phantom_rev))
   
    data_path = os.path.join(os.getcwd(), data_dir, exp_setting)
    save_path = os.path.join(os.getcwd(), save_dir, exp_setting, str(exp_no))

    if os.path.isdir(data_path):
        print('Data directory:', data_path)
    else:
         raise NotADirectoryError('Expected', data_path)

    os.makedirs(save_path, exist_ok=True)
    print('Result directory:', save_path)
    save_mod_path = os.path.join(save_path, 'model')
    save_img_path = os.path.join(save_path, 'image')
    save_mat_path = os.path.join(save_path, 'matrix')
    os.makedirs(save_mod_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_mat_path, exist_ok=True)

    return [data_path, save_path, save_mod_path, save_img_path, save_mat_path]


def set_io_path_auto(exp_type: str, exp_no: int, data_dir: str, save_dir: str,
                     tar_shape: str, ray_type: str, data_name: str, noise_level: float = 0, phantom_rev: int = 0) \
                    -> list[str]:
   
    if not phantom_rev:
        if not noise_level:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name))
        else:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name, 'noise-'+noise_level))
    else:
        if not noise_level:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name, 'alt'+phantom_rev))
        else:
            exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name, 'noise-'+noise_level, 'alt'+phantom_rev))
   
    data_path = os.path.join(os.getcwd(), data_dir, exp_setting)
    save_path = os.path.join(os.getcwd(), save_dir, exp_setting, str(exp_no))

    if os.path.isdir(data_path):
        print('Data directory:', data_path)
    else:
         raise NotADirectoryError('Expected', data_path)

    os.makedirs(save_path, exist_ok=True)
    print('Result directory:', save_path)
    save_mod_path = os.path.join(save_path, 'model')
    save_img_path = os.path.join(save_path, 'image')
    save_mat_path = os.path.join(save_path, 'matrix')
    os.makedirs(save_mod_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_mat_path, exist_ok=True)

    save_gibbs_path = os.path.join(save_path, 'gibbs')
    os.makedirs(save_gibbs_path, exist_ok=True)

    return [data_path, save_path, save_mod_path, save_img_path, save_mat_path]


def set_io_path_v4(exp_type: str, exp_no: int, data_dir: str, save_dir: str,
                   tar_shape: str, ray_type: str, data_name: str, noise_level: float = 0) \
                -> list[str]:
   
    if not noise_level:
        exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name))
    else:
        exp_setting = '-'.join((tar_shape, exp_type, ray_type[:3], data_name, 'noise-'+noise_level))
   
    data_path = os.path.join(os.getcwd(), data_dir, exp_setting)
    save_path = os.path.join(os.getcwd(), save_dir, exp_setting, str(exp_no))

    if os.path.isdir(data_path):
        print('Data directory:', data_path)
    else:
         raise NotADirectoryError('Expected', data_path)

    os.makedirs(save_path, exist_ok=True)
    print('Result directory:', save_path)
    save_mod_path = os.path.join(save_path, 'model')
    save_img_path = os.path.join(save_path, 'image')
    save_mat_path = os.path.join(save_path, 'matrix')
    os.makedirs(save_mod_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_mat_path, exist_ok=True)

    return [data_path, save_path, save_mod_path, save_img_path, save_mat_path]
