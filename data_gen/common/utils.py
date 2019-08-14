import os.path

import numpy as np


def vec_normal(vec):
    return np.linalg.norm(vec)


def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)


def normalize_training(x, axis, save_file=None):
    x_mean, x_std = x.mean(axis=axis), x.std(axis=axis)
    x_std = np.where(x_std == 0, 1, x_std)
    norm_x = (x - x_mean) / x_std
    if save_file is not None:
        x_mean.tofile(save_file + 'mean.bin')
        x_std.tofile(save_file + 'std.bin')
    return norm_x


def normalize_infer(x, save_file):
    x_mean = np.fromfile(save_file + 'mean.bin', dtype=np.float32)
    x_std = np.fromfile(save_file + 'std.bin', dtype=np.float32)
    norm_x = (x - x_mean) / x_std
    return norm_x


def renormalize(y, save_file):
    y_mean = np.fromfile(save_file + 'mean.bin', dtype=np.float32)
    y_std = np.fromfile(save_file + 'std.bin', dtype=np.float32)
    renorm_y = y_mean + y * y_std
    return renorm_y


def lerp(start, end, ratio):
    return (1 - ratio) * start + ratio * end


def normalize_vec(vec):
    vec = vec / (np.sqrt(np.sum(vec ** 2, axis=-1)) + 1e-10)[:, np.newaxis]
    return vec


def get_rela_pos_to(global_pos, root_pos, root_rot):
    local_pos = root_rot * (global_pos - root_pos)
    # print 222, local_pos[1]
    # print type(local_pos)
    return local_pos


def get_rela_dir_to(global_dir, root_rot):
    return root_rot * global_dir


def get_rela_pos_from(local_pos, root_pos, root_rot):
    global_pos = root_pos + (-root_rot) * local_pos
    return global_pos


def get_rela_dir_from(local_dir, root_rot):
    return (-root_rot) * local_dir


def down_sample(origin, fps=30):
    """
    FPS only support factors of 60.
    TODO: support all framerate.
    """
    assert 60 % fps == 0, 'Only support fps == factors of 60.'

    return origin[::60 // fps]


if __name__ == '__main__':
    a = list(range(120))
    print(down_sample(a))