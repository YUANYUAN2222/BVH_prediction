import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from common.bvh_joints import BVHParents


def set_equal_aspect(ax, data):
    """
    Create white cubic bounding box to make sure that 3d axis is in equal aspect.
    :param ax: 3D axis
    :param data: shape of(frames, 3), generated from BVH using convert_bvh2dataset.py
    """
    X, Y, Z = data[..., 0], data[..., 1], data[..., 2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def viz_data3d(data, save_path, fps=30):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Test')

    # ax.set_aspect('equal') Not work, using self-defined tricky
    set_equal_aspect(ax, data)

    points = None
    lines = None

    def get_plot_para(this_data, i, p):
        return [this_data[i][0], this_data[p][0]], \
               [this_data[i][1], this_data[p][1]], \
               [this_data[i][2], this_data[p][2]]

    def init():
        nonlocal points, lines
        first = data[0]
        points, = ax.plot(*first.T, 'bo', ms=6)

        lines = [ax.plot(*get_plot_para(first, i, p), zdir='z') if p != -1 else [] for i, p in enumerate(BVHParents)]

        return points

    def update_plot(frame):
        nonlocal points, lines
        this_data = data[frame]

        points.set_data(this_data.T[:2])
        points.set_3d_properties(this_data.T[2], 'z')
        points.set_markersize(6)

        for i, line in enumerate(lines):
            if not line:  # omit root
                continue

            p = BVHParents[i]
            line3d = line[0]

            xyz = get_plot_para(this_data, i, p)
            line3d.set_xdata(xyz[0])
            line3d.set_ydata(xyz[1])
            line3d.set_3d_properties(xyz[2], zdir='z')

        return points

    # Creating the Animation object
    line_ani = FuncAnimation(fig, update_plot, frames=len(data), init_func=init, interval=1000 / fps)
    plt.rcParams['animation.ffmpeg_path'] = 'F:\\ffmpeg\\bin\\ffmpeg.exe'
    line_ani.save(save_path, writer='ffmpeg', fps=30)


def preprocess_xyz(npy):
    """
    Load xyzw data
    :param npy: loaded numpy array
    :return:
    """
    xyz = npy[:, 0]

    # Change xzy(BVH format) into xyz, then reverse x (x ---> -x)
    xyz[..., [1, 2]] = xyz[..., [2, 1]]
    xyz[..., 0] = -xyz[..., 0]

    return xyz


def viz_folder(folder_path):
    npy_files = [os.path.join(folder_path, name) for name in filter(lambda x: '.npy' in x, os.listdir(folder_path))]
    pbar = tqdm(total=len(npy_files))

    pool = Pool()
    for npy_path in npy_files:
        pool.apply_async(viz_file, args=(npy_path,), callback=lambda _: pbar.update())

    pool.close()
    pool.join()

    pbar.close()


def viz_file(npy_path):
    save_path = npy_path.replace('.npy', '.mp4')
    data = preprocess_xyz(np.load(npy_path, allow_pickle=True))

    viz_data3d(data, save_path)


if __name__ == '__main__':
    viz_folder('../dataset/processed')
