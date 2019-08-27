import multiprocessing
import os

import numpy as np
import scipy.ndimage.filters as filters
from tqdm import tqdm

import data_gen.motion.Animation as Animation
import data_gen.motion.BVH as BVH
from common.bvh_joints import BVHJoints, BVH_adj
from data_gen import utils
from data_gen.configs import config
from data_gen.motion.Quaternions import Quaternions
from visualization.bvh_viz import viz_folder

FRAME_RATE = config.data.FRAME_RATE


def extract_data(anim):
    glob_joint_trans = Animation.transforms_global(anim)
    glob_joint_pos = glob_joint_trans[:, :, :3, 3] / glob_joint_trans[:, :, 3:, 3]
    glob_joint_rot = Quaternions.from_transforms(glob_joint_trans)
    """ Extract Forward Direction """

    sdr_l, sdr_r, hip_l, hip_r = config.data.sdr_l, config.data.sdr_r, config.data.hip_l, config.data.hip_r
    across = ((glob_joint_pos[:, sdr_l] - glob_joint_pos[:, sdr_r])
              + (glob_joint_pos[:, hip_l] - glob_joint_pos[:, hip_r]))
    across = utils.normalize_vec(across)
    """ Smooth Forward Direction """

    direction_filter_width = 20
    glob_root_dir = filters.gaussian_filter1d(np.cross(across, np.array([[0, 1, 0]])),
                                              direction_filter_width, axis=0, mode='nearest')
    glob_root_dir = utils.normalize_vec(glob_root_dir)

    glob_root_pos = glob_joint_pos[:, 0:1, :].copy()
    glob_root_pos[..., 1] = 0
    glob_root_rot = Quaternions.between(glob_root_dir, np.array([0, 0, 1]))[:, np.newaxis]

    """ Local Space """

    local_joint_pos = utils.get_rela_pos_to(glob_joint_pos[1:], glob_root_pos[: -1], glob_root_rot[: -1])

    local_joint_trans = (glob_root_rot[: -1] * glob_joint_rot[1:]).transforms()
    local_joint_up = local_joint_trans[..., 1]
    local_joint_forward = local_joint_trans[..., 2]
    # print(local_joint_forward)

    """Start convert"""
    assert local_joint_pos.shape == local_joint_up.shape == local_joint_forward.shape, 'joints position and rotation shape not matched!'

    result = [np.array([pos, up, forward]) for pos, up, forward in zip(local_joint_pos, local_joint_up, local_joint_forward)]

    return np.array(result)


def distance(a, b):
    """
    Calculate the distance between a and b. (x1, y1, z1) & (x2, y2, z2)
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b)


def _skeleton_standardization(data, spine_len=0.741):
    """
    Standardize the skeleton according to the spine length.
    :param data: The data of one frame [21, xzy]
    :param spine_len: The length from Pelvis --> Spine --> Spine1 --> Head.
    :return:
    """
    spine_index = [BVHJoints.Pelvis, BVHJoints.Spine, BVHJoints.Spine1, BVHJoints.Head]
    spines = [data[i.value] for i in spine_index]

    # The scale index of skeletons
    scale = spine_len / sum([distance(spines[i], spines[i + 1]) for i in range(len(spines) - 1)])

    def dfs(parent, node, change):
        """
        Update node position and influenced positions recursively.
        :param parent: The base point which keep stable.
        :param node: The end point which changes.
        :param change: The cumulative changes influenced by the formal point change.
        :return:
        """
        data[node] += change
        relative_change = (data[node] - data[parent]) * (scale - 1)
        data[node] += relative_change
        change = change + relative_change  # Don't use +=, which would lead to in-place add for ndarray

        influenced = BVH_adj[node]
        for i in influenced:
            dfs(node, i, change)

    # DFS update
    for n in BVH_adj[BVHJoints.Pelvis.value]:
        dfs(BVHJoints.Pelvis.value, n, change=0)


def standardize_data(data):
    """
    1. Put Pelvis into (0, 0, 0)
    2. Standardize xzy into [-1, 1]
    :param data: data to be normalized, shape of (frames, 3, 21, xzy)
    :return: normalized data
    """
    # Drop y of Pelvis into 0
    data[:, 0] -= np.tile(np.expand_dims(data[:, 0, BVHJoints.Pelvis.value], 1), (1, len(BVHJoints), 1))

    # Standardize
    # Standardize position according to the skeleton length
    for d in range(len(data)):
        _skeleton_standardization(data[d][0])

    # TODO: Standardize rotations

    return data


def process_worker(anim_dir, anim_name, prep_dir):
    anim_file = os.path.join(anim_dir, anim_name)

    anim, names, _ = BVH.load(anim_file)
    anim = utils.down_sample(anim, fps=30)

    prep_file = os.path.join(prep_dir, anim_name.replace('.bvh', '.npy'))
    # print('Save prepare file', prep_file)

    data = extract_data(anim)
    result = standardize_data(data)

    np.save(prep_file, result, allow_pickle=True)


def main():
    anim_dir = os.path.join(config.data.data_dir, config.data.data_name)
    prep_dir = os.path.join(config.data.data_dir, config.data.data_processed_name)
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)

    def check_valid(x):
        return 'bvh' in x and not (config.data.withBall and x.startswith('without_ball'))

    print('pre-processing...')
    anim_names = list(filter(check_valid, os.listdir(anim_dir)))
    pbar = tqdm(total=len(anim_names))
    pool = multiprocessing.Pool(1)
    for anim_name in anim_names:
        # pool.apply_async(func=process_worker, args=(anim_dir, anim_name, prep_dir), callback=lambda _: pbar.update())
        process_worker(anim_dir, anim_name, prep_dir)
        pbar.update()

    pool.close()
    pool.join()
    pbar.close()

    print('visualizing...')
    viz_folder(os.path.join(config.data.data_dir, config.data.data_processed_name))


if __name__ == '__main__':
    main()
