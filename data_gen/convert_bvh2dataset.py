import multiprocessing
import os

import numpy as np
import scipy.ndimage.filters as filters
from tqdm import tqdm

import data_gen.motion.Animation as Animation
import data_gen.motion.BVH as BVH
from data_gen.common import utils
from data_gen.configs import config
from data_gen.motion.Quaternions import Quaternions

FRAME_RATE = config.data.FRAME_RATE


def get_index(list_list, value):
    for i, l in enumerate(list_list):
        if value in l:
            return i


def process_data(anim):
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


def process_worker(anim_dir, anim_name, prep_dir):
    anim_file = os.path.join(anim_dir, anim_name)
    # print('Load animation file', anim_file)

    anim, names, _ = BVH.load(anim_file)
    # for i in range(len(names)):
    #     print(i, names[i])

    anim = utils.down_sample(anim, fps=30)

    # print("len(anim): ", len(anim))

    prep_file = os.path.join(prep_dir, anim_name.replace('.bvh', '.npy'))
    # print('Save prepare file', prep_file)

    result = process_data(anim)

    np.save(prep_file, result, allow_pickle=True)


def main():
    anim_dir = os.path.join(config.data.data_dir, config.data.data_name)
    prep_dir = os.path.join(config.data.data_dir, config.data.data_processed_name)
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)

    def check_valid(x):
        return 'bvh' in x and not (config.data.withBall and x.startswith('without_ball'))

    anim_names = list(filter(check_valid, os.listdir(anim_dir)))
    pbar = tqdm(total=len(anim_names))
    pool = multiprocessing.Pool()
    for anim_name in anim_names:
        pool.apply_async(func=process_worker, args=(anim_dir, anim_name, prep_dir), callback=lambda _: pbar.update())

    pool.close()
    pool.join()
    pbar.close()

    # print('Done')


if __name__ == '__main__':
    main()
