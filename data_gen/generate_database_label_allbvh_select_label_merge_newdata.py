import os

import scipy.ndimage.filters as filters
from data_gen.motion.Quaternions import Quaternions
import data_gen.motion.Animation as Animation
from data_gen.motion.Pivots import Pivots
import numpy as np
import data_gen.motion.BVH as BVH
from data_gen import utils
import json
import itertools
import multiprocessing

from data_gen.configs import config

FRAME_RATE = config.data.FRAME_RATE


def get_index(list_list, value):
    for i, l in enumerate(list_list):
        if value in l:
            return i


def process_data(anim, styles):
    # import pdb
    # pdb.set_trace()
    glob_joint_trans = Animation.transforms_global(anim)
    glob_joint_pos = glob_joint_trans[:, :, :3, 3] / glob_joint_trans[:, :, 3:, 3]
    glob_joint_rot = Quaternions.from_transforms(glob_joint_trans)
    glob_joint_vel = (glob_joint_pos[1:] - glob_joint_pos[: -1]) * FRAME_RATE
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
    glob_root_vel = (glob_root_pos[1:] - glob_root_pos[: -1]) * FRAME_RATE
    local_root_r_vel = Pivots.from_quaternions(glob_root_rot[1:] * (-glob_root_rot[: -1])).ps

    """ Local Space """

    local_joint_pos = utils.get_rela_pos_to(glob_joint_pos[1:], glob_root_pos[: -1], glob_root_rot[: -1])
    local_joint_vel = utils.get_rela_dir_to(glob_joint_vel, glob_root_rot[: -1])

    local_joint_trans = (glob_root_rot[: -1] * glob_joint_rot[1:]).transforms()
    local_joint_up = local_joint_trans[..., 1]
    local_joint_forward = local_joint_trans[..., 2]
    # print(local_joint_forward)
    """ Styles """

    local_style = np.zeros([len(styles[0]),
                            len(config.data.hand_styles_sel) + len(config.data.foot_styles_sel) + len(config.data.body_style_sel)])
    for i in range(len(styles[0])):
        foot_ind = get_index(config.data.foot_styles_sel, styles[0][i])
        local_style[i, foot_ind] = 1
        hand_ind = get_index(config.data.hand_styles_sel, styles[1][i])
        local_style[i, len(config.data.foot_styles_sel) + hand_ind] = 1
        body_ind = get_index(config.data.body_style_sel, styles[2][i])
        local_style[i, len(config.data.foot_styles_sel) + len(config.data.hand_styles_sel) + body_ind] = 1

    """ Start Windows """

    xc, yc = [], []

    # import pdb
    # pdb.set_trace()

    window = config.data.window
    print(window)
    for i in range(window + 1, len(anim) - window, 1):
        # print(i)
        local_traj_pos = utils.get_rela_pos_to(glob_root_pos[i - window: i + window: 10, 0],
                                               glob_root_pos[i, 0], glob_root_rot[i, 0])
        local_traj_dir = utils.get_rela_dir_to(glob_root_dir[i - window: i + window: 10], glob_root_rot[i, 0])
        local_traj_vel = utils.get_rela_dir_to(glob_root_vel[i - 1 - window: i - 1 + window: 10, 0],
                                               glob_root_rot[i, 0])
        # local_traj_styles = local_style[i - window: i + window: 10]

        xc.append(np.hstack([
            local_traj_pos[:, 0].ravel(), local_traj_pos[:, 2].ravel(),  # Trajectory Pos
            local_traj_dir[:, 0].ravel(), local_traj_dir[:, 2].ravel(),  # Trajectory Dir
            local_traj_vel[:, 0].ravel(), local_traj_vel[:, 2].ravel(),  # Trajectory Vel
            local_style[i - 1].ravel(),  # Trajectory Sty
            local_joint_pos[i - 1].ravel(),  # Joint Pos
            local_joint_vel[i - 1].ravel(),  # Joint Vel
            local_joint_up[i - 1].ravel(),  # Joint Up
            local_joint_forward[i - 1].ravel(),  # joint Forward
        ]))

        next_traj_pos = utils.get_rela_pos_to(glob_root_pos[i + 1: i + 1 + window: 10, 0],
                                              glob_root_pos[i + 1, 0], glob_root_rot[i + 1, 0])
        next_traj_dir = utils.get_rela_dir_to(glob_root_dir[i + 1: i + 1 + window: 10], glob_root_rot[i + 1, 0])
        next_traj_vel = utils.get_rela_dir_to(glob_root_vel[i: i + window: 10, 0], glob_root_rot[i + 1, 0])

        yc.append(np.hstack([
            next_traj_pos[:, 0].ravel(), next_traj_pos[:, 2].ravel(),  # Next Trajectory Pos
            next_traj_dir[:, 0].ravel(), next_traj_dir[:, 2].ravel(),  # Next Trajectory Dir
            next_traj_vel[:, 0].ravel(), next_traj_vel[:, 2].ravel(),  # Next Trajectory Vel
            local_joint_pos[i].ravel(),  # Joint Pos
            local_joint_vel[i].ravel(),  # Joint Vel
            local_joint_up[i].ravel(),  # Joint Up
            local_joint_forward[i].ravel(),  # joint Forward
            local_root_r_vel[i].ravel(),  # Root Rot Vel
        ]))

    return np.array(xc), np.array(yc)


def load_labelfile(file_name):
    f = open(file_name, "r")
    marks = [[], [], []]
    timestamps = []
    start = last = -1
    for line in f:
        p = line.strip().split()
        if len(p) == 4:
            if last == -1:
                start = int(p[0])
            else:
                marks[0] += [int(p[1])] * (int(p[0]) - last)
                marks[1] += [int(p[2])] * (int(p[0]) - last)
                marks[2] += [int(p[3])] * (int(p[0]) - last)
            last = int(p[0])
            timestamps.append(int(p[0]))
    # 根据所需要的label进行过滤
    start_new = timestamps[0]
    end_new = timestamps[0]
    flag = 0
    for i in range(len(timestamps) - 1):
        time_mid = (timestamps[i] + timestamps[i + 1]) // 2 - timestamps[0]
        if marks[0][time_mid] in list(itertools.chain(*config.data.foot_styles_sel)) and \
                marks[1][time_mid] in list(itertools.chain(*config.data.hand_styles_sel)) and \
                marks[2][time_mid] in list(itertools.chain(*config.data.body_style_sel)):
            if flag == 0:
                start_new = timestamps[i]
                flag = 1
            end_new = timestamps[i + 1]
        elif flag:
            break
    marks = [marks[0][start_new - timestamps[0]:end_new - timestamps[0]],
             marks[1][start_new - timestamps[0]:end_new - timestamps[0]],
             marks[2][start_new - timestamps[0]:end_new - timestamps[0]]]
    return marks, start_new, end_new


def process_worker(anim_dir, anim_name, label_pth, prep_dir):
    anim_file = os.path.join(anim_dir, anim_name)
    print('Load animation file', anim_file)

    anim, names, _ = BVH.load(anim_file)
    for i in range(len(names)):
        print(i, names[i])

    phase, start, end = load_labelfile(label_pth)
    if start == end:
        return
    # phase = [phase[0][start:end], phase[1][start:end], phase[2][start:end]]
    anim = anim[start:end]
    # print(phase)
    # phase = phase[1::2]
    phase = [phase[0][1::2], phase[1][1::2], phase[2][1::2]]
    anim = anim[1::2]

    # print(phase)
    print("len(phase): ", len(phase[0]))
    print("len(anim): ", len(anim))
    assert len(phase[0]) == len(anim)

    prep_file = os.path.join(prep_dir, anim_name.replace('.bvh', '.txt'))
    print('Save prepare file', prep_file)
    # print(phase)

    xc, yc = process_data(anim, phase)

    print("len(xc): ", len(xc))
    print("len(yc): ", len(yc))
    if len(xc) > 0:
        print("len(xc[0]): ", len(xc[0]))
        print("len(yc[0]): ", len(yc[0]))

        with open(prep_file, 'w') as pf:
            for i in range(len(yc)):
                training_line = [[round(f, 6) for f in xc[i]], [round(f, 6) for f in yc[i]]]
                json.dump(training_line, pf)
                pf.write('\n')


def main():
    pool = multiprocessing.Pool(1)
    anim_dir = os.path.join(config.data.data_dir, config.data.data_name)
    prep_dir = os.path.join(config.data.data_dir, config.data.data_processed_name)
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    for anim_name in os.listdir(anim_dir):
        if 'bvh' not in anim_name:
            continue
        if config.data.withBall and anim_name.startswith('without_ball'):
            continue
        label_pth = os.path.join(anim_dir, anim_name.replace('.bvh', ".label"))
        if not os.path.exists(label_pth):
            continue

        pool.apply_async(func=process_worker, args=(anim_dir, anim_name, label_pth, prep_dir))

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
