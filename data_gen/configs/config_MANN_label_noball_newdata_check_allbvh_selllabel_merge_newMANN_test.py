import os
import os.path as osp
from easydict import EasyDict

config = EasyDict()
config.model_name = 'MANN_foot_hand_sep'
config.version = 1
config.exp_description = "baseline of MANN, NBA motion control, label, noball, new data, " \
                         "all label, version%d" % config.version
config.gpus = "3"
config.store_dir = "./result"
config.store_name = "MANN_NBA_motion_label_noball_all_label_allbvh_merge_newMANN_newdata_check_test_v%d" % config.version

# add data config
config.data = EasyDict()
config.data.data_dir = "../dataset"
config.data.data_name = 'NBA_small_test'
config.data.data_processed_name = 'processed'
# config.data.data_processed_name = 'NBA_small_test_processed'
config.data.FRAME_RATE = 60
config.data.withBall = False
# Extract Forward Direction
config.data.sdr_l = 4
config.data.sdr_r = 8
config.data.hip_l = 13
config.data.hip_r = 17

config.data.window = 60

config.data.num_joints = 21
# foot action: 0(idle), 1(idle-move), 2(move-forward), 3(move-backward), 4(turn-left)
# 5(turn-right), 6(jump)
config.data.foot_styles_num_ori = 7
config.data.foot_styles_sel = [[0], [1], [2], [3], [4], [5], [6]]
# hand action: 0(left-hand), 1(right-hand), 2(crossover), 3(pass-ball), 4(catch-ball)
# 5(shot), 6(layup), 7(defense), 8(others)
config.data.hand_styles_num_ori = 9
config.data.hand_styles_sel = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
# body action: 0(bent), 1(upright)
config.data.body_style_num_ori = 2
config.data.body_style_sel = [[0], [1]]

config.data.num_styles = len(config.data.hand_styles_sel) + len(config.data.foot_styles_sel) + len(
    config.data.body_style_sel)

config.data.TrajectorySize = 12  # frames
config.data.TrajectoryInputSize = 6

config.data.withlabel = True

# add train config
config.train = EasyDict()
config.train.num_experts_foot = 8
config.train.num_experts_hand = 8
config.train.num_experts = 8
config.train.input_size = 342
config.train.output_size = 289
# config.train.start_bone_index = [14, 15, 16, 17, 18, 20, 6, 10]
config.train.start_bone_index_hand = [4, 5, 6, 8, 9, 10]
config.train.start_bone_index_foot = [14, 15, 16, 18, 19, 20]
config.train.hidden_size = 512
config.train.hidden_size_gt_foot = 64
config.train.hidden_size_gt_hand = 64
config.train.batch_size = 2 ** 7
config.train.epoch = 200
config.train.Te = 10
config.train.Tmult = 2
config.train.learning_rate_ini = 0.0001
config.train.weight_decay_ini = 0.0025
config.train.keep_prob_ini = 0.7
config.train.shuffle = False

config.train.resume = False

# add anim config
config.anim = EasyDict()
config.anim.TOTAL_POINTS = 111
config.anim.PAST_POINTS = 60
config.anim.FUTURE_POINTS = 50
config.anim.ROOT_POINT_INDEX = 60
config.anim.FRAME_RATE = 60
config.anim.TOTAL_BONES = 21
config.anim.POINT_DENSITY = 10
config.anim.anim_file_name = '001-1_Take_001.bvh'
config.anim.anim_dir = './data_new'
config.anim.trajectory_type = 'go_stop'
config.anim.gen_file_name = 'round_gen_label_newdata_%s.bvh' % config.anim.trajectory_type
