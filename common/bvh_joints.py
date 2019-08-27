from enum import Enum

BVHParents = [-1, 0, 1, 2, 3, 4, 5, 2, 7, 8, 9, 2, 11, 0, 13, 14, 15, 0, 17, 18, 19]

"""Adjacent list directed graph representation for BVH joints"""
BVH_adj = [{1, 13, 17}, {2}, {11, 3, 7}, {4}, {5}, {6}, set(), {8}, {9}, {10}, set(), {12}, set(), {14}, {15}, {16}, set(), {18}, {19}, {20}, set()]


# BVH_adj = [set() for i in range(len(BVHParents))]
# for i, s in enumerate(BVH_adj):
#     s.update([index for index, x in enumerate(BVHParents) if x == i])


class BVHJoints(Enum):
    Pelvis = 0  # 骨盆
    Spine = 1  # 脊柱
    Spine1 = 2  # 脊柱1
    LClavicle = 3  # 左锁骨
    LUpperArm = 4  # 左上臂
    LForearm = 5  # 左前臂
    LHand = 6  # 左手
    RClavicle = 7  # 右锁骨
    RUpperArm = 8  # 右上臂
    RForearm = 9  # 右前臂
    RHand = 10  # 右手
    Neck = 11  # 颈
    Head = 12  # 头
    LThigh = 13  # 左大腿
    LCalf = 14  # 左小腿
    LFoot = 15  # 左脚
    LToe0 = 16  # 左脚趾
    RThigh = 17  # 右大腿
    RCalf = 18  # 右小腿
    RFoot = 19  # 右脚
    RToe0 = 20  # 右脚趾


if __name__ == '__main__':
    print(BVH_adj)
    # def get_parents():
    #     animation, _, _ = BVH.load('../data_gen/data_new/NBA_small_test/001-1_Take_001.bvh')
    #     parents = animation.parents
    #
    #     print(*parents, sep=', ')
    #
    #
    # get_parents()
