from enum import Enum

from data_gen.motion import BVH

BVHParents = [-1, 0, 1, 2, 3, 4, 5, 2, 7, 8, 9, 2, 11, 0, 13, 14, 15, 0, 17, 18, 19]


class BVHJoints(Enum):
    Pelvis = 0
    Spine = 1
    Spine1 = 2
    LClavicle = 3
    LUpperArm = 4
    LForearm = 5
    LHand = 6
    RClavicle = 7
    RUpperArm = 8
    RForearm = 9
    RHand = 10
    Neck = 11
    Head = 12
    LThigh = 13
    LCalf = 14
    LFoot = 15
    LToe0 = 16
    RThigh = 17
    RCalf = 18
    RFoot = 19
    RToe0 = 20


if __name__ == '__main__':
    def get_parents():
        animation, _, _ = BVH.load('../data_gen/data_new/NBA_small_test/001-1_Take_001.bvh')
        parents = animation.parents

        print(*parents, sep=', ')


    get_parents()
