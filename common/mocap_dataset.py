import numpy as np

from common.bvh_joints import BVHJoints


class MocapDataset:
    def __init__(self, path, fps=30, skeleton=BVHJoints):
        """

        :param path: npy path, shape of (frames, 3(pos, up, forward), 21(#joints), 3(xzy)).
        :param fps: fps of npy data, extracted from BVH files. 30 (down sample from 60) in default
        :param skeleton: BVH joints
        """
        self.data = np.load(path, allow_pickle=True)
        self.fps = fps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        this_data = self.data[idx]
        sample = {'position': this_data[0], 'rotation': this_data[1:]}

        return sample
