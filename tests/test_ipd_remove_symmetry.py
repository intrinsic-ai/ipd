import unittest
from intrinsic_ipd.reader import IPDReader
from intrinsic_ipd.constants import IPDCamera, IPDLightCondition
import numpy as np


class TestRemoveSymmetry(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = IPDReader("./datasets", "dataset_darkbg_0", IPDCamera.PHOTONEO, lighting=IPDLightCondition.ROOM, download=False)

    def test_indiv_vs_batch(self):
        reader = self.reader
        o2c = reader.o2c
        part1 = reader.objects[0][0]
        no_sym_batch = reader.remove_symmetry(part1, o2c.sel(part=part1))
        for scene in reader.scenes:
            for part2, instance in reader.objects:
                if part2 == part1:
                    pose = o2c.sel(object=(part2, instance), scene=scene)
                    poses1 = reader.remove_symmetry(part2, pose)
                    poses2 = no_sym_batch.sel(scene=scene, instance=instance)
                    np.testing.assert_array_almost_equal(poses1, poses2)
    
    def test_repeated_indiv(self):
        reader = self.reader
        o2c = reader.o2c
        for scene in reader.scenes:
            part, instance = reader.objects[0]
            pose = o2c.sel(object=(part, instance), scene=scene)
            poses1 = reader.remove_symmetry(part, pose)
            poses2 = reader.remove_symmetry(part, reader.remove_symmetry(part, pose))
            np.testing.assert_array_almost_equal(poses1, poses2) 

    def test_repeated_batch(self):
        reader = self.reader
        o2c = reader.o2c
        for part, _ in reader.objects:
            poses = o2c.sel(part=part)
            poses1 = reader.remove_symmetry(part, poses)
            poses2 = reader.remove_symmetry(part, reader.remove_symmetry(part, poses))
            np.testing.assert_array_almost_equal(poses1, poses2)


if __name__ == '__main__':
    unittest.main()