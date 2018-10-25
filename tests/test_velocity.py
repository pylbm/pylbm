
import numpy as np
from pylbm.stencil import Velocity
import pytest
import random

def test_velocity_argument():
    with pytest.raises(Exception):
        Velocity()

@pytest.mark.parametrize('dim, axis',
                         [(1, [None, 0]),
                          (2, [None, 0, 1]),
                          (3, [None, 0, 1, 2])])
def test_symmetric(dim, axis):
    for i in np.random.randint(1000, size=100):
        v = Velocity(dim=dim, num=i)
        for a in axis:
            vs = v.get_symmetric(axis=a).get_symmetric(axis=a)
            assert(vs.v == v.v and vs.num == v.num)

@pytest.mark.parametrize('dim', [1, 2, 3])
def test_symmetric_shuffle(dim):
    for i in np.random.randint(1000, size=100):
        v = Velocity(dim=dim, num=i)
        axis = np.random.randint(dim, size=5)
        vs = v
        for a in axis:
            vs = vs.get_symmetric(axis=a)
        for a in axis[::-1]:
            vs = vs.get_symmetric(axis=a)
        assert(vs.v == v.v and vs.num == v.num)
