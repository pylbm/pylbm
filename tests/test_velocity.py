
from six.moves import range
from pylbm.stencil import Velocity
import nose.tools as tools
import random

@tools.raises(Exception)
def test_velocity_argument():
    Velocity()

class test_velocity_1D(object):
    num = [0, 1, 2, 3, 4, 5, 6, 7]
    vx = [0, 1, -1, 2, -2, 3, -3, 4]

    def test_with_num(self):
        for n, vx in zip(self.num, self.vx):
            v = Velocity(dim=1, num=n)
            assert(v.vx == vx)

    def test_with_vx(self):
        for n, vx in zip(self.num, self.vx):
            v = Velocity(vx=vx)
            assert(v.num == n and v.dim == 1)

    def test_symmetry1(self):
        v = Velocity(dim=1, num=random.randint(0,1000))
        vs = v.get_symmetric().get_symmetric()
        assert(vs.vx == v.vx and vs.num == v.num)

    def test_symmetry2(self):
        v = Velocity(dim=1, num=random.randint(0,1000))
        vs = v.get_symmetric(axis=0).get_symmetric(axis=0)
        assert(vs.vx == v.vx and vs.num == v.num)

    @tools.raises(ValueError)
    def test_symmetry3(self):
        v = Velocity(dim=1, num=random.randint(0,1000))
        vs = v.get_symmetric(axis=1).get_symmetric(axis=1)

class test_velocity_2D(object):
    num = list(range(9))
    snum = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    vx = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    vy = [0, 0, 1,  0, -1, 1,  1, -1, -1]
    nrep = 10

    def test_with_num(self):
        for n, vx, vy in zip(self.num, self.vx, self.vy):
            v = Velocity(dim=2, num=n)
            assert(v.vx == vx and v.vy == vy)

    def test_with_v(self):
        for n, vx, vy in zip(self.num, self.vx, self.vy):
            v = Velocity(vx=vx, vy=vy)
            assert(v.num == n and v.dim == 2)

    def test_symmetry1(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=2, num=n)
            vs = v.get_symmetric().get_symmetric()
            assert(vs.vx == v.vx and vs.vy == v.vy and vs.num == v.num)

    def test_symmetry2(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=2, num=n)
            vs = v.get_symmetric(axis=0).get_symmetric(axis=0)
            assert(vs.vx == v.vx and vs.vy == v.vy and vs.num == v.num)

    def test_symmetry3(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=2, num=n)
            vs = v.get_symmetric(axis=0).get_symmetric(axis=1)
            vs = vs.get_symmetric(axis=0).get_symmetric(axis=1)
            assert(vs.vx == v.vx and vs.vy == v.vy and vs.num == v.num)

    def test_symmetry4(self):
        for n, sn in zip(self.num, self.snum):
            v = Velocity(dim=2, num=n)
            vs = v.get_symmetric()
            assert(vs.num == sn)

    @tools.raises(ValueError)
    def test_symmetry5(self):
        v = Velocity(dim=2, num=0)
        vs = v.get_symmetric(axis=2).get_symmetric(axis=2)

class test_velocity_3D(object):
    num = list(range(27))
    nrep = 10
    vx = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1]
    vy = [0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1]
    vz = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1]

    def test_with_num(self):
        for n, vx, vy, vz in zip(self.num, self.vx, self.vy, self.vz):
            v = Velocity(dim=3, num=n)
            assert(v.vx == vx and v.vy == vy and v.vz == vz)

    def test_with_v(self):
        for n, vx, vy, vz in zip(self.num, self.vx, self.vy, self.vz):
            v = Velocity(vx=vx, vy=vy, vz=vz)
            assert(v.num == n and v.dim == 3)

    def test_symmetry1(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=3, num=n)
            vs = v.get_symmetric().get_symmetric()
            assert(vs.vx == v.vx and vs.vy == v.vy
                   and vs.vz == v.vz and vs.num == v.num)

    def test_symmetry2(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=3, num=n)
            vs = v.get_symmetric(axis=0).get_symmetric(axis=0)
            assert(vs.vx == v.vx and vs.vy == v.vy and
                   vs.vz == v.vz and vs.num == v.num)

    def test_symmetry3(self):
        for i in range(self.nrep):
            n = random.randint(0,1000)
            v = Velocity(dim=3, num=n)
            vs = v.get_symmetric(axis=0).get_symmetric(axis=1).get_symmetric(axis=2)
            vs = vs.get_symmetric(axis=0).get_symmetric(axis=1).get_symmetric(axis=2)
            assert(vs.vx == v.vx and vs.vy == v.vy and
                   vs.vz == v.vz and vs.num == v.num)
