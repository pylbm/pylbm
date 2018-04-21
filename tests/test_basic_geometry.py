from __future__ import print_function
from __future__ import division
import numpy as np
import pylbm.elements as elem

class TestParallelogram(object):
    def setUp(self):
        self.nx, self.ny = 10, 20
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        self.grid = np.meshgrid(x, y, sparse=True, indexing='ij')
        self.p1 = elem.Parallelogram([0, 0], [0, self.ny], [self.nx, 0])

    def test_inside(self):
        in1 = self.p1.point_inside(self.grid)
        assert(in1.all() == True)

    def test_bounds(self):
        np.testing.assert_array_equal(self.p1.get_bounds(), [[0, 0], [self.nx, self.ny]])

class TestTriangle(object):
    def setUp(self):
        nx, ny = 11, 11
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.grid = np.meshgrid(x, y, sparse=True, indexing='ij')

        self.t1 = elem.Triangle([0, 0], [1, 0], [0, 1])
        self.t2 = elem.Triangle([0, 0], [nx, 0], [0, nx])
        self.t3 = elem.Triangle([nx - 1, 1], [-nx, nx], [0, nx - 1])

    def test_inside(self):
        in1 = self.t2.point_inside(self.grid)
        in2 = self.t3.point_inside(self.grid)
        assert(np.logical_or(in1, in2).all() == True)

    def test_bounds(self):
        np.testing.assert_array_equal(self.t1.get_bounds(), [[0, 0], [1, 1]])

###
# ajouter un test qui verifie que si on met les vecteurs a, b ou b, a
# on obtient les memes resultats pour les fonctions point_inside et distance
#
if __name__ == '__main__':
    x = np.linspace(-0.95, .95, 100)
    y = np.linspace(-0.95, .95, 100)

    dx = 2*.95/100
    #print x, y
    t1 = elem.Triangle([0, 0], [1, 0], [0, .5])
    t2 = elem.Triangle([0.5, 0.5], [0, .3], [.2, 0])

    q1 = elem.Parallelogram([-0.5, -0.5], [1, 0], [0, 1])

    c = elem.Circle([.5, .5], .2)

    b, e = t1.get_bounds()

    nx = [(b[0]-x[0])/.1, (e[0]-x[0])/.1]
    ny = [(b[1]-y[0])/.1, (e[1]-y[0])/.1]

    #print b, e, nx, ny

    #gridx = x[np.newaxis, nx[0]:nx[1]+1]
    #gridy = y[ny[0]:ny[1]+1, np.newaxis]

    gridx = x[np.newaxis, :]
    gridy = y[:, np.newaxis]

    # print t1.point_inside(gridx, gridy)
    # print
    # print t2.point_inside(gridx, gridy)


    #t1.distance(gridx, gridy, (0, .001))
    #t1.distance(gridx, gridy, (0, -.001))

    print(q1.point_inside(gridx, gridy))
    # import time
    # t= time.time()
    # d, b = q1.distance(gridx, gridy, (0., -0.1))
    # print time.time() - t
    # print d, b
    # print help(c.point_inside)
    # import matplotlib.pylab as plt
    # t = t2.point_inside(gridx, gridy)
    # t -= c.point_inside(gridx, gridy)
    # plt.imshow(t, origin='lower')
    # plt.show()
    import time
    t = time.time()

    d, b = q1.distance(gridx, gridy, (0, dx))
    print(time.time() - t)
    print(d)
    import matplotlib.pylab as plt
    plt.imshow(d, origin='lower')
    plt.show()
