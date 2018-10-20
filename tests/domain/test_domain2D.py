from six.moves import range
import numpy as np
import copy
import os
import pylbm

create_ref = False

def check_from_file(dom, fname):
    path = os.path.join(os.path.dirname(__file__), 'data', fname)
    if not create_ref:
        dom_ref = np.load(path)
        assert(np.allclose(dom.distance, dom_ref['distance'], 1e-16))
        assert(np.all(dom.in_or_out == dom_ref['in_or_out']))
        assert(np.all(dom.flag == dom_ref['flag']))
    else:
        np.savez(path,
                 in_or_out = dom.in_or_out,
                 distance = dom.distance,
                 flag = dom.flag)

class test_domain2D(object):
    dom2d = {'box':{'x': [0, 1], 'y': [0, 2], 'label': 0},
             'space_step':0.25,
             'schemes': [{'velocities':list(range(5))}],
             }

    valin = 999
    valout = -1

    def test_simple_domain_with_labels(self):
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['box']['label'] = [0, 1, 2, 0]
        dom = pylbm.Domain(dom2d)

        # assert(dom.Ng == [4, 8])
        # assert(dom.N == [4, 8])
        assert(dom.dx == .25)
        #assert(np.all(dom.bounds == [[0., 1.], [0., 2.]]))
        assert(np.all(dom.x_halo == [np.linspace(-.125, 1.125, 6)]))
        assert(np.all(dom.y_halo == [np.linspace(-.125, 2.125, 10)]))

    def test_domain_with_one_scheme(self):
        fname = 'simple_domain.npz'
        dom = pylbm.Domain(self.dom2d)

        check_from_file(dom, fname)

    def test_domain_with_rectangle(self):
        fname = 'rectangle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Parallelogram([0.23, 0.73], [0.5, 0], [0., .5], label=10)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)

    def test_domain_with_fluid_rectangle(self):
        fname = 'fluid_rectangle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Parallelogram([0., 0.], [1., 0], [0., 2.], label=20),
                             pylbm.Parallelogram([0.23, 0.73], [0.5, 0], [0., .5], label=10, isfluid=True)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)

    def test_domain_with_circle(self):
        fname = 'circle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Circle([0.5, 1.], .5, label=10)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)

    def test_domain_with_fluid_circle(self):
        fname = 'fluid_circle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Parallelogram([0., 0.], [1., 0], [0., 2.], label=20),
                             pylbm.Circle([0.5, 1.], .5, label=10, isfluid=True)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)

    def test_domain_with_triangle(self):
        fname = 'triangle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Triangle([0.23, 0.73], [0.5, 0], [0., .5], label=10)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)

    def test_domain_with_fluid_triangle(self):
        fname = 'fluid_triangle.npz'
        dom2d = copy.deepcopy(self.dom2d)
        dom2d['elements'] = [pylbm.Parallelogram([0., 0.], [1., 0], [0., 2.], label=20),
                             pylbm.Triangle([0.23, 0.73], [0.5, 0], [0., .5], label=10, isfluid=True)]
        dom = pylbm.Domain(dom2d)

        check_from_file(dom, fname)
