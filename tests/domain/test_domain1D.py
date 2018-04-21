from __future__ import print_function
from six.moves import range
import numpy as np
import copy
import pylbm

class test_domain1D(object):
    dom1d = {'box':{'x': [0, 1], 'label': 0},
             'space_step':0.25,
             'schemes': [{'velocities':list(range(3))}],
             }
    valin = 999
    valout = -1

    def test_simple_domain_with_labels(self):
        dom1d = copy.deepcopy(self.dom1d)
        dom1d['box'] = {'x': [0, 1], 'label': [0, 1]}
        dom = pylbm.Domain(dom1d)

        assert(dom.shape_halo == [6])
        assert(dom.shape_in == [4])
        assert(dom.dx == .25)
        assert(np.all(dom.geom.bounds == [[0., 1.]]))
        assert(np.all(dom.x_halo == np.linspace(-.125, 1.125, 6)))

    def test_with_given_geometry_and_stencil(self):
        geom = pylbm.Geometry({'box': {'x': [0, 1]}})
        sten = pylbm.Stencil({'dim': 1, 'schemes': [{'velocities':list(range(3))}]})
        dom = pylbm.Domain(geometry = geom, stencil=sten, space_step=.25)

    def test_domain_with_one_scheme(self):
        dom = pylbm.Domain(self.dom1d)

        desired_in_or_out = self.valin*np.ones(6)
        desired_in_or_out[[0, -1]] = self.valout
        assert(np.all(dom.in_or_out == desired_in_or_out))

        desired_distance = self.valin*np.ones((3, 6))
        desired_distance[[(1, 2), (-2, 1)]] = .5
        print(dom.distance)
        assert(np.all(dom.distance == desired_distance))

        desired_flag = self.valin*np.ones((3, 6), dtype=int)
        desired_flag[[(1, 2), (-2, 1)]] = 0
        assert(np.all(dom.flag == desired_flag))

    def test_domain_with_two_schemes_without_label(self):
        dom1d = copy.deepcopy(self.dom1d)
        dom1d['schemes'].append({'velocities':list(range(5))})

        dom = pylbm.Domain(dom1d)

        desired_in_or_out = self.valin*np.ones(8)
        desired_in_or_out[[0, 1, -2, -1]] = self.valout
        assert(np.all(dom.in_or_out == desired_in_or_out))

        desired_distance = self.valin*np.ones((5, 8))
        desired_distance[[(1, 2), (-3, 2)]] = .5
        desired_distance[[(3, 4), (-3, 2)]] = .25
        desired_distance[[(3, 4), (-4, 3)]] = .75
        assert(np.all(dom.distance == desired_distance))

        desired_flag = self.valin*np.ones((5, 8), dtype=int)
        ind0 = (1, 2) + (3, 4)*2
        ind1 = (-3, 2)*2 + (-4, 3)
        desired_flag[[ind0, ind1]] = 0
        assert(np.all(dom.flag == desired_flag))

    def test_domain_with_two_schemes_with_label(self):
        lleft, lright = 1, 2
        dom1d = copy.deepcopy(self.dom1d)
        dom1d['box'] = {'x': [0, 1], 'label': [lleft, lright]}
        dom1d['schemes'].append({'velocities':list(range(5))})

        dom = pylbm.Domain(dom1d)

        desired_in_or_out = self.valin*np.ones(8)
        desired_in_or_out[[0, 1, -2, -1]] = self.valout
        assert(np.all(dom.in_or_out == desired_in_or_out))

        desired_distance = self.valin*np.ones((5, 8))
        desired_distance[[(1, 2), (-3, 2)]] = .5
        desired_distance[[(3, 4), (-3, 2)]] = .25
        desired_distance[[(3, 4), (-4, 3)]] = .75
        assert(np.all(dom.distance == desired_distance))

        desired_flag = self.valin*np.ones((5, 8), dtype=int)
        ind0_left = (2,) + (4,)*2
        ind1_left = (2,)*2 + (3,)
        ind0_right = (1,) + (3,)*2
        ind1_right = (-3,)*2 + (-4,)
        desired_flag[[ind0_left, ind1_left]] = lleft
        desired_flag[[ind0_right, ind1_right]] = lright
        assert(np.all(dom.flag == desired_flag))
