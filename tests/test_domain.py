from pyLBM.elements import *
from pyLBM.geometry import Geometry
from pyLBM.domain import Domain
import pyLBM.stencil as stencil
import pickle
import pyLBM

SAVE_GOOD_RES = False

class test_domain1D:
    dim = 1
    solid = 0
    fluid = 1
    output = "/tmp/testdom"
    path = pyLBM.__path__[0]

    def setup(self):
        dgeom = {'dim': 1,
                 'box':{'x': [0, 1], 'label': [0, 1]},
                 }

        self.dico = {'dim': 1,
                'geometry': dgeom,
                'space_step':0.1
                }
        self.geom = Geometry(self.dico)

    def save(self, dom, n):
        f = open(self.path + "/../tests/data/domain/case1D_%d"%n, 'w')
        pickle.dump(dom, f)
        f.close()

    def compare(self, dom, n): 
        if SAVE_GOOD_RES:
            self.save(dom, n)
        f = open(self.path + "/../tests/data/domain/case1D_%d"%n, 'r')
        right_dom = pickle.load(f)
        f.close()

        assert((dom.distance == right_dom.distance).all())
        assert((dom.in_or_out == right_dom.in_or_out).all())
        assert((dom.flag == right_dom.flag).all())
        
    def test_segment_1(self):
        n = 0
        self.dico['number_of_schemes'] = 1
        self.dico[0] = {'velocities':range(3)}
        dom = Domain(self.geom, self.dico)

        self.compare(dom, n)

    def test_segment_2(self):
        n = 1
        self.dico['number_of_schemes'] = 2
        self.dico[0] = {'velocities':range(5)}
        self.dico[1] = {'velocities':range(3)}
        dom = Domain(self.geom, self.dico)

        self.compare(dom, n)

    def test_segment_3(self):
        n = 2
        self.dico['number_of_schemes'] = 1
        self.dico[0] = {'velocities':range(11)}
        dom = Domain(self.geom, self.dico)

        self.compare(dom, n)
        
class test_domain2D:
    dim = 2
    solid = 0
    fluid = 1
    output = "/tmp/testdom"
    path = pyLBM.__path__[0]

    def save(self, dom, n):
        f = open(self.path + "/../tests/data/domain/case2D_%d"%n, 'w')
        pickle.dump(dom, f)
        f.close()

    def compare(self, dom, n): 
        if SAVE_GOOD_RES:
            self.save(dom, n)
        f = open(self.path + "/../tests/data/domain/case2D_%d"%n, 'r')
        right_dom = pickle.load(f)
        f.close()

        assert((dom.distance == right_dom.distance).all())
        assert((dom.in_or_out == right_dom.in_or_out).all())
        assert((dom.flag == right_dom.flag).all())
        
    def test_rectangle(self):
        n = 0
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 2], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.2,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_rectangle_with_square_obs(self):
        n = 1
        dgeom = {'dim': self.dim,
                 'box':{'x': [-1, 1], 'y': [-1, 1], 'label': [0, 0, 0, 0]},
                 }
        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.15,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((-0.5,-0.5), (1.,0.), (0.,1.)), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)
        
    def test_rectangle_with_parallelogram_obs(self):
        n = 2
        dgeom = {'dim': self.dim,
                 'box':{'x': [-1, 1], 'y': [-1, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.1,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((-0.5,0.), (0.5,0.5), (0.5,-0.5)), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_down_step(self):
        n = 3
        dgeom = {'dim': self.dim,
                 'box':{'x': [-1, 5], 'y': [-1, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.5,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((-1,-1), (0,1), (1,0)), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_circular_cavity(self):
        n = 4
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 2], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.05,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((0,0), (2,0), (0,1)), 0, self.solid)
        geom.add_elem(Parallelogram((0,0.4), (2,0), (0,0.2)), 0, self.fluid)
        geom.add_elem(Circle((1,0.5), 0.5), 0, self.fluid)
        geom.add_elem(Circle((1,0.5), 0.2), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_rectangle_with_triangular_obs(self):
        n = 5
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 1], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.05,
                'number_of_schemes':2,
                0:{'velocities':range(9)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Triangle((0.2,0.5), (0.3,0.1), (0.4,-0.1)), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_triangular_domain(self):
        n = 6
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 1], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.05,
                'number_of_schemes':1,
                0:{'velocities':range(5)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((0,0), (0,1), (1,0)), 0, self.solid)
        geom.add_elem(Triangle((.2,.2), (0.5,0.), (0.,.5)), 0, self.fluid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_polygonal_domain(self):
        n = 7
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 1], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.05,
                'number_of_schemes':1,
                0:{'velocities': range(9)}
                }
        geom = Geometry(dico)
        geom.add_elem(Parallelogram((0.,0.), (1.,0.), (0.,1)), 0, self.solid)
        geom.add_elem(Parallelogram((0.,.25), (1.,0.), (0.,.25)), 0 , self.fluid)
        geom.add_elem(Triangle((0.25,0.5), (.5,0.), (.25,.25)), 0, self.fluid)

        dom = Domain(geom, dico)
        self.compare(dom, n)

    def test_rectangle_circular_obs(self):
        n = 8
        dgeom = {'dim': self.dim,
                 'box':{'x': [0, 2], 'y': [0, 1], 'label': [0, 0, 0, 0]},
                 }

        dico = {'dim': self.dim,
                'geometry': dgeom,
                'space_step':0.05,
                'number_of_schemes':2,
                0:{'velocities':range(1,5)},
                1:{'velocities':range(33)}
                }
        geom = Geometry(dico)
        geom.add_elem(Circle((1,0.5), 0.2), 0, self.solid)

        dom = Domain(geom, dico)
        self.compare(dom, n)
