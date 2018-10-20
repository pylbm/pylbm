import numpy as np
from nose import tools
import importlib
import sys
import os
import pylbm

path = os.path.dirname(__file__) + '/data/scheme'
sys.path.insert(0, path)

# list python files in path
scheme_to_test = []
for f in os.listdir(path):
    if f[-3:] == '.py':
        scheme_to_test.append(f.strip('.py'))

def test_scheme():
    for stest in scheme_to_test:
        module = importlib.import_module(stest)
        seq = []
        for e1 in module.eq:
            seq.append([list(map(str, e2)) for e2 in e1])

        for pa in module.param:
            for p in module.poly:
                for ie, e in enumerate(module.eq + seq):
                    schemes = []
                    for js, s in enumerate(e):
                        schemes.append({
                        'velocities':module.velocity,
                        'polynomials':p,
                        'relaxation_parameters':module.relax,
                        'equilibrium':s,
                        'conserved_moments': module.consm[ie%2][js]
                        })

                    dico = {'dim':2,
                            'scheme_velocity': 1.,
                            'schemes': schemes,
                            'parameters': pa,
                            }
                    yield construct_scheme, module, dico

def construct_scheme(module, dico):
    s = pylbm.Scheme(dico)
    for m1, m2 in zip(s.M, module.M):
        tools.ok_(np.all(m1==m2))
    tools.eq_(s._EQ, module.EQ_result)

test_scheme()
