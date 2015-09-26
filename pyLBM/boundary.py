# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from .logs import setLogger
from .storage import Array

class Boundary_Velocity:
    """
    Indices and distances for the label and the velocity ksym
    """
    def __init__(self, domain, label, ksym):
        self.label = label
        # on cherche les points de l'exterieur qui ont une vitesse qui rentre (indice ksym)
        # sur un bord labelise par label
        # on parcourt toutes les vitesses et on determine les points interieurs qui ont la vitesse
        # symmetrique (indice k) qui sort
        # puis on ecrit dans une liste reprenant l'ordre des vitesses du schema
        # - les indices des points exterieurs correspondants
        # - les distances associees
        self.v = domain.stencil.unique_velocities[ksym]
        v = self.v.get_symmetric()
        num = domain.stencil.unum2index[v.num]

        ind = np.where(domain.flag[num] == self.label)
        self.indices = np.array(ind)
        if self.indices.size != 0:
            self.indices += np.asarray(v.v)[:, np.newaxis]
        self.distance = np.array(domain.distance[(num,) + ind])

class Boundary_method(object):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        self.istore = istore
        self.feq = np.zeros((stencil.nv_ptr[-1], istore.shape[1]))
        self.rhs = np.zeros(istore.shape[1])
        self.ilabel = ilabel
        self.distance = distance
        self.stencil = stencil
        self.iload = []
        self.value_bc = {}
        for k in np.unique(self.ilabel):
            self.value_bc[k] = value_bc[k]
        self.func = None

    def add_iload(self, iloadfunc, iloadargs=()):
        iload = iloadfunc(self.istore, self.stencil, *iloadargs)
        self.iload.append(iload)

    def prepare_rhs(self, simulation):
        nv = simulation._m.nv
        sorder = simulation._m.sorder
        nspace = [1]*(len(sorder)-1)
        v = self.stencil.get_all_velocities()

        for key, value in self.value_bc.iteritems():
            if value is not None:
                indices = np.where(self.ilabel == key)
                # TODO: check the index in sorder to be the most contiguous
                nspace[0] = indices[0].size
                k = self.istore[0, indices]
                x = simulation.domain.x[0][self.istore[1, indices]]
                y = simulation.domain.x[1][self.istore[2, indices]]

                s = 1 - self.distance[indices]

                x += s*v[k, 0]*simulation.domain.dx
                y += s*v[k, 1]*simulation.domain.dx
                m = Array(nv, nspace , 0, sorder)
                f = Array(nv, nspace , 0, sorder)

                value(f, m, x.T, y.T)
                simulation.scheme.equilibrium(m)
                simulation.scheme.m2f(m, f)

                self.feq[:, indices[0]] = f.swaparray.reshape((nv, indices[0].size))

class Bounce_back(Boundary_method):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        super(Bounce_back, self).__init__(istore, ilabel, distance, stencil, value_bc)

    def set_iload(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def set_rhs(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    def update(self, f):
        f[tuple(self.istore)] = f[tuple(self.iload[0])] + self.rhs

class Bouzidi_bounce_back(Boundary_method):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        super(Bouzidi_bounce_back, self).__init__(istore, ilabel, distance, stencil, value_bc)
        self.s = np.empty(self.istore.shape[1])

    def set_iload(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()

        iload1 = np.zeros(self.istore.shape, dtype=np.int)
        iload2 = np.zeros(self.istore.shape, dtype=np.int)

        mask = self.distance < .5
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = ksym[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + 2*v[k[mask]].T
        self.s[mask] = 2.*self.distance[mask]

        mask = np.logical_not(mask)
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = k[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        self.s[mask] = .5/self.distance[mask]

        self.iload.append(iload1)
        self.iload.append(iload2)

    def set_rhs(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    def update(self, f):
        f[tuple(self.istore)] = self.s*f[tuple(self.iload[0])] + (1 - self.s)*f[tuple(self.iload[1])] + self.rhs

class Anti_bounce_back(Boundary_method):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        super(Anti_bounce_back, self).__init__(istore, ilabel, distance, stencil, value_bc)

    def set_rhs(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    def set_iload(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def update(self, f):
        f[tuple(self.istore)] = -f[tuple(self.iload[0])] + self.rhs

class Bouzidi_anti_bounce_back(Boundary_method):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        super(Bouzidi_anti_bounce_back, self).__init__(istore, ilabel, distance, stencil, value_bc)
        self.s = np.empty(self.istore.shape[1])

    def set_iload(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()

        iload1 = np.zeros(self.istore.shape, dtype=np.int)
        iload2 = np.zeros(self.istore.shape, dtype=np.int)

        mask = self.distance < .5
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = ksym[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + 2*v[k[mask]].T
        self.s[mask] = 2.*self.distance[mask]

        mask = np.logical_not(mask)
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = k[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        self.s[mask] = .5/self.distance[mask]

        self.iload.append(iload1)
        self.iload.append(iload2)

    def set_rhs(self):
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    def update(self, f):
        f[tuple(self.istore)] = -self.s*f[tuple(self.iload[0])] + (self.s - 1)*f[tuple(self.iload[1])] + self.rhs

class Neumann_vertical(Boundary_method):
    def __init__(self, istore, ilabel, distance, stencil, value_bc):
        super(Neumann_vertical, self).__init__(istore, ilabel, distance, stencil, value_bc)

    def set_rhs(self):
        pass

    def set_iload(self):
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[0] += v[k].T[0]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    def update(self, f):
        f[tuple(self.istore)] = f[tuple(self.iload[0])]

class Boundary:
    def __init__(self, domain, dico):
        self.log = setLogger(__name__)
        self.domain = domain

        # build the list of indices for each unique velocity and for each label
        self.bv = {}
        for label in self.domain.geom.list_of_labels():
            dummy_bv = []
            for k in xrange(self.domain.stencil.unvtot):
                dummy_bv.append(Boundary_Velocity(self.domain, label, k))
            self.bv[label] = dummy_bv

        # build the list of boundary informations for each stencil and each label

        dico_bound = dico.get('boundary_conditions',{})
        stencil = self.domain.stencil

        istore = {}
        ilabel = {}
        distance = {}
        value_bc = {}

        for label in self.domain.geom.list_of_labels():
            if label == -1: # periodic conditions
                pass
            elif label == -2: # interface conditions
                pass
            else: # non periodic conditions
                value_bc[label] = dico_bound[label].get('value', None)
                methods = dico_bound[label]['method']
                for k, v in methods.iteritems():
                    for inumk, numk in enumerate(stencil.num[k]):
                        if self.bv[label][stencil.unum2index[numk]].indices.size != 0:
                            indices = self.bv[label][stencil.unum2index[numk]].indices
                            distance_tmp = self.bv[label][stencil.unum2index[numk]].distance
                            velocity = (inumk + stencil.nv_ptr[k])*np.ones(indices.shape[1], dtype=np.int32)[np.newaxis, :]
                            ilabel_tmp = label*np.ones(indices.shape[1], dtype=np.int32)
                            istore_tmp = np.concatenate([velocity, indices])
                            if istore.get(v, None) is None:
                                istore[v] = istore_tmp.copy()
                                ilabel[v] = ilabel_tmp.copy()
                                distance[v] = distance_tmp.copy()
                            else:
                                istore[v] = np.concatenate([istore[v], istore_tmp], axis=1)
                                ilabel[v] = np.concatenate([ilabel[v], ilabel_tmp])
                                distance[v] = np.concatenate([distance[v], distance_tmp])
        self.methods = []
        for k in istore.keys():
            self.methods.append(k(istore[k], ilabel[k], distance[k], stencil, value_bc))

def bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance >= 0
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]

    if nv_on_beg:
        f[[k] + i1] = f[[ksym] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] - feq[ksym, mask]
    else:
        f[i1 + [k]] = f[i2 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] - feq[mask, ksym]

def anti_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance >= 0
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]

    if nv_on_beg:
        f[[k] + i1] = -f[[ksym] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] + feq[ksym, mask]
    else:
        f[i1 + [k]] = -f[i2 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] + feq[mask, ksym]

def bouzidi_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance < .5
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]
    i3 = [i + 2*j for i, j in zip(i1, v.v)]
    s = 2.*bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = s*f[[ksym] + i2] + (1.-s)*f[[ksym] + i3]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] - feq[ksym, mask]
    else:
        f[i1 + [k]] = s*f[i2 + [ksym]] + (1.-s)*f[i3 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] - feq[mask, ksym]

    mask = np.logical_not(mask)
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]
    s = 0.5/bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = s*f[[ksym] + i2] + (1.-s)*f[[k] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] - feq[ksym, mask]
    else:
        f[i1 + [k]] = s*f[i2 + [ksym]] + (1.-s)*f[i2 + [k]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] - feq[mask, ksym]

def bouzidi_anti_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance < .5
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]
    i3 = [i + 2*j for i, j in zip(i1, v.v)]
    s = 2.*bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = -s*f[[ksym] + i2] - (1.-s)*f[[ksym] + i3]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] + feq[ksym, mask]
    else:
        f[i1 + [k]] = -s*f[i2 + [ksym]] - (1.-s)*f[i3 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] + feq[mask, ksym]

    mask = np.logical_not(mask)
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v)]
    s = 0.5/bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = -s*f[[ksym] + i2] - (1.-s)*f[[k] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] + feq[ksym, mask]
    else:
        f[i1 + [k]] = -s*f[i2 + [ksym]] - (1.-s)*f[i2 + [k]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] + feq[mask, ksym]

def neumann_vertical(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid these copies
    i1 = list(bv.indices.copy())
    i2 = list(bv.indices.copy())
    i2[0] += v.vx

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

def neumann_horizontal(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid these copies
    i1 = list(bv.indices.copy())
    i2 = list(bv.indices.copy())
    dim = bv.indices.shape[0]
    i2[1 if dim == 2 else -1] += v.vy

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

def neumann(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid this copy
    i1 = list(bv.indices.copy())
    i2 = [i + j for i, j in zip(i1, v.v)]

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

if __name__ == "__main__":
    from pyLBM.elements import *
    import geometry, domain
    import numpy as np

    dim = 2
    dx = .1
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.

    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,0,1,0]},
                     'Elements':[0],
                     0:{'Element':Circle([0.5*(xmin+xmax),0.5*(ymin+ymax)], 0.3),
                        'del':True,
                        'label':2}
                     }

    dico   = {'dim':dim,
              'eometry':dico_geometry,
              'space_step':dx,
              'number_of_schemes':1,
              0:{'velocities':range(9),}
              }

    geom = Geometry.Geometry(dico)
    dom = Domain.Domain(geom,dico)
    b = Boundary(dom, 2, 0)
    print b.indices
    print
    print b.distance
