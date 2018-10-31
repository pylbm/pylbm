# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from .storage import Array, AOS, SOA

class BaseContainer:
    gpu_support = False
    def __init__(self, domain, scheme, sorder, default_type):
        self.dim = domain.dim
        self.mpi_topo = domain.mpi_topo

        self.nv = scheme.stencil.nv_ptr[-1]
        self.nspace = domain.global_size
        self.vmax = domain.stencil.vmax
        self.sorder = sorder

        if sorder:
            self.m = Array(self.nv, self.nspace, self.vmax, sorder, self.mpi_topo, gpu_support=self.gpu_support)
            self.F = Array(self.nv, self.nspace, self.vmax, sorder, self.mpi_topo, gpu_support=self.gpu_support)
        else:
            self.m = default_type(self.nv, self.nspace, self.vmax, self.mpi_topo, gpu_support=self.gpu_support)
            self.F = default_type(self.nv, self.nspace, self.vmax, self.mpi_topo, gpu_support=self.gpu_support)
            sorder = [i for i in range(self.dim + 1)]

        self.m.set_conserved_moments(scheme.consm)
        self.F.set_conserved_moments(scheme.consm)

        self._set_sorder(sorder)

    def _set_sorder(self, sorder):
        pass

class NumpyContainer(BaseContainer):
    def __init__(self, domain, scheme, sorder=None, default_type=SOA):
        super(NumpyContainer, self).__init__(domain, scheme, sorder, default_type)
        self.Fnew = self.F

    def _set_sorder(self, sorder):
        if not self.sorder:
            self.sorder = [i for i in range(self.dim + 1)]

class CythonContainer(BaseContainer):
    def __init__(self, domain, scheme, sorder=None, default_type=AOS):
        super(CythonContainer, self).__init__(domain, scheme, sorder, default_type)
        self.Fnew = Array(self.nv, self.nspace, self.vmax, self.sorder, self.mpi_topo, gpu_support=self.gpu_support)
        self.Fnew.set_conserved_moments(scheme.consm)

    def _set_sorder(self, sorder):
        if not self.sorder:
            self.sorder = [self.dim] + [i for i in range(self.dim)]

class LoopyContainer(CythonContainer):
    gpu_support = True
    def __init__(self, domain, scheme, sorder=None, default_type=AOS):
        super(LoopyContainer, self).__init__(domain, scheme, sorder, default_type)

    def move2gpu(self, array):
        try:
            import pyopencl as cl
            import pyopencl.array #pylint: disable=unused-variable
            from .context import queue
        except ImportError:
            raise ImportError("Please install loo.py")
        return cl.array.to_device(queue, array)