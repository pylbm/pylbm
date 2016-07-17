# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
import tempfile
import atexit
import shutil
import sys
import os
import importlib
import mpi4py.MPI as mpi

from ..logs import setLogger

INDENT = ' '*4

class Generator(object):
    """
    the generic class to generate the code

    Parameters
    ----------

    build_dir : string, optional
      the directory where the code is written
    suffix : string, optional
      the suffix of the file where the code is written

    Attributes
    ----------

    build_dir : string
      the directory where the code is written
    f : file identifier
      the file where the code is written
    code : string
      the generated code

    Methods
    -------

    setup :
      default setup function (empty)
    f2m :
      default f2m function (empty)
    m2f :
      default m2f function (empty)
    transport :
      default transport function (empty)
    relaxation :
      default relaxation function (empty)
    onetimestep :
      default one time step function (empty)
    compile :
      default compile function (writte the code in the file)
    get_module :
      get the name of the file where the code is written
    exit :
      exit function that erases the code

    Notes
    -----

    With pyLBM, the code can be generated in several langages.
    Each phase of the Lattice Boltzmann Method
    (as transport, relaxation, f2m, m2f, ...) is treated by an optimzed
    function written, compiled, and executed by the generator.

    The generated code can be read by typesetting the attribute
    ``code``.
    """
    def __init__(self, build_dir=None, suffix='.py'):
        self.log = setLogger(__name__)
        self.inv = None
        self.inspace = None
        self.build_dir = build_dir
        self.modulename = None
        self.importmodule = None
        self.code = ''

        self.build_dir = build_dir
        if build_dir is None:
            lbm_tmp_dir = os.path.expanduser("~") + '/.pylbm/'
            if not os.path.exists(lbm_tmp_dir):
                os.mkdir(lbm_tmp_dir)
            self.build_dir = tempfile.mkdtemp(dir=lbm_tmp_dir) +'/'
            self.f = tempfile.NamedTemporaryFile(dir=self.build_dir, suffix=suffix, delete=False)
            self.modulename = self.f.name.replace(self.build_dir, "").split('.')[0]

        self.build_dir = mpi.COMM_WORLD.bcast(self.build_dir, root=0)
        self.modulename = mpi.COMM_WORLD.bcast(self.modulename, root=0)

        self.log.info("Temporary file use for code generator :\n{0}".format(self.modulename))

    def setup(self):
        pass

    def f2m(self):
        pass

    def m2f(self,):
        pass

    def transport(self,):
        pass

    def relaxation(self,):
        pass

    def onetimestep(self,):
        pass

    def compile(self):
        if mpi.COMM_WORLD.Get_rank() == 0:
            self.log.info("*"*30 + "\n" + self.code + "\n" + "*"*30)
            self.f.write(self.code.encode("UTF-8"))
            self.f.close()

    def get_module(self):
        if self.importmodule is None:
            sys.path.append(self.build_dir)
            self.importmodule = importlib.import_module(self.modulename)
            sys.path.remove(self.build_dir)
        return self.importmodule

    def __del__(self):
        self.log.info("delete generator")
        if mpi.COMM_WORLD.Get_rank() == 0:
            shutil.rmtree(self.build_dir)
