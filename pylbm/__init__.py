# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
A flexible Python package for lattice Boltzmann method.
"""
import logging
from colorlog import ColoredFormatter
import mpi4py.MPI as mpi

from colorama import init
init()

from .domain import Domain
from .stencil import Stencil
from .simulation import Simulation
from . import boundary as bc
from .scheme import Scheme
from .elements import * #pylint: disable=wildcard-import
from .geometry import Geometry
from . import viewer
from .hdf5 import H5File
from .options import options
from . import monitoring
from .analysis import EquivalentEquation, Stability

from .version import version as __version__

numeric_level = getattr(logging, options().loglevel, None) #pylint: disable=invalid-name

#pylint: disable=bad-continuation
formatter = ColoredFormatter( #pylint: disable=invalid-name
        "%(log_color)s[{}] %(levelname)-8s %(name)s in function %(funcName)s line %(lineno)s\n%(reset)s%(message)s".format(mpi.COMM_WORLD.Get_rank()), #pylint: disable=too-many-format-args
        datefmt=None,
        reset=True,
        log_colors={'DEBUG':    'green',
                    'INFO':     'cyan',
                    'WARNING':  'blue',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                   },
        style='%'
        )

logger = logging.getLogger(__name__) #pylint: disable=invalid-name
logger.setLevel(level=numeric_level)

console = logging.StreamHandler() #pylint: disable=invalid-name
console.setFormatter(formatter)
logger.addHandler(console)
