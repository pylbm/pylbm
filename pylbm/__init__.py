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

# pylint: disable=wrong-import-position
# pylint: disable=wildcard-import
# pylint: disable=invalid-name

from .version import version as __version__  # noqa: E402
from .domain import Domain                   # noqa: E402
from .stencil import Stencil                 # noqa: E402
from .simulation import Simulation           # noqa: E402
from . import boundary as bc                 # noqa: E402
from .scheme import Scheme                   # noqa: E402
from .elements import *                      # noqa: E402
from .geometry import Geometry               # noqa: E402
from . import viewer                         # noqa: E402
from .hdf5 import H5File                     # noqa: E402
from .options import options                 # noqa: E402
from . import monitoring                     # noqa: E402
from .analysis import EquivalentEquation, Stability  # noqa: E402
from .utils import progress_bar              # noqa: E402

numeric_level = getattr(logging, options().loglevel, None)

formatter = ColoredFormatter(
        f"%(log_color)s[{mpi.COMM_WORLD.Get_rank()}] "
        "%(levelname)-8s %(name)s in function %(funcName)s "
        "line %(lineno)s\n%(reset)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
                'DEBUG':    'green',
                'INFO':     'cyan',
                'WARNING':  'blue',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
        },
        style='%'
)

logger = logging.getLogger(__name__)
logger.setLevel(level=numeric_level)

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
