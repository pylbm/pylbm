# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import mpi4py.MPI as mpi
import logging
from colorlog import ColoredFormatter

from .options import options

loggers = {}

def setLogger(name):
    """
    usage: python *.py --log=INFO
    """
    global loggers

    numeric_level = getattr(logging, options().loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s'.format(options().loglevel))

    if loggers.get(name):
        return loggers.get(name)
    else:
        formatter = ColoredFormatter(
        "%(log_color)s[{0}] %(levelname)-8s%(reset)s %(blue)s%(name)s in function %(funcName)s line %(lineno)s\n%(black)s%(message)s".format(mpi.COMM_WORLD.Get_rank()),
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

        console = logging.StreamHandler()
        console.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level=numeric_level)
        logger.addHandler(console)

        loggers[name] = logger

        return logger
