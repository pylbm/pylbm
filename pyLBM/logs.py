# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import mpi4py.MPI as mpi
import logging

from .options import options

def setLogger(name):
    numeric_level = getattr(logging, options().loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s'.format(options().loglevel))

    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logger = logging.getLogger(name)
    logger.setLevel(level=numeric_level)

    return logger
