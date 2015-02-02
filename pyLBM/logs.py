# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import mpi4py.MPI as mpi
import logging
from argparse import ArgumentParser

INIT_LOG = False

def init_logs():
    parser = ArgumentParser()
    parser.add_argument("--log", dest="loglevel", default="WARNING",
                         help="Set the log level (DEBUG, WARNING, ...)")
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s'.format(args.loglevel))

    ### for both packages
    logging.basicConfig(level=numeric_level)
    r = logging.getLogger()
    r.handlers = []

def __setLogger(name):
    global INIT_LOG
    if not INIT_LOG:
        init_logs()
        INIT_LOG = True
    log = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s on proc {0} \n%(message)s\n'.format(mpi.COMM_WORLD.Get_rank()))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return log
