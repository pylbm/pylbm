# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import mpi4py.MPI as mpi
import logging
r = logging.getLogger()
r.handlers = []


def setLogger(name, lvl = None):
    log = logging.getLogger(name)
    #level = logging.getLogger().level
    #print level
    #log.setLevel(level)
    if lvl is not None:
        log.setLevel(lvl)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s on proc {0} \n%(message)s\n'.format(mpi.COMM_WORLD.Get_rank()))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return log

def compute_lvl(lvl_s):
    if lvl_s == 'CRITICAL':
        lvl = 50
    elif lvl_s == 'ERROR':
        lvl = 40
    elif lvl_s == 'WARNING':
        lvl = 30
    elif lvl_s == 'INFO':
        lvl = 20
    elif lvl_s == 'DEBUG':
        lvl = 10
    else:
        lvl = 40
    return lvl
