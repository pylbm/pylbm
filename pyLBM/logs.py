# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import mpi4py.MPI as mpi
import logging
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--log", dest="loglevel", default="WARNING"
                  )
(options, args) = parser.parse_args()

numeric_level = getattr(logging, options.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % options.loglevel)
logging.basicConfig(level=numeric_level)

r = logging.getLogger()
r.handlers = []

def setLogger(name):
    log = logging.getLogger(name)
    #level = logging.getLogger().level
    #print level
    #log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s on proc {0} \n%(message)s\n'.format(mpi.COMM_WORLD.Get_rank()))
    stream_handler = logging.StreamHandler()
    #stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return log
