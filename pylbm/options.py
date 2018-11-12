# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
pylbm CLI options
"""
from argparse import ArgumentParser

def options():
    """
    pylbm command line options
    """
    parser = ArgumentParser()
    logging = parser.add_argument_group('log')
    logging.add_argument("--log", dest="loglevel", default="WARNING",
                         choices=['WARNING', 'INFO', 'DEBUG', 'ERROR'],
                         help="Set the log level")
    monitoring = parser.add_argument_group('monitoring')
    monitoring.add_argument("--monitoring", action="store_true",
                            help="Set the monitoring")
    mpi = parser.add_argument_group('mpi splitting')
    mpi.add_argument("-npx", dest="npx", default=1, type=int,
                     help="Set the number of processes in x direction")
    mpi.add_argument("-npy", dest="npy", default=1, type=int,
                     help="Set the number of processes in y direction")
    mpi.add_argument("-npz", dest="npz", default=1, type=int,
                     help="Set the number of processes in z direction")
    args, _ = parser.parse_known_args()
    return args
