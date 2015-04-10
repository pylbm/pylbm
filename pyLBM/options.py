# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from argparse import ArgumentParser

def options():
    parser = ArgumentParser()
    logging = parser.add_argument_group('log')
    logging.add_argument("--log", dest="loglevel", default="WARNING",
                         choices=['WARNING', 'INFO', 'DEBUG', 'ERROR'],
                         help="Set the log level")
    mpi = parser.add_argument_group('mpi splitting')
    mpi.add_argument("-npx", dest="npx", default=1, type=int,
                             help="Set the number of processes in x direction")
    mpi.add_argument("-npy", dest="npy", default=1, type=int,
                             help="Set the number of processes in y direction")
    mpi.add_argument("-npz", dest="npz", default=1, type=int,
                     help="Set the number of processes in z direction")
    args, unknown = parser.parse_known_args()
    return args
    #return parser.parse_args()
