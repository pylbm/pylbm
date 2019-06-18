# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
module monitoring
usage: python filename.py --monitoring
"""

# import time
import inspect
from functools import wraps
import atexit
import numpy as np
import mpi4py.MPI as mpi

from .options import options


class PerfMonitor:
    def __init__(self):
        self.count = 0
        self.total_time = []
        self.self_time = []


class Node:
    def __init__(self, parent=None, name='root'):
        self.parent = parent
        self.name = name
        self.sons = []
        self.time = 0

    def add_node(self, name):
        self.sons.append(Node(self, name))
        return self.sons[-1]

    def del_node(self):
        self.sons.clear()
        return self.parent

    def add_time(self, t):
        self.parent.time += t


class Monitoring:
    def __init__(self):
        self.func = {}
        self.size = 0
        self.tree = Node()

    @staticmethod
    def information(f):
        mod_name = inspect.getmodule(f).__name__
        try:
            f_name = f.__qualname__
            return (mod_name, f_name)
        except:
            pass

    def set_size(self, size):
        self.size = size

    def register(self, f):
        info = self.information(f)
        if not self.func.get(info, None):
            self.func[info] = PerfMonitor()

    def start_timing(self, f):
        info = self.information(f)
        self.tree = self.tree.add_node(info)
        self.func[info].total_time.append(mpi.Wtime())
        self.func[info].self_time.append(0)

    def stop_timing(self, f):
        info = self.information(f)
        t = mpi.Wtime()
        self.func[info].total_time[-1] = t - self.func[info].total_time[-1]
        self.tree.add_time(self.func[info].total_time[-1])
        self.func[info].self_time[-1] = self.func[info].total_time[-1] \
            - self.tree.time
        self.tree = self.tree.del_node()

    def __str__(self):
        if mpi.COMM_WORLD.rank == 0:
            titles = [
                '%', 'module name', 'function name',
                'ncall', 'total time', 'self time', 'MLUPS'
            ]
            row_format = "{:>6}{:>25}{:>30}{:>8}{:>15}{:>15}{:>8}"
            print('\n', row_format.format(*titles), '\n')
            row_format = "{:6.1f}{:>25}{:>30}{:8}{:15.9f}{:15.9f}{:8.2f}"
            names = [[k[0], k[1]] for k in self.func.keys()]
            data = [
                [
                    len(v.total_time),
                    np.sum(v.total_time),
                    np.sum(v.self_time)
                ] for v in self.func.values()
            ]
            ind = np.argsort(np.asarray(data)[:, 1])
            for i in ind[::-1]:
                print(row_format.format(data[i][1]/data[ind[-1]][1]*100,
                                        *names[i],
                                        *data[i],
                                        data[i][0]*self.size/data[i][1]/1e6))

Monitor = Monitoring()  # pylint: disable=invalid-name

if options().monitoring:
    atexit.register(Monitor.__str__)


def monitor(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        Monitor.register(f)
        Monitor.start_timing(f)
        output = f(*args, **kwds)
        Monitor.stop_timing(f)
        return output
    return wrapper
