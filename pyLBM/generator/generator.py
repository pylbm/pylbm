import collections
from .codegen import make_routine
from .autowrap import autowrap

class Generator(object):

    def __init__(self):
        self.routines = collections.OrderedDict()
        self.module = None

    def add_routine(self, name_expr, argument_sequence=None, local_vars=None, settings={}):
        self.routines[name_expr[0]] = make_routine(name_expr, argument_sequence, local_vars, settings)[0]

    def compile(self, backend="cython", verbose=False):
        self.module = autowrap(self.routines.values(), backend, verbose=verbose)

generator = Generator()

