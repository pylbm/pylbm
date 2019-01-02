# FIXME: make pylint happy !
# pylint: disable=all

import collections
from .codegen import make_routine
from .autowrap import autowrap


class Generator:
    def __init__(self, backend, verbose=False):
        self.routines = collections.OrderedDict()
        self.module = None
        self.backend = backend
        self.verbose = verbose

    def add_routine(self, name_expr, argument_sequence=None,
                    local_vars=None, settings={}):
        self.routines[name_expr[0]] = make_routine(name_expr,
                                                   argument_sequence,
                                                   local_vars,
                                                   settings)[0]

    def compile(self):
        self.module = autowrap(self.routines.values(),
                               self.backend,
                               verbose=self.verbose)
