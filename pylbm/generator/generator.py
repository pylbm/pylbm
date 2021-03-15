# FIXME: make pylint happy !
# pylint: disable=all

import collections
from .codegen import make_routine
from .autowrap import autowrap


class Generator:
    def __init__(self, backend, directory=None, generate=True, verbose=False):
        self.routines = collections.OrderedDict()
        self.module = None
        self.directory = directory
        self.generate = generate
        self.backend = backend
        self.verbose = verbose

    def add_routine(self, name_expr,
                    local_vars=None, settings={}):
        self.routines[name_expr[0]] = make_routine(name_expr[0], name_expr[1],
                                                   user_local_vars=local_vars,
                                                   language=self.backend,
                                                   settings=settings)

    def compile(self):
        self.module = autowrap(self.routines.values(),
                               self.backend,
                               self.directory,
                               generate=self.generate,
                               verbose=self.verbose)
