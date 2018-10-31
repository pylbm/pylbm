# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from .base import BaseAlgorithm

class PullAlgorithm(BaseAlgorithm):
    def one_time_step_local(self, f, fnew, m):
        code = [self.f2m_local(f, m)]

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.relaxation_local(m))

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.m2f_local(m, fnew))

        return code
