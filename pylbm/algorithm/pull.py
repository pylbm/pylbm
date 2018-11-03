# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from .base import BaseAlgorithm

class PullAlgorithm(BaseAlgorithm):
    def one_time_step_local(self, f, fnew, m):
        with_rel_velocity = True if self.rel_vel_symb else False

        f2m = self.f2m_local(f, m, with_rel_velocity)
        if isinstance(f2m, list):
            code = f2m
        else:
            code = [f2m]

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.relaxation_local(m, with_rel_velocity))

        if with_rel_velocity:
            code.append(self.restore_conserved_moments(m, f))

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.m2f_local(m, fnew, with_rel_velocity))

        return code
