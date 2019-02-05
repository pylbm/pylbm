# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from .base import BaseAlgorithm

class PullAlgorithm(BaseAlgorithm):
    def one_time_step_local(self, f, fnew, m):
        """
        Return symbolic expression which makes one time step of
        LBM algorithm using the pull algorithm. The difference with
        the basic algorithm is the transport and the computation
        of the moments from the distributed functions is made in
        one step.

            - compute the moments from the distributed functions + transport
            - source terms with dt/2 (with the moments)
            - relaxation (with the moments)
            - source terms with dt/2 (with the moments)
            - compute the new distributed functions from the moments

        Parameters
        ----------

        f : SymPy Matrix
            indexed objects for the old distributed functions

        fnew : SymPy Matrix
            indexed objects for the new distributed functions

        m : SymPy Matrix
            indexed objects for the moments

        """
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
