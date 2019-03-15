# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
stability analysis
"""

import sympy as sp
import numpy as np

from .. import viewer
from ..utils import print_progress
from ..symbolic import rel_ux, rel_uy, rel_uz, alltogether

XI = sp.symbols('xi_1, xi_2, xi_3')


class Stability:
    """
    generic class
    """
    def __init__(self, scheme):
        # pylint: disable=unsubscriptable-object
        self.nvtot = scheme.s.shape[0]
        self.consm = list(scheme.consm.keys())
        self.param = scheme.param.items()
        self.dim = scheme.dim
        self.is_stable_l2 = True

        if scheme.rel_vel is None:
            jacobian = scheme.EQ.jacobian(self.consm)
        else:
            jacobian = (scheme.Tu * scheme.EQ).jacobian(self.consm)
        relax_mat_m = sp.eye(self.nvtot) - sp.diag(*scheme.s)
        relax_mat_m[:, :len(self.consm)] += sp.diag(*scheme.s) * jacobian

        if scheme.rel_vel is not None:
            relax_mat_m = scheme.Tmu * relax_mat_m * scheme.Tu
            relax_mat_m = relax_mat_m.subs(
                [
                    (i, j) for i, j in zip(
                        [rel_ux, rel_uy, rel_uz], scheme.rel_vel
                    )
                ]
            )
        alltogether(relax_mat_m)
        self.relax_mat_f = scheme.invM * relax_mat_m * scheme.M
        alltogether(self.relax_mat_f)

        velocities = sp.Matrix(scheme.stencil.get_all_velocities())
        wave_vector = sp.Matrix(XI[:self.dim])
        self.stream_mat_f = sp.exp(
            sp.diag(*(velocities.dot(wave_vector)))
        )
        self.wave_vector = XI[:self.dim]

    def eigenvalues(self, consm0, n_wv):
        """
        Compute the eigenvalues of the amplification matrix
        for n_wv wave vectors
        """
        to_subs = list((i, j) for i, j in zip(self.consm, consm0))
        to_subs += list(self.param)
        relax_mat_f_num = self.relax_mat_f.subs(to_subs)
        amplification_matrix = self.stream_mat_f * relax_mat_f_num

        if self.dim == 1:
            v_xi = np.linspace(0, 2*np.pi, n_wv, endpoint=False)
            v_xi = v_xi[np.newaxis, :]
        elif self.dim == 2:
            n_wv_0 = int(np.sqrt(n_wv))
            v_xi_0 = np.linspace(0, 2*np.pi, n_wv_0, endpoint=False)
            v_xi_x, v_xi_y = np.meshgrid(v_xi_0, v_xi_0)
            v_xi = np.array([v_xi_x.flatten(), v_xi_y.flatten()])
            n_wv = v_xi.shape[1]
        eigs = np.empty((n_wv, self.nvtot), dtype='complex')

        print("*"*80)
        print("Compute the eigenvalues")
        print_progress(0, n_wv, barLength=50)
        # data_num = np.empty((self.nvtot, self.nvtot), dtype='complex')
        for k in range(n_wv):
            data = amplification_matrix.subs(
                list(
                    (i, j) for i, j in zip(
                        self.wave_vector, 1j*v_xi[:, k]
                    )
                )
            )
            data = np.array(data).astype('complex')
            eigs[k] = np.linalg.eig(data)[0]
            # data_num[:] = np.asarray(data)
            # eigs[k] = np.linalg.eig(data_num)[0]
            print_progress(k+1, n_wv, barLength=50)
        print('')
        print("*"*80)

        ind_pb, = np.where(np.max(np.abs(eigs), axis=1) > 1 + 1.e-10)
        pb_stable_l2 = v_xi[:, ind_pb]
        self.is_stable_l2 = pb_stable_l2.shape[1] == 0

        if self.is_stable_l2:
            print("*"*80)
            print("The scheme is stable")
            print("*"*80)
        else:
            print("*"*80)
            print("The scheme is not stable for these wave vectors:")
            print(pb_stable_l2.T)
            print("*"*80)

        return v_xi, eigs

    def visualize(self, dico=None, viewer_app=viewer.matplotlib_viewer):
        """
        visualize the stability
        """
        if dico is None:
            dico = {}
        consm0 = [0.] * len(self.consm)
        dicolin = dico.get('linearization', None)
        if dicolin is not None:
            for k, moment in enumerate(self.consm):
                consm0[k] = dicolin.get(moment, 0.)

        n_wv = dico.get('number_of_wave_vectors', 1024)
        v_xi, eigs = self.eigenvalues(consm0, n_wv)

        fig = viewer_app.Fig(1, 2, figsize=(12, 6))
        if self.dim == 1:
            color = 'orange'
        elif self.dim == 2:
            color = .5 + .5/np.pi*np.arctan2(v_xi[0, :], v_xi[1, :])
        pos = np.empty((v_xi.shape[1], 2))

        # real and imaginary part
        view0 = fig[0]
        view0.title = "Stability"
        view0.axis(-1.1, 1.1, -1.1, 1.1, aspect='equal')
        view0.grid(visible=False)
        view0.set_label('real part', 'imaginary part')
        view0.ax.set_xticks([-1, 0, 1])
        view0.ax.set_xticklabels([r"$-1$", r"$0$", r"$1$"])
        view0.ax.set_yticks([-1, 0, 1])
        view0.ax.set_yticklabels([r"$-1$", r"$0$", r"$1$"])

        theta = np.linspace(0, 2*np.pi, 1000)
        view0.plot(
            np.cos(theta), np.sin(theta),
            alpha=0.5, color='navy', width=0.5,
        )
        for k in range(self.nvtot):
            pos[:, 0] = np.real(eigs[:, k])
            pos[:, 1] = np.imag(eigs[:, k])
            view0.markers(pos, 2, color=color, alpha=0.5)

        # modulus
        view1 = fig[1]
        view1.title = "Stability"
        view1.axis(0, 2*np.pi, -.1, 1.1)
        view1.grid(visible=True)
        view1.set_label('wave vector modulus', 'modulus')
        view1.ax.set_xticks([k*np.pi/4 for k in range(0, 9)])
        view1.ax.set_xticklabels(
            [
                r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",
                r"$\frac{3\pi}{4}$", r"$\pi$",
                r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$",
                r"$\frac{7\pi}{4}$", r"$2\pi$"
            ]
        )
        view1.plot(
            [0, 2*np.pi], [1., 1.],
            alpha=0.5, color='navy', width=0.5,
        )
        pos[:, 0] = np.sqrt(np.sum(v_xi**2, axis=0))
        for k in range(self.nvtot):
            pos[:, 1] = np.abs(eigs[:, k])
            view1.markers(pos, 2, color=color, alpha=0.5)

        fig.show()
