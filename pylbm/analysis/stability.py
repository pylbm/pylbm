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
from ..symbolic import rel_ux, rel_uy, rel_uz, recursive_sub


class Stability:
    """
    generic class
    """
    def __init__(self, scheme, output_txt=False):
        # pylint: disable=unsubscriptable-object
        self.nvtot = scheme.s.shape[0]
        self.consm = list(scheme.consm.keys())
        self.param = scheme.param
        self.dim = scheme.dim
        self.is_stable_l2 = True
        self.output_txt = output_txt

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
        self.relax_mat_f = scheme.invM * relax_mat_m * scheme.M
        # alltogether(self.relax_mat_f)

        velocities = sp.Matrix(scheme.stencil.get_all_velocities())
        self.velocities = np.asarray(velocities).astype('float')

    def eigenvalues(self, consm0, n_wv, extra_parameters=None):
        """
        Compute the eigenvalues of the amplification matrix
        for n_wv wave vectors
        """
        extra_parameters = extra_parameters or {}
        to_subs = list((i, j) for i, j in zip(self.consm, consm0))
        to_subs += list(self.param.items())
        to_subs += list(extra_parameters.items())

        relax_mat_f_num = recursive_sub(self.relax_mat_f, to_subs)

        if self.dim == 1:
            v_xi = np.linspace(0, 2*np.pi, n_wv, endpoint=False)
            v_xi = v_xi[np.newaxis, :]
        elif self.dim == 2:
            n_wv_0 = int(np.sqrt(n_wv))
            v_xi_0 = np.linspace(0, 2*np.pi, n_wv_0, endpoint=False)
            v_xi_x, v_xi_y = np.meshgrid(v_xi_0, v_xi_0)
            v_xi = np.array([v_xi_x.flatten(), v_xi_y.flatten()])
            n_wv = v_xi.shape[1] #pylint: disable=unsubscriptable-object
        eigs = np.empty((n_wv, self.nvtot), dtype='complex')

        if self.output_txt:
            print("*"*80)
            print("Compute the eigenvalues")
            print_progress(0, n_wv, barLength=50)

        relax_mat_f_num = np.asarray(relax_mat_f_num).astype('float')

        def set_matrix(wave_vector):
            return np.exp(
                self.velocities.dot(wave_vector)
            )[np.newaxis, :] * relax_mat_f_num

        for k in range(n_wv):
            data = set_matrix(1j*v_xi[:, k])
            eigs[k] = np.linalg.eig(data)[0]
            if self.output_txt:
                print_progress(k+1, n_wv, barLength=50)

        ind_pb, = np.where(np.max(np.abs(eigs), axis=1) > 1 + 1.e-10)
        pb_stable_l2 = v_xi[:, ind_pb]
        self.is_stable_l2 = pb_stable_l2.shape[1] == 0

        if self.output_txt:
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

    def visualize(self, dico=None, viewer_app=viewer.matplotlib_viewer, with_widgets=False):
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
        nx = v_xi.shape[1]

        fig = viewer_app.Fig(1, 2, figsize=(12.8, 6.4))  # , figsize=(12, 6))
        if self.dim == 1:
            color = 'orange'
        elif self.dim == 2:
            color = .5 + .5/np.pi*np.arctan2(v_xi[0, :], v_xi[1, :])
            color = np.repeat(
                color[np.newaxis, :], self.nvtot, axis=0
            ).flatten()

        # real and imaginary part
        view0 = fig[0]
        view0.title = "Stability: {}".format(self.is_stable_l2)
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

        pos0 = np.empty((nx*self.nvtot, 2))
        for k in range(self.nvtot):
            pos0[nx*k:nx*(k+1), 0] = np.real(eigs[:, k])
            pos0[nx*k:nx*(k+1), 1] = np.imag(eigs[:, k])
        markers0 = view0.markers(pos0, 5, color=color, alpha=0.5)

        # modulus
        view1 = fig[1]
        view1.title = "Stability: {}".format(self.is_stable_l2)
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

        pos1 = np.empty((nx*self.nvtot, 2))
        for k in range(self.nvtot):
            # pos1[nx*k:nx*(k+1), 0] = np.sqrt(np.sum(v_xi**2, axis=0))
            pos1[nx*k:nx*(k+1), 0] = np.max(v_xi, axis=0)
            pos1[nx*k:nx*(k+1), 1] = np.abs(eigs[:, k])
        markers1 = view1.markers(pos1, 5, color=color, alpha=0.5)

        # create sliders to play with parameters
        dicosliders = dico.get('parameters', None)

        if with_widgets:
            from ipywidgets import widgets
            from IPython.display import display, clear_output

            out = widgets.Output()

            sliders = {}
            if dicosliders:
                for k, v in dicosliders.items():
                    sliders[k] = widgets.FloatSlider(value=v['init'],
                                                    min=v['range'][0],
                                                    max=v['range'][1],
                                                    step=v['step'],
                                                    continuous_update=False,
                                                    description=v.get('name', sp.pretty(k)),
                                                    layout=widgets.Layout(width='80%'))

            with out:
                fig.show()

            def update(val):  # pylint: disable=unused-argument
                for k, v in sliders.items():
                    if k in self.param.keys():
                        self.param[k] = v.value
                    for i_m, moment in enumerate(self.consm):
                        if moment == k:
                            consm0[i_m] = v.value

                v_xi, eigs = self.eigenvalues(consm0, n_wv)

                for k in range(self.nvtot):
                    pos0[nx*k:nx*(k+1), 0] = np.real(eigs[:, k])
                    pos0[nx*k:nx*(k+1), 1] = np.imag(eigs[:, k])

                markers0.set_offsets(pos0)
                view0.title = "Stability: {}".format(self.is_stable_l2)

                for k in range(self.nvtot):
                    # pos1[nx*k:nx*(k+1), 0] = np.sqrt(np.sum(v_xi**2, axis=0))
                    pos1[nx*k:nx*(k+1), 0] = np.max(v_xi, axis=0)
                    pos1[nx*k:nx*(k+1), 1] = np.abs(eigs[:, k])

                markers1.set_offsets(pos1)
                view1.title = "Stability: {}".format(self.is_stable_l2)
                fig.fig.canvas.draw_idle()
                with out:
                    clear_output(wait=True)
                    display(fig.fig)

            for k in sliders.keys():
                sliders[k].observe(update)

            display(out)
            if dicosliders:
                for k, v in dicosliders.items():
                    display(sliders[k])

        else:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider

            axcolor = 'lightgoldenrodyellow'

            viewer_app.Fig(figsize=(6, 2))
            sliders = {}
            if dicosliders:
                item = 0
                length = 0.8/len(dicosliders)
                for k, v in dicosliders.items():
                    axe = plt.axes(
                        [0.2, 0.1+item*length, 0.65, 0.8*length],
                        facecolor=axcolor,
                    )
                    sliders[k] = Slider(
                        axe,
                        v.get('name', sp.pretty(k)),
                        *v['range'],
                        valinit=v['init'],
                        valstep=v['step']
                    )
                    item += 1

            def update(val):  # pylint: disable=unused-argument
                for k, v in sliders.items():
                    if k in self.param.keys():
                        self.param[k] = v.val
                    for i_m, moment in enumerate(self.consm):
                        if moment == k:
                            consm0[i_m] = v.val

                v_xi, eigs = self.eigenvalues(consm0, n_wv)

                for k in range(self.nvtot):
                    pos0[nx*k:nx*(k+1), 0] = np.real(eigs[:, k])
                    pos0[nx*k:nx*(k+1), 1] = np.imag(eigs[:, k])

                markers0.set_offsets(pos0)
                view0.title = "Stability: {}".format(self.is_stable_l2)

                for k in range(self.nvtot):
                    # pos1[nx*k:nx*(k+1), 0] = np.sqrt(np.sum(v_xi**2, axis=0))
                    pos1[nx*k:nx*(k+1), 0] = np.max(v_xi, axis=0)
                    pos1[nx*k:nx*(k+1), 1] = np.abs(eigs[:, k])

                markers1.set_offsets(pos1)
                view1.title = "Stability: {}".format(self.is_stable_l2)
                fig.fig.canvas.draw_idle()

            for k in sliders.keys():
                sliders[k].on_changed(update)

            fig.show()
