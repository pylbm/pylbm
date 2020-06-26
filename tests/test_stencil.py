import pytest
import numpy as np
import pylbm

class TestStencil:
    def test_extract_dim(self, schemes):
        assert pylbm.Stencil.extract_dim(schemes.dico) == schemes.dim

    def test_is_symmetric(self, schemes):
        stencil = pylbm.Stencil(schemes.dico)
        assert stencil.is_symmetric()

    def test_get_symmetric_error(self, schemes):
        stencil = pylbm.Stencil(schemes.dico)
        with pytest.raises(ValueError):
            stencil.get_symmetric(axis=-3)
        with pytest.raises(ValueError):
            stencil.get_symmetric(axis=5)

    def test_get_all_velocities(self, schemes):
        stencil = pylbm.Stencil(schemes.dico)
        all_vel = stencil.get_all_velocities(0)

        assert all_vel[:, 0] == pytest.approx(stencil.vx[0])
        if schemes.dim > 1:
            assert all_vel[:, 1] == pytest.approx(stencil.vy[0])
        if schemes.dim == 3:
            assert all_vel[:, 2] == pytest.approx(stencil.vz[0])

    @pytest.mark.parametrize('unique_velocities', [True, False])
    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_visualize(self, schemes, unique_velocities):
        stencil = pylbm.Stencil(schemes.dico)
        views = stencil.visualize(unique_velocities=unique_velocities, view_label=False)
        return views.fig

    def test_velocities(self, schemes):
        stencil = pylbm.Stencil(schemes.dico)
        if isinstance(schemes.scheme, list):
            for il, l in enumerate(schemes.scheme):
                assert stencil.vx[il] == pytest.approx(l['vx'])
                if schemes.dim > 1:
                    assert stencil.vy[il] == pytest.approx(l['vy'])
                    if schemes.dim > 2:
                        assert stencil.vz[il] == pytest.approx(l['vz'])
        else:
            assert stencil.num[0] == pytest.approx(schemes.scheme['num'])
            assert stencil.unum == pytest.approx(schemes.scheme['num'])
            assert stencil.unvtot == len(schemes.scheme['num'])

            assert stencil.vx[0] == pytest.approx(schemes.scheme['vx'])
            assert stencil.uvx == pytest.approx(schemes.scheme['vx'])
            if schemes.dim > 1:
                assert stencil.vy[0] == pytest.approx(schemes.scheme['vy'])
                assert stencil.uvy == pytest.approx(schemes.scheme['vy'])
                if schemes.dim > 2:
                    assert stencil.vz[0] == pytest.approx(schemes.scheme['vz'])
                    assert stencil.uvz == pytest.approx(schemes.scheme['vz'])

#     assert dim == stencil.dim
#     assert nvel == stencil.num[0].size