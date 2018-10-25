import pytest
import numpy as np
import pylbm

elements = [
    [2, pylbm.Circle([0, 0], 1)],
    [2, pylbm.Ellipse([0, 0], [1, 0], [0, 1])],
    [2, pylbm.Triangle([-1, -1], [0, 2], [2, 0])],
    [2, pylbm.Parallelogram([-1, -1], [0, 2], [2, 0])],
    # [3, pylbm.CylinderCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    # [3, pylbm.CylinderEllipse([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    # [3, pylbm.CylinderTriangle([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    # [3, pylbm.Parallelepiped([-1, -1, -1], [2, 0, 0], [0, 2, 0], [0, 0, 2])],
    [3, pylbm.Ellipsoid([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    [3, pylbm.Sphere([0, 0, 0], 1)],
]


@pytest.fixture(params=elements, ids=[elem[1].__class__.__name__ for elem in elements])
def get_element(request):
    return request.param

class TestBase:
    def test_bounds(self, get_element):
        dim, element = get_element
        bounds = element.get_bounds()
        assert bounds[0] == pytest.approx([-1]*dim)
        assert bounds[1] == pytest.approx([1]*dim)

    # def test_point_inside(self, get_element):
    #     dim, element = get_element
    #     assert element.point_inside([0]*dim)
    #     assert element.point_inside([1]+ [0]*(dim-1))
    #     assert not element.point_inside([1.5]*dim)

    # def test_distance(self, get_element):
    #     dim, element = get_element
    #     dist = np.zeros(dim)
    #     dist[0] = 1
    #     print(element.distance([np.zeros(1)]*dim, [-1]+[0]*(dim-1)))
    #     assert element.distance([np.zeros(1)]*dim, [-1]+[0]*(dim-1)) == pytest.approx(dist)
