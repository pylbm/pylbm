# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
"""
test the class Domain
"""

import pytest
import pylbm

CASES = [
    {
        'box': {'x': [0, 1], 'label':0},
        'space_step': 0.1,
        'schemes': [{'velocities': list(range(3))}],
    },
    {
        'box': {'x': [0, 2], 'y': [0, 1], 'label': 0},
        'elements': [
            pylbm.Ellipse((0.5, 0.5), (0.25, 0.25), (0.1, -0.1), label=1)
        ],
        'space_step': 0.05,
        'schemes':[{'velocities': list(range(13))}],
    },
    {
        'box': {'x': [0, 2], 'y': [0, 1], 'label': 0},
        'elements':[pylbm.Circle((0.5, 0.5), 0.2, label=1)],
        'space_step': 0.05,
        'schemes':[{'velocities': list(range(13))}],
    },
    {
        'box': {'x': [0, 1], 'y': [0, 1], 'label': [0, 1, 2, 3]},
        'space_step': 0.1,
        'schemes':[{'velocities': list(range(9))}],
    },
    {
        'box': {'x': [0, 3], 'y': [0, 1], 'label': [0, 1, 0, 2]},
        'elements': [
            pylbm.Parallelogram((0., 0.), (.5, 0.), (0., .5), label=0)
        ],
        'space_step': 0.125,
        'schemes':[{'velocities': list(range(9))}],
    },
    {
        'box': {'x': [-3, 3], 'y': [-3, 3], 'z': [-3, 3], 'label': 0},
        'elements': [
            pylbm.CylinderEllipse(
                [0.5, 0, 0], [0, 1, 1], [0, -1.5, 1.5], [1, -1, 0],
                label=[1, 2, 3]
            )
        ],
        'space_step': .5,
        'schemes': [{'velocities': list(range(19))}]
    },
    {
        'box': {
            'x': [0, 3],
            'y': [0, 3],
            'z': [0, 3],
            'label': list(range(1, 7))
        },
        'elements': [
            pylbm.Ellipsoid(
                (1.5, 1.5, 1.5),
                [.5, 0, 0],
                [0, .5, 0],
                [0, 0, 1],
                label=0
            )
        ],
        'space_step': 0.25,
        'schemes': [{'velocities': list(range(19))}]
    },
    {
        'box': {
            'x': [0, 2],
            'y': [0, 2],
            'z': [0, 2],
            'label': list(range(1, 7))
        },
        'elements': [pylbm.Sphere((1, 1, 1), 0.5, label=0)],
        'space_step': 0.25,
        'schemes': [{'velocities': list(range(19))}]
    },
    {
        'box': {
            'x': [0, 2],
            'y': [0, 2],
            'z': [0, 2],
            'label': list(range(6))
        },
        'space_step': 0.5,
        'schemes': [{'velocities': list(range(19))}]
    }
]


@pytest.fixture(params=CASES)
def case(request):
    """
    return the test cases
    """
    return request.param


@pytest.mark.mpl_image_compare(remove_text=True)
def test_domain_visualize(case):  # pylint: disable=redefined-outer-name
    """
    test the domain visualization
    """
    dom = pylbm.Domain(case)
    views = dom.visualize(
        view_distance=True,
        view_in=False,
        view_out=False,
        view_bound=True
    )
    return views.fig
