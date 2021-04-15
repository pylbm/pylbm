# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
__init__ of the module containing the geometrical elements
"""

from .circle import Circle
from .ellipse import Ellipse
from .parallelogram import Parallelogram
from .triangle import Triangle

from .sphere import Sphere
from .ellipsoid import Ellipsoid
from .cylinder import CylinderCircle, CylinderEllipse, CylinderTriangle
from .cylinder import Parallelepiped

from .stl_element import STLElement

__all__ = ['Circle', 'Ellipse', 'Parallelogram', 'Triangle',
           'Sphere', 'Ellipsoid', 'CylinderCircle', 'CylinderEllipse', 'CylinderTriangle', 'Parallelepiped', 'STLElement']