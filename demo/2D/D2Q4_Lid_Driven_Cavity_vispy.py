##############################################################################
# Solver D2Q4^3 for a Lid driven cavity
#
# d_t(p) + d_x(ux) + d_y(uy) = 0
# d_t(ux) + d_x(ux^2) + d_y(ux*uy) + d_x(p) = mu (d_xx+d_yy)(ux)
# d_t(uy) + d_x(ux*uy) + d_y(uy^2) + d_y(p) = mu (d_xx+d_yy)(uy)
#
# in a square cavity of length 1.
#
#   -- -> -> -> -> --
#   |               |
#   |               |
#   |               |
#   |               |
#   |               |
#   |               |
#   -----------------
#
# the variables of the three D2Q4 are p, ux, and uy
# initialization with 0.
# boundary conditions
#     - ux = uy = 0. on bottom, left, and right
#     - ux = u0, uy = 0. on top
#
##############################################################################

import sys

import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import pyLBM

from vispy import gloo
from vispy import app
from vispy.gloo import gl
from vispy.util.transforms import ortho


VERT_SHADER = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * vec4(a_position,0.0,1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    float c = texture2D(u_texture, v_texcoord).r;
    //gl_FragColor = vec4(c, 0., 1.-c, 1.);
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}
"""

class Canvas(app.Canvas):

    def __init__(self, dico):
        coeff = 4
        self.sol = pyLBM.Simulation(dico)
        self.indout = np.where(self.sol.domain.in_or_out == self.sol.domain.valout)
        W, H = self.sol.domain.N
        self.W, self.H = W, H
        # A simple texture quad
        self.data = np.zeros(4, dtype=[ ('a_position', np.float32, 2),
                                        ('a_texcoord', np.float32, 2) ])
        self.data['a_position'] = np.array([[0, 0], [coeff * W, 0], [0, coeff * H], [coeff * W, coeff * H]])
        self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        app.Canvas.__init__(self)
        self.title = "Solution t={0:f}".format(0.)
        self.min, self.max = 0., .5*uo
        self.ccc = 1./(self.max-self.min)
        self.size = W * coeff, H * coeff
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.ccc*(self.sol.m[0][0][1:-1, 1:-1].astype(np.float32).transpose() - self.min))
        self.texture.interpolation = gl.GL_LINEAR

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(self.data))

        self.projection = np.eye(4, dtype=np.float32)
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        self.timer = app.Timer(self.sol.dt)
        self.timer.connect(self.on_timer)

    def on_initialize(self, event):
        gl.glClearColor(1,1,1,1)

    def on_resize(self, event):
        width, height = event.size
        gl.glViewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

        W, H = self.W, self.H
        # Compute the new size of the quad
        r = width / float(height)
        R = W / float(H)
        if r < R:
            w, h = width, width / R
            x, y = 0, int((height - h) / 2)
        else:
            w, h = height * R, height
            x, y = int((width - w) / 2), 0
        self.data['a_position'] = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(gloo.VertexBuffer(self.data))

    def on_draw(self, event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.program.draw(gl.GL_TRIANGLE_STRIP)

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()
        if event.text == 'n':
            self.go_on()
            self.maj()
        if event.text == 't':
            print "MLUPS: {0:5.1f}".format(self.sol.cpu_time['MLUPS'])

    def on_timer(self, event):
        self.go_on()
        self.maj()

    def go_on(self):
        nrep = 32
        for i in xrange(nrep):
            self.sol.one_time_step()

    def maj(self):
        self.title = "Solution t={0:f}".format(self.sol.t)
        self.sol.f2m()
        self.texture.set_data(self.ccc*(np.sqrt(
            self.sol.m[1][0][1:-1, 1:-1].astype(np.float32)**2
            + self.sol.m[2][0][1:-1, 1:-1].astype(np.float32)**2).transpose()
            - self.min))
        self.update()

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def initialization_qx(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def bc_top(f, m, x, y, scheme):
    ######### BEGIN OF WARNING #########
    # the order depends on the compilater
    # through the variable nv_on_beg
    m[:, 0] = 0.
    m[:, 4] = uo
    m[:, 8] = 0.
    #########  END OF WARNING  #########
    scheme.equilibrium(m)
    scheme.m2f(m, f)

if __name__ == "__main__":
    # parameters
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    dx = 1./128 # spatial step
    la = 1. # velocity of the scheme
    uo = 0.1
    mu   = 0.000185
    zeta = 1.e-2
    cte = 3.

    dummy = 3.0/(la*dx)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)

    vitesse = range(1, 5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 0, 1, 0]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype': 'moments',
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s1, s1, 1.],
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s2, s2, 1.],
                    'equilibrium':Matrix([u[1][0], u[1][0]**2 + u[0][0]/cte, u[1][0]*u[2][0], 0.]),
                    'init':{0:(initialization_qx,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s2, s2, 1.],
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0], u[2][0]**2 + u[0][0]/cte, 0.]),
                    'init':{0:(initialization_qy,)},
                    },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.bouzidi_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back
                         },
                'value':None,
            },
            1:{'method':{0: pyLBM.bc.bouzidi_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back
                         },
                'value':bc_top,
            },
        },
        'generator':pyLBM.generator.CythonGenerator,
    }

    c = Canvas(dico)
    c.show()
    app.run()
