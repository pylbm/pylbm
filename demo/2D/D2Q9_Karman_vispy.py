import sys

import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import pyLBM
from pyLBM.elements import *
import pyLBM.geometry as pyLBMGeom
import pyLBM.stencil as pyLBMSten
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme
import pyLBM.simulation as pyLBMSimu
import pyLBM.boundary as pyLBMBound
import pyLBM.generator as pyLBMGen

import numba

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
    gl_FragColor = vec4(c, 0., 1.-c, 1.);
    //gl_FragColor = texture2D(u_texture, v_texcoord);
    //gl_FragColor.a = 1.0;
}
"""

class Canvas(app.Canvas):

    def __init__(self, dico):
        coeff = 2
        self.sol = pyLBMSimu.Simulation(dico, nv_on_beg=False)
        self.indout = np.where(self.sol.domain.in_or_out == self.sol.domain.valout)
        W, H = self.sol._m.shape[:-1]
        W -= 2
        H -= 2
        self.W, self.H = W, H
        # A simple texture quad
        self.data = np.zeros(4, dtype=[ ('a_position', np.float32, 2),
                                        ('a_texcoord', np.float32, 2) ])
        self.data['a_position'] = np.array([[0, 0], [coeff * W, 0], [0, coeff * H], [coeff * W, coeff * H]])
        self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        app.Canvas.__init__(self, close_keys='escape')
        self.title = "Solution t={0:f}".format(0.)
        deltarho = rhoo
        self.min, self.max = 0., 2*self.sol.domain.dx #-2*self.sol.domain.dx, 2*self.sol.domain.dx
        self.ccc = 1./(self.max-self.min)
        self.size = W * coeff, H * coeff
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.ccc*(self.sol._m[1:-1, 1:-1, 0].astype(np.float32) - self.min))
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

    def on_timer(self, event):
        self.go_on()
        self.maj()

    def go_on(self):
        nrep = 1
        for i in xrange(nrep):
            self.sol.one_time_step()

    def maj(self):
        self.title = "Solution t={0:f}".format(self.sol.t)
        self.sol.scheme.f2m(self.sol._F, self.sol._m)

        #self.sol._m[self.indout[0], self.indout[1], :] = 1.
        #print "f"*20
        #print self.sol._F.T
        #print "m"*20
        #print self.sol._m.T

        # self.texture.set_data(self.ccc*np.abs(
        #     self.sol.m[0][2][2:, 1:-1].astype(np.float32) - self.sol.m[0][2][:-2,1:-1].astype(np.float32)
        #     - self.sol.m[0][1][1:-1,2:].astype(np.float32) + self.sol.m[0][1][1:-1,:-2].astype(np.float32)
        #     - self.min))

        #self.texture.set_data(self.ccc*np.abs(
        #    self.sol._m[2:, 1:-1, 2].astype(np.float32) - self.sol._m[:-2,1:-1, 2].astype(np.float32)
        #    - self.sol._m[1:-1, 2:, 1].astype(np.float32) + self.sol._m[1:-1, :-2, 1].astype(np.float32)
        #    - self.min))

        self.texture.set_data(self.ccc*np.abs(
            self.sol._m[1:-1, 2:, 2].astype(np.float32) - self.sol._m[1:-1, :-2, 2].astype(np.float32)
            - self.sol._m[2:, 1:-1, 1].astype(np.float32) + self.sol._m[:-2, 1:-1, 1].astype(np.float32)
            - self.min))


        self.update()

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((y.shape[0], x.shape[0]), dtype='float64')

def initialization_qx(x,y):
    return uo * np.ones((y.shape[0], x.shape[0]), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((y.shape[0], x.shape[0]), dtype='float64')

def bc_rect(f, m, x, y, scheme):
    # m[0][0] = 0.
    # m[0][1] = rhoo*uo
    # m[0][2] = 0.
    m[:, 0] = 0.
    m[:, 1] = rhoo*uo
    m[:, 2] = 0.
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot_vorticity(sol,num):
    V = sol.m[0][2][2:,1:-1] - sol.m[0][2][0:-2,1:-1] - sol.m[0][1][1:-1,2:] + sol.m[0][1][1:-1,0:-2]
    V /= np.sqrt(V**2+1.e-5)
    plt.imshow(np.float32(V.transpose()), origin='lower', cmap=cm.gray)
    plt.title('Vorticity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.savefig("sauvegarde_images/Karman_{0:04d}.pdf".format(num))
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = 0., 5., 0., 3
    rayon = 0.125
    dx = 1./64 # spatial step
    la = 1. # velocity of the scheme
    rhoo = 1.
    uo = 0.05
    mu   = 2.e-5 #0.00185
    zeta = 10*mu
    dummy = 3.0/(la*rhoo*dx)
    s3 = 1.0/(0.5+zeta*dummy)
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = 1.0/(0.5+mu*dummy)
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 1, 0, 0]},
        'elements':{0: {'element':Circle([0.75, 0.5*(ymin+ymax)+2*dx], rayon), 'label':2, 'del':0}},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        0:{'velocities':range(9),
           'polynomials':Matrix([1,
                                 LA*X, LA*Y,
                                 3*(X**2+Y**2)-4,
                                 0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                                 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                                 X**2-Y**2, X*Y]),
            'relaxation_parameters':s,
            'equilibrium':Matrix([u[0][0],
                                  u[0][1], u[0][2],
                                  -2*u[0][0] + 3*q2,
                                  u[0][0]+1.5*q2,
                                  -u[0][1]/LA, -u[0][2]/LA,
                                  qx2-qy2, qxy]),
        },
        'init':{'type':'moments', 0:{0:(initialization_rho,),
                                     1:(initialization_qx,),
                                     2:(initialization_qy,)
                                     }
        },
        'boundary_conditions':{
            0:{'method':{0: pyLBMBound.bouzidi_bounce_back}, 'value':bc_rect},
            1:{'method':{0: pyLBMBound.neumann_vertical}, 'value':None},
            2:{'method':{0: pyLBMBound.bouzidi_bounce_back}, 'value':None},
        },
        'generator': pyLBMGen.CythonGenerator,
    }

    Re = rhoo*uo*2*rayon/mu
    print "Reynolds number {0:10.3e}".format(Re)
    c = Canvas(dico)
    #for i in xrange(100):
    #    c.go_on()
    #np.set_printoptions(threshold=1e6)
    #print c.sol._F.T
    c.show()
    app.run()
