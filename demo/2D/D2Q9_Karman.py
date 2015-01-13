import sys
import h5py
import os
import os.path

import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

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

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

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


def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qx(x,y):
    return uo * np.ones((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def bc_rect(f, m, x, y, scheme):
    m[0][0] = 0.
    m[0][1] = rhoo*uo
    m[0][2] = 0.
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot_vorticity(sol, num):
    V = sol.m[0][2][2:,1:-1] - sol.m[0][2][0:-2,1:-1] - sol.m[0][1][1:-1,2:] + sol.m[0][1][1:-1,0:-2]
    V /= np.sqrt(V**2+1.e-5)
    plt.imshow(np.float32(V.transpose()), origin='lower', cmap=cm.gray)
    plt.title('Vorticity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.savefig("sauvegarde_images/Karman_{0:04d}.pdf".format(num))
    plt.pause(1.e-3)

def save_hdf5(sol, num):
    min, max = 0., 2*sol.domain.dx
    ccc = 1./(max-min)
    # Create a new group
    gr = file.create_group("/GR{0:04d}".format(num))
    gr.attrs["time"] = sol.t
    # Create a dataset under the GR_k group.
    print "Writing data for t={0:f}...".format(sol.t)
    sol.m[0][:,sol.indout[1], sol.indout[0]] = 1.
    dataset = gr.create_dataset("vorticity", data = ccc*np.abs(
        sol.m[0][2,2:,1:-1].astype(np.float32) - sol.m[0][2,:-2,1:-1].astype(np.float32)
        - sol.m[0][1,1:-1,2:].astype(np.float32) + sol.m[0][1,1:-1,:-2].astype(np.float32)
        - min))
    # Close the group
    del gr

def simu():
    # parameters
    NbImages = 500  # number of figures
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = 0., 5., 0., 3
    rayon = 0.125
    dx = 1./128 # spatial step
    la = 1. # velocity of the scheme
    Tf = 500
    mu   = 5.e-5 #0.00185
    zeta = 10*mu#3.e-3
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
        'elements':{0: {'element':Circle([0.75, 0.5*(ymin+ymax)+2*dx], rayon), 'label':2, 'del':0},
                    1: {'element':Circle([1.25, 0.5*(ymin+ymax)+2*dx+0.25], rayon), 'label':2, 'del':0},
                    2: {'element':Circle([1.25, 0.5*(ymin+ymax)+2*dx-0.25], rayon), 'label':2, 'del':0},},
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

    sol = pyLBMSimu.Simulation(dico)
    #sol.domain.geom.visualize()
    #sys.exit()
    sol.indout = np.where(sol.domain.in_or_out == sol.domain.valout)

    Re = rhoo*uo*2*rayon/mu
    print "Reynolds number {0:10.3e}".format(Re)

    #fig = plt.figure(0,figsize=(16, 8))
    #fig.clf()
    #plt.ion()
    im = 0
    #plot_vorticity(sol, im)
    save_hdf5(sol, im)
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    while (sol.t<Tf):
        sol.one_time_step()
        compt += 1
        if (compt%Ncompt==0):
            im += 1
            save_hdf5(sol, im)
            #plot_vorticity(sol, im)
    #plt.ioff()
    #plt.show()

class Canvas(app.Canvas):
    def __init__(self, file):
        coeff = 2
        self.imagecourante = 0
        self.gr = file["/GR{0:04d}".format(self.imagecourante)]
        W, H = self.gr['vorticity'].shape
        self.W, self.H = W, H
        # A simple texture quad
        self.data = np.zeros(4, dtype=[ ('a_position', np.float32, 2),
                                        ('a_texcoord', np.float32, 2) ])
        self.data['a_position'] = np.array([[0, 0], [coeff * W, 0], [0, coeff * H], [coeff * W, coeff * H]])
        self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        app.Canvas.__init__(self, close_keys='escape')
        self.title = "Solution t={0:f}".format(self.gr.attrs["time"])
        self.size = W * coeff, H * coeff
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.gr['vorticity'])
        self.texture.interpolation = gl.GL_LINEAR

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(self.data))

        self.projection = np.eye(4, dtype=np.float32)
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        self.timer = app.Timer(1./24)
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
        if event.text == 'r':
            self.imagecourante = 0
            self.gr = file["/GR{0:04d}".format(self.imagecourante)]
            self.timer.start()

    def on_timer(self, event):
        self.go_on()
        self.maj()

    def go_on(self):
        self.imagecourante += 1
        try:
            self.gr = file["/GR{0:04d}".format(self.imagecourante)]
        except:
            print 'End of the simulation'
            self.timer.stop()

    def maj(self):
        self.title = "Solution t={0:f}".format(self.gr.attrs["time"])
        self.texture.set_data(self.gr['vorticity'])
        self.update()        

if __name__ == "__main__":
    compute = False
    visu = True
    #name = "Karman_Re=125.hdf5"
    name = "Karman_Re=250.hdf5"
    #name = "Karman_Triple.hdf5"
    if compute:
        rhoo = 1.
        uo = 0.05
        # Create a new file using defaut properties.
        if os.path.isfile(name):
            os.remove(name)
        file = h5py.File(name, 'w')
        file.attrs['title'] = 'Von_Karman_alley'
        simu()
        file.close()
    if visu:
        file = h5py.File(name, 'r')
        print 'Visualization of the file: {0}'.format(file.attrs['title'])
        c = Canvas(file)
        c.show()
        app.run()
        file.close()
