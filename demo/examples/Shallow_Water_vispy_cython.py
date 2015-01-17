import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros

import pyLBM
import pyLBM.simulation as pyLBMSimu
import pyLBM.generator as pyLBMGen

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
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
    //float c = texture2D(u_texture, v_texcoord).r;
    //gl_FragColor = vec4(c, 0., 1.-c, 1.);
}
"""

class Canvas(app.Canvas):

    def __init__(self, dico):
        coeff = 3
        self.sol = pyLBMSimu.Simulation(dico, nv_on_beg=False)
        H, W = self.sol._m.shape[:-1]
        W -= 2
        H -= 2
        self.W, self.H = W, H
        # A simple texture quad
        self.data = np.zeros(4, dtype=[ ('a_position', np.float32, 2),
                                        ('a_texcoord', np.float32, 2) ])
        self.data['a_position'] = np.array([[0, 0], [coeff * W, 0], [0, coeff * H], [coeff * W, coeff * H]])
        self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        app.Canvas.__init__(self)
        self.title = "Solution t={0:f}".format(0.)
        self.min, self.max = rhoo-0.75*deltarho, rhoo+1.1*deltarho
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
        for k in xrange(4):
            self.sol.one_time_step()

    def maj(self):
        self.title = "Solution t={0:f}".format(self.sol.t)
        self.sol.scheme.f2m(self.sol._F, self.sol._m)
        self.texture.set_data(self.ccc*(self.sol._m[1:-1, 1:-1, 0].astype(np.float32) - self.min))
        self.update()

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

def initialization_q(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

if __name__ == "__main__":
    # parameters
    dx = 1./256 # spatial step
    la = 4 # velocity of the scheme
    rhoo = 1.
    deltarho = 1.
    g = 1.

    Taille = 1.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    vitesse = range(1,5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1,-1,-1,-1]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype':'moments',
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 2., 2., 1.5],
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + 0.5*g*u[0][0]**2, u[1][0]*u[2][0]/u[0][0], 0.]),
                    'init':{0:(initialization_q,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + 0.5*g*u[0][0]**2, 0.]),
                    'init':{0:(initialization_q,)},
                    },
        ],
        # 'init':{'type':'moments',
        #         0:{0:(initialization_rho,)},
        #         1:{0:(initialization_q,)},
        #         2:{0:(initialization_q,)},
        #         },
        'generator': pyLBMGen.CythonGenerator,
        }

    c = Canvas(dico)
    c.show()
    app.run()
