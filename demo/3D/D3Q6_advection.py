import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM
from pyevtk.hl import imageToVTK
import pylab as plt

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

# advective velocity
ux, uy, uz = .5, .2, .1
# domain of the computation
xmin, xmax, ymin, ymax, zmin, zmax = 0., 1., 0., 1., 0., 1.

def initialization(x, y, z):
    xm, ym, zm = .5*(xmin+xmax), .5*(ymin+ymax), .5*(zmin+zmax)
    return .5*np.ones((x.size, y.size, z.size)) \
          + .5*(((x-xm)**2+(y-ym)**2+(z-zm)**2)<.25**2)

def plot(sol, num):
    sol.f2m()
    ix = 1+sol.m[0][0].shape[0]/2
    iy = 1+sol.m[0][0].shape[1]/2
    iz = 1+sol.m[0][0].shape[2]/2
    """
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(sol.m[0][0][:,:,iz])
    plt.title('Coupe z={0}'.format(sol.domain.x[2][iz]))
    plt.subplot(132)
    plt.imshow(sol.m[0][0][:,iy,:])
    plt.title('Coupe y={0}'.format(sol.domain.x[1][iy]))
    plt.subplot(133)
    plt.imshow(sol.m[0][0][ix,:,:])
    plt.title('Coupe x={0}'.format(sol.domain.x[0][ix]))
    plt.draw()
    plt.pause(1.e-3)
    """
    sol.save[:] = sol.m[0][0][1:-1,1:-1,1:-1]
    imageToVTK("./data/image_{0}".format(num), pointData = {"mass" : sol.save} )

s = 1.
la = 1.
dx = 1./128
d = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':-1},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[{
        'velocities': range(1,7),
        'polynomials': [1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2],
        'equilibrium': [u[0][0], ux*u[0][0], uy*u[0][0], uz*u[0][0], 0., 0.],
        'relaxation_parameters': [0., s, s, s, s, s],
        'init':{0:(initialization,),},
    },],
    'parameters': {LA: la},
    'generator': pyLBM.generator.CythonGenerator,
}
"""
s = pyLBM.Scheme(d)
print s
print s.generator.code
"""
sol = pyLBM.Simulation(d)
nx, ny, nz = sol.m[0][0].shape
sol.save = np.empty((nx-2, ny-2, nz-2))

im = 0
plot(sol,im)

while sol.t<1.:
    sol.one_time_step()
    print sol.t
    im += 1
    plot(sol,im)

sol.time_info()
