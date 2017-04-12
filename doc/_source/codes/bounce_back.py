from __future__ import print_function, division
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

L = 2
H = 2
color_in = 'b'
color_out = 'r'

fig = plt.figure(1, figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, aspect='equal')
ax.plot([0, 0], [-H, H-1], 'k-', linewidth = 2)
ax.add_patch(Rectangle((0, -H), -L, 2*H-1, alpha=0.1, fill=True, color=color_in))
# inner points
mesh_x = np.arange(-L,0) + 0.5
mesh_y = np.arange(-H,H-1) + 0.5
mesh_Y, mesh_X = np.meshgrid(mesh_y, mesh_x)
ax.scatter(mesh_X, mesh_Y, marker='o', color=color_in)
# outer points
mesh_x = np.arange(0,L-1) + 0.5
mesh_y = np.arange(-H,H-1) + 0.5
mesh_Y, mesh_X = np.meshgrid(mesh_y, mesh_x)
ax.scatter(mesh_X, mesh_Y, marker='s', color=color_out)
# inner arrows
e = 0.1
x, y = -0.5, -0.5
for i in [-1,0]:
    for j in [-1,0,1]:
        if i != 0 or j != 0:
            ax.arrow(x+i*(1-e), y+j*(1-e), -i*(1-2*e), -j*(1-2*e),
                      length_includes_head=True,
                      head_width=.5*e,
                      head_length=e,
                      fc=color_in,
                      ec=color_in)
# outer arrows
for j in [-1,0,1]:
    vx = np.array([x+e, x+0.5])
    vy = np.array([y+j*e, y+j*.5])
    ax.plot(vx, vy+.25*(1+.5*abs(j))*e, c=color_in)
    ax.arrow(vx[1], vy[1]-.25*(1+.5*abs(j))*e, -0.5+e, j*(e-.5),
             length_includes_head=True,
             head_width=.5*e,
             head_length=e,
             fc=color_out,
             ec=color_out)
    ax.plot([x+1-e, x+0.5], [y+(1-e)*j,y+.5*j], c=color_out, linestyle='--')
ax.axis('off')
plt.title("bounce back: the exiting particles bounce back without sign modification")
plt.show()
