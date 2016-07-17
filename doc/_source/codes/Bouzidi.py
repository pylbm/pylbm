from __future__ import print_function, division
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

color_in = 'b'
color_out = 'r'

d = .02
D = .075
e = 0.05
compt = 0
for s in [0.3, 0.7]:
    compt += 1
    fig = plt.figure(compt, figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(Rectangle((s, -.5), -1.5-s, 1, alpha=0.1, fill=True, color=color_in))
    ax.plot([s,s],[-.5,.5], 'k-', linewidth=2)
    ax.plot([-1,1],[-1,-1],'k-')
    ax.plot([-1,-1],[-1-d,-1+d], 'k-')
    ax.plot([0,0],[-1-d,-1+d], 'k-')
    ax.plot([1,1],[-1-d,-1+d], 'k-')
    ax.text(-.5,-1-D, r'$\Delta x=1$',
            verticalalignment = 'center',
            horizontalalignment = 'center',)
    ax.text(.5,-1-D, r'$\Delta x$',
            verticalalignment = 'center',
            horizontalalignment = 'center',)

    ax.plot([0,0],[-.75-d,-.75+d], 'k-')
    ax.plot([s,s],[-.75-d,-.75+d], 'k-')
    ax.plot([1,1],[-.75-d,-.75+d], 'k-')
    ax.text(s/2,-.75-D, r'$s$',
            verticalalignment = 'center',
            horizontalalignment = 'center',)
    ax.text((1+s)/2,-.75-D, r'$1-s$',
            verticalalignment = 'center',
            horizontalalignment = 'center',)
    if s<.5:
        ax.plot([-1,1],[-.75,-.75],'k-')
        ax.plot([-1,-1],[-.75-d,-.75+d], 'k-')
        ax.plot([2*s-1,2*s-1],[-.75-d,-.75+d], 'k-')
        ax.text(s-.5,-.75-D, r'$1-2s$',
                verticalalignment = 'center',
                horizontalalignment = 'center',)
        ax.text(s-1,-.75-D, r'$2s$',
                verticalalignment = 'center',
                horizontalalignment = 'center',)
        ax.plot([2*s-1,2*s-1],[-.75,.5],'k--')
    else:
        ax.plot([0,1],[-.75,-.75],'k-')
        ax.plot([2*s-1,2*s-1],[-.65,.5],'k--')
        ax.plot([0,s],[-.65,-.65],'k-')
        ax.plot([0,0],[-.65-d,-.65+d],'k-')
        ax.plot([s,s],[-.65-d,-.65+d],'k-')
        ax.plot([2*s-1,2*s-1],[-.65-d,-.65+d],'k-')
        ax.text((3*s-1)/2,-.65+D, r'$1{-}s$',
                verticalalignment = 'center',
                horizontalalignment = 'center',)
        ax.text(s-.5,-.65+D, r'$2s{-}1$',
                verticalalignment = 'center',
                horizontalalignment = 'center',)

    ax.scatter([-1,0],[0,0], marker='o', color=color_in)
    ax.scatter([1],[0], marker='s', color=color_out)

    ax.plot([1-e,s],[0,0],'--', color=color_out)
    ax.plot([2*s-1,s], [e, e], '-', color=color_in)
    ax.arrow(s, -e, -s, 0.,
             length_includes_head=True,
             head_width=.5*e,
             head_length=e,
             fc=color_out,
             ec=color_out)

    ax.axis('off')
    if s<.5:
        signe = "<"
    else:
        signe = ">"
    plt.title(r"Bouzidi condition: $s{0}1/2$".format(signe))
    plt.show()
