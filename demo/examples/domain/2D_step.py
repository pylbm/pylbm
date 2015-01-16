import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 3], 'y': [0, 1], 'label':[0, 1, 0, 2]},
        'elements':[pyLBM.Parallelogram((0.,0.), (.5,0.), (0., .5), label=0)],
        'space_step':0.125,
        'schemes':[{'velocities':range(9)}]
    }
    dom = pyLBM.Domain(dico)
    dom.visualize(opt=0)
    dom.visualize(opt=1)
