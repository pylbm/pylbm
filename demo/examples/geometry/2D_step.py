import pyLBM

if __name__ == "__main__":
    dgeom = {
        'box':{'x': [0, 3], 'y': [0, 1], 'label':[0, 1, 0, 2]},
        'elements':[pyLBM.Parallelogram((0.,0.), (.5,0.), (0., .5), label = 0)],
    }
    geom = pyLBM.Geometry(dgeom)
    geom.visualize()
