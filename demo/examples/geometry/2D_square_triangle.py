import pyLBM

if __name__ == "__main__":
    dgeom = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':[pyLBM.Triangle((0.,0.), (0.,.5), (.5, 0.), label = 1)],
    }
    geom = pyLBM.Geometry(dgeom)
    geom.visualize(viewlabel=True)
