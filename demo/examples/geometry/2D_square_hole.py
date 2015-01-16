import pyLBM

if __name__ == "__main__":
    dgeom = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':{0:{'element':pyLBM.Circle((0.5,0.5), 0.125), 'label':1, 'del':0}},
    }
    geom = pyLBM.Geometry(dgeom)
    geom.visualize()
