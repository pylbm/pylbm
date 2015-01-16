import pyLBM

if __name__ == "__main__":
    dgeom = {'box':{'x': [0, 1], 'label': [0,1]},}
    geom = pyLBM.Geometry(dgeom)
    geom.visualize(viewlabel = True)
