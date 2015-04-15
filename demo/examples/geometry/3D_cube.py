import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'z':[0, 1], 'label':range(6)},
    }
    geom = pyLBM.Geometry(dico)
    print geom
    geom.visualize(viewlabel=True)
