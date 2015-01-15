import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 2], 'y': [0, 1], 'label': [0,1,2,3]},
        'space_step':0.2,
        'number_of_schemes':1,
        0:{'velocities':range(9)},
    }
    dom = pyLBM.Domain(dico)
    pyLBM.domain.visualize(dom, opt=1)

    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':{0:{'element':pyLBM.Circle((0.5,0.5), 0.2), 'label':1, 'del':0}},
        'space_step':0.1,
        'number_of_schemes':1,
        0:{'velocities':range(9)},
    }
    dom = pyLBM.Domain(dico)
    pyLBM.domain.visualize(dom, opt=1)
