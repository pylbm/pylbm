import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 1], 'label': [0,1]},
        'space_step':0.1,
        'number_of_schemes':1,
        0:{'velocities':range(3)},
    }
    dom = pyLBM.Domain(dico)
    dom.visualize()
