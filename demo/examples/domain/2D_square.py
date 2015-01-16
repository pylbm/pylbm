import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label': [0,1,2,3]},
        'space_step':0.1,
        'schemes':[{'velocities':range(9)}]
    }
    dom = pyLBM.Domain(dico)
    dom.visualize(opt=0)
    dom.visualize(opt=1)
