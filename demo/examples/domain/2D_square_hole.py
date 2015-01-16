import pyLBM

if __name__ == "__main__":
    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':[pyLBM.Circle((0.5,0.5), 0.2, label = 1)],
        'space_step':0.05,
        'schemes':[{'velocities':range(9)}]
    }
    dom = pyLBM.Domain(dico)
    dom.visualize(opt=0)
    dom.visualize(opt=1)
    pyLBM.domain.verification(dom, with_color = True)
