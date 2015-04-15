import pyLBM
dico = {
    'box':{'x': [0, 2], 'y': [0, 2], 'z':[0, 2], 'label':range(6)},
    'space_step':1,
    'schemes':[{'velocities':range(19)}]
}
dom = pyLBM.Domain(dico)
print dom
dom.visualize(opt=1)
