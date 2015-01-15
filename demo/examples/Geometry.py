import pyLBM

solid, fluid = 0, 1
Carre = pyLBM.Parallelogram((.1, .1), (.8, 0), (0, .8))
Bande = pyLBM.Parallelogram((0, .4), (1, 0), (0, .2))
Cercle = pyLBM.Circle((.5, .5), .25)
petit_Carre = pyLBM.Parallelogram((.4, .5), (.1, .1), (.1, -.1))
dgeom = {
    'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
    'elements':{
        0:{'element':Carre, 'del':solid},
        1:{'element':Bande, 'del':fluid},
        2:{'element':Cercle, 'del':fluid},
        3:{'element':petit_Carre, 'del':solid},
    }
}
geom = pyLBM.Geometry(dgeom)
geom.visualize(tag=False, viewlabel=False)
"""
# arrondir les angles
geom.add_elem(Parallelogram((0.1, 0.9), (0.05, 0), (0, -0.05)), 0, fluid)
geom.add_elem(Circle((0.15, 0.85), 0.05), 0, solid)
geom.add_elem(Parallelogram((0.1, 0.1), (0.05, 0), (0, 0.05)), 0, fluid)
geom.add_elem(Circle((0.15, 0.15), 0.05), 0, solid)
geom.add_elem(Parallelogram((0.9, 0.9), (-0.05, 0), (0, -0.05)), 0, fluid)
geom.add_elem(Circle((0.85, 0.85), 0.05), 0, solid)
geom.add_elem(Parallelogram((0.9, 0.1), (-0.05, 0), (0, 0.05)), 0, fluid)
geom.add_elem(Circle((0.85, 0.15), 0.05), 0, solid)
geom.visualize()
"""
