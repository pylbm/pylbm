import pyLBM

if __name__ == "__main__":
    solid, fluid = 0, 1
    square = pyLBM.Parallelogram((.1, .1), (.8, 0), (0, .8))
    strip = pyLBM.Parallelogram((0, .4), (1, 0), (0, .2))
    circle = pyLBM.Circle((.5, .5), .25)
    inner_square = pyLBM.Parallelogram((.4, .5), (.1, .1), (.1, -.1))
    dgeom = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':{
            0:{'element':square, 'del':solid},
            1:{'element':strip, 'del':fluid},
            2:{'element':circle, 'del':fluid},
            3:{'element':inner_square, 'del':solid},
        }
    }
    geom = pyLBM.Geometry(dgeom)
    geom.visualize()
    # rounded inner angles
    geom.add_elem(pyLBM.Parallelogram((0.1, 0.9), (0.05, 0), (0, -0.05)), 0, fluid)
    geom.add_elem(pyLBM.Circle((0.15, 0.85), 0.05), 0, solid)
    geom.add_elem(pyLBM.Parallelogram((0.1, 0.1), (0.05, 0), (0, 0.05)), 0, fluid)
    geom.add_elem(pyLBM.Circle((0.15, 0.15), 0.05), 0, solid)
    geom.add_elem(pyLBM.Parallelogram((0.9, 0.9), (-0.05, 0), (0, -0.05)), 0, fluid)
    geom.add_elem(pyLBM.Circle((0.85, 0.85), 0.05), 0, solid)
    geom.add_elem(pyLBM.Parallelogram((0.9, 0.1), (-0.05, 0), (0, 0.05)), 0, fluid)
    geom.add_elem(pyLBM.Circle((0.85, 0.15), 0.05), 0, solid)
    geom.visualize()
