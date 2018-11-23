

import sympy as sp

def equivalent_equation(scheme):
    t, x, y, z, f = sp.symbols("t x y z f", type='real')
    consm = list(scheme.consm.keys())
    nconsm = len(scheme.consm)

    dx = sp.Derivative(f, x)
    dy = sp.Derivative(f, y)
    dz = sp.Derivative(f, z)
    dspace = [dx, dy, dz]
    dt = sp.Derivative(f, t)
    LA = scheme.symb_la
    if not LA:
        LA = scheme.la

    func = []
    for i in range(nconsm):
        func.append(sp.Function('f{}'.format(i))(t, x))

    sublist = [(i, j) for i, j in zip(consm, func)]
    sublist_inv = [(j, i) for i, j in zip(consm, func)]
    eq_func = sp.Matrix(scheme.EQ[nconsm:]).subs(sublist)
    s = sp.Matrix(scheme.s[nconsm:])

    source = {}
    for sterm in scheme._source_terms:
        if sterm:
            for k, v in sterm.items():
                source[k] = v

    vd = []
    all_vel = scheme.stencil.get_all_velocities()
    for p in scheme.permutations:
        all_vel[p] = all_vel[p[-1::-1]]
    for vv in all_vel:
        som = 0
        for i, vvv in enumerate(vv):
            som += vvv*LA*dspace[i]
        vd.append(som)
    vd = sp.diag(*vd)

    Gamma = scheme.M*vd*scheme.invM

    A, B = sp.Matrix([Gamma[:nconsm, :nconsm]]), sp.Matrix([Gamma[:nconsm, nconsm:]])
    C, D = sp.Matrix([Gamma[nconsm:, :nconsm]]), sp.Matrix([Gamma[nconsm:, nconsm:]])

    gamma_1 = sp.zeros(A.shape[0], 1)
    for i in range(gamma_1.shape[0]):
        dummy = 0
        for j in range(A.shape[1]):
            dummy += A[(i,j)].replace(f, func[j])
        gamma_1[i] = dummy
        
        dummy = 0
        for j in range(B.shape[1]):
            dummy += B[(i, j)].replace(f, eq_func[j])
        gamma_1[i] += dummy 
        
    sigma = sp.diag(*s).inv() - sp.eye(len(s))/2

    phi1 = sp.zeros(C.shape[0], 1)
    for i in range(phi1.shape[0]):
        dummy = 0
        for j in range(C.shape[1]):
            dummy += C[(i,j)].replace(f, func[j])
        phi1[i] = -dummy
        
        dummy = 0
        for j in range(D.shape[1]):
            dummy += D[(i, j)].replace(f, eq_func[j])
        phi1[i] -= dummy  

    delta1 = sp.zeros(phi1.shape[0], 1)
    meq = sp.Matrix(scheme.EQ[nconsm:])
    print(consm)
    jac = meq.jacobian(consm)
    jac = jac.subs(sublist)
    for i in range(delta1.shape[0]):
        for j in range(jac.shape[1]):
            delta1[i] += jac[(i, j)]*gamma_1[j]

    phi1 += delta1
    sphi1 = sigma*phi1
    gamma_2 = sp.zeros(B.shape[0], 1)

    for i in range(gamma_2.shape[0]):
        dummy = 0
        for j in range(B.shape[1]):
            dummy += B[(i,j)].replace(f, sphi1[j])
        gamma_2[i] = dummy

    time = sp.zeros(A.shape[0], 1)
    for i in range(time.shape[0]):
        time[i] = dt.replace(f, func[i])

    res = (time + gamma_1 + gamma_2).doit().subs(sublist_inv)
    for i, c in enumerate(consm):
        res[i] = sp.Eq(res[i], source.get(c, 0))

    return {'full': res,
            'order1': gamma_1.subs(sublist_inv),
            'order2': gamma_2.subs(sublist_inv),
    }
