from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from six.moves import range
from six import string_types
import types
import sympy as sp
import numpy as np
import pylbm


class PrintInColor(object):
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHT_PURPLE = '\033[94m'
    PURPLE = '\033[95m'
    END = '\033[0m'

    @classmethod
    def error(cls, s):
        return cls.RED + str(s) + cls.END

    @classmethod
    def correct(cls, s):
        return cls.END + str(s) + cls.END

    @classmethod
    def missing(cls, s):
        return cls.PURPLE + str(s) + cls.END

    @classmethod
    def unknown(cls, s, b):
        if b:
            return cls.correct(s)
        else:
            return cls.error(s)

def space(ntab):
    return "    "*ntab

def debut(b):
    if b:
        return PrintInColor.correct("\n   |")
    else:
        return PrintInColor.error("\n>>>|")

def is_dico_generic(d, ltk, ltv, ntab=0):
    test = isinstance(d, dict)
    if test:
        ligne = ''
        for k, v in list(d.items()):
            test_k = True
            if isinstance(k, ltk):
                ligne_k = PrintInColor.correct(k)
            else:
                test_k = False
                ligne_k = PrintInColor.error(k)
            ligne_k += ": "
            if isinstance(v, ltv):
                ligne_k += PrintInColor.correct(v)
            else:
                test_k = False
                ligne_k += PrintInColor.error(v)
            ligne += debut(test_k) + space(ntab) + ligne_k
            test = test and test_k
    else:
        ligne = ''
    return test, ligne

def is_list_generic(l, lte, size=None):
    test = isinstance(l, (list, tuple))
    ligne = ''
    if test:
        ligne += '['
        compt = 0
        if size is not None and len(l) != size:
            test_s = False
            test = False
        else:
            test_s = True
        for e in l:
            if isinstance(e, lte) and test_s:
                ligne += PrintInColor.correct(e)
            else:
                test = False
                ligne += PrintInColor.error(e)
            compt += 1
            if compt < len(l):
                ligne += ', '
        ligne += '],'
    return test, ligne

def is_dico_sp_float(d, ntab=0):
    return is_dico_generic(d, (sp.Symbol, string_types), (int, float), ntab=ntab)

def is_dico_sp_sporfloat(d, ntab=0):
    return is_dico_generic(d, (sp.Symbol, string_types), (int, float, sp.Symbol, string_types), ntab=ntab)

def is_dico_int_func(d, ntab=0):
    return is_dico_generic(d, int, types.FunctionType, ntab=ntab)

def is_dico_box(d, ntab=0):
    return test_dico_prototype(d, pylbm.geometry.proto_box, ntab=ntab)

def is_dico_bc(d, ntab=0):
    test = isinstance(d, dict)
    ligne = ''
    if test:
        for label, dico_bc_label in list(d.items()):
            if not isinstance(label, (int, string_types)):
                test = False
                debut_l = debut(False) + space(ntab)
                ligne_l = PrintInColor.error(label) + ": "
            else:
                debut_l = debut(True) + space(ntab)
                ligne_l = PrintInColor.correct(label) + ": "
                if isinstance(dico_bc_label, dict):
                    test_lk, ligne_lk = test_dico_prototype(dico_bc_label, pylbm.boundary.proto_bc, ntab=ntab+1)
                    if not test_lk:
                        debut_l = debut(False) + space(ntab)
                    ligne_l += ligne_lk
                    test = test and test_lk
                else:
                    test = False
                    debut_l = debut(False) + space(ntab)
                    ligne_l += PrintInColor.error(dico_bc_label) + "\n"
            ligne += debut_l + ligne_l
    return test, ligne

def is_dico_init(d, ntab=0):
    return is_dico_generic(d, (sp.Symbol, string_types, int), (tuple, int, float), ntab=ntab)

def is_dico_sources(d, ntab=0):
    return is_dico_generic(d, (sp.Symbol, string_types), (tuple, int, float, sp.Expr, string_types), ntab=ntab)

def is_dico_stab(d, ntab=0):
    return test_dico_prototype(d, pylbm.scheme.proto_stab, ntab=ntab)

def is_dico_cons(d, ntab=0):
    return test_dico_prototype(d, pylbm.scheme.proto_cons, ntab=ntab)

def is_dico_bcmethod(d, ntab=0):
    test = isinstance(d, dict)
    ligne = ''
    if test:
        for label, value in list(d.items()):
            if not isinstance(label, int):
                test_l = False
                debut_l = debut(False) + space(ntab)
                ligne_l = PrintInColor.error(label) + ": "
            else:
                test_l = True
                debut_l = debut(True) + space(ntab)
                ligne_l = PrintInColor.correct(label) + ": "
                try:
                    if issubclass(value, pylbm.boundary.Boundary_method):
                        test_l = True
                        ligne_l = PrintInColor.correct(value)
                    else:
                        test_l = False
                        ligne_l = PrintInColor.error(value)
                except:
                    test_l = False
                    ligne_l = PrintInColor.error(value)
            test = test and test_l
            ligne += debut_l + ligne_l + "\n"
    return test, ligne

def is_list_sch(l, ntab=0):
    test = isinstance(l, (list, tuple))
    ligne = ''
    if test:
        compt = 0
        for sch in l:
            if isinstance(sch, dict):
                test_l, ligne_l = test_dico_prototype(sch, pylbm.scheme.proto_sch, ntab=ntab+1)
            else:
                test_l = False
                ligne_l = PrintInColor.error(sch)
            ligne += debut(test_l)
            ligne += space(ntab) + '{0}:'.format(compt) + ligne_l
            compt += 1
            test = test and test_l
    return test, ligne

def is_list_sch_dom(l, ntab=0):
    test = isinstance(l, (list, tuple))
    ligne = ''
    if test:
        compt = 0
        for sch in l:
            if isinstance(sch, dict):
                test_l, ligne_l = test_dico_prototype(sch, pylbm.scheme.proto_sch_dom, ntab=ntab+1)
            else:
                test_l = False
                ligne_l = PrintInColor.error(sch)
            ligne += debut(test_l)
            ligne += space(ntab) + '{0}:'.format(compt) + ligne_l
            compt += 1
            test = test and test_l
    return test, ligne

def is_list_int(l, ntab=None):
    return is_list_generic(l, int)

def is_list_int_or_string(l, ntab=None):
    return is_list_generic(l, (int, string_types))

def is_list_float(l, ntab=None):
    return is_list_generic(l, (int, float))

def is_2_list_int_or_float(l, ntab=None):
    return is_list_generic(l, (int, float), size=2)

def is_list_string_or_tuple(l, ntab=None):
    return is_list_generic(l, (tuple, string_types))

def is_generator(d, ntab=None):
    try:
        test = d.upper() in ["NUMPY", "CYTHON", "LOOPY"]
    except:
        test = False
    return test, PrintInColor.unknown(d, test)

def is_ode_solver(d, ntab=None):
    try:
        test = issubclass(d, pylbm.generator.ode_schemes.ode_solver)
    except:
        test = False
    return test, PrintInColor.unknown(d, test)

def is_list_elem(l, ntab=None):
    return is_list_generic(l, pylbm.elements.base.Element)

def is_list_sp(l, ntab=None):
    return is_list_generic(l, (sp.Expr, string_types))

def is_list_sp_or_nb(l, ntab=None):
    return is_list_generic(l, (int, float, sp.Expr, string_types))

def is_list_symb(l, ntab=None):
    return is_list_generic(l, (sp.Symbol, string_types))

def test_dico_prototype(dico, proto, ntab=0):
    test_g = True
    aff = ''
    for key, value in list(dico.items()):
        value_p = proto.get(key, None)
        test_loc = False
        if value_p is None:
            aff_k = PrintInColor.error(key) + ": "
        else:
            aff_k = PrintInColor.correct(key)+ ": "
            for vpk in value_p:
                if isinstance(vpk, type):
                    if isinstance(value, vpk):
                        aff_k += str(value)
                        test_loc = True
                        break
                elif isinstance(vpk, types.FunctionType):
                    testk, strk = vpk(value, ntab=ntab+1)
                    aff_k += strk
                    if testk:
                        test_loc = True
                        break
                else:
                    print("\n\n" + "*"*50 + "\nUnknown type:", vpk, "\n" + "*"*50)
        aff += debut(test_loc) + space(ntab) + aff_k
        test_g = test_g and test_loc
    for key_p, value_p in list(proto.items()):
        value = dico.get(key_p, None)
        if value is None:
            if value_p[0] == type(None):
                aff += debut(True) + space(ntab) + PrintInColor.correct(key_p) + ': None'
            else:
                aff += debut(False) + space(ntab) + PrintInColor.missing(str(key_p) + ': ???')
                test_g = False
    return test_g, aff

def test_compatibility_dim(dico):
    test = True
    aff = ''
    dim = dico.get('dim', None)
    dbox = dico.get('box', None)
    if (dbox is not None) and (dim is not None):
        dx = dbox.get('x', None)
        dy = dbox.get('y', None)
        dz = dbox.get('z', None)
        if dim == 1:
            if (dx is None) or (dy is not None) or (dz is not None):
                aff += PrintInColor.error("The dimension 1 is not compatible with the box.\n")
                test = False
        if dim == 2:
            if (dx is None) or (dy is None) or (dz is not None):
                aff += PrintInColor.error("The dimension 2 is not compatible with the box.\n")
                test = False
        if dim == 3:
            if (dx is None) or (dy is None) or (dz is None):
                aff += PrintInColor.error("The dimension 3 is not compatible with the box.\n")
                test = False
    return test, aff

def test_compatibility_schemes(dico):
    test = True
    aff = ''
    lds = dico.get('schemes', None)
    inittype = dico.get('inittype', 'moments')
    if lds is not None:
        for ds in lds: # loop over the schemes
            # test over the length of the lists
            v = ds.get('velocities', [])
            n = len(v)
            for k in ['polynomials', 'equilibrium', 'relaxation_parameters']:
                kk = ds.get(k, None)
                if kk is None:
                    aff += PrintInColor.missing("The key '" + k + "' is not given.\n")
                elif len(kk) != n:
                    aff += PrintInColor.error("The size of the list '" + k + "' is not valid.\n")
                    test = False
            # test over the conserved moments
            cm = ds.get('conserved_moments', None)
            ceq = ds.get('equilibrium', None)
            crp = ds.get('relaxation_parameters', None)
            if cm is not None and ceq is not None and crp is not None:
                if not isinstance(cm, list):
                    cm = [cm,]
                for cmk in cm:
                    search_cmk = False
                    for l in range(len(ceq)):
                        if (sp.simplify(ceq[l] - cmk) == 0) and (crp[l] == 0.):
                            search_cmk = True
                    if not search_cmk:
                        aff += PrintInColor.error("The moment " + str(cmk) + " is not conserved.\n")
                        test = False
            # test if the conserved moments are initialized
            dsi = ds.get('init', None)
            if (inittype == 'moments') and (dsi is not None):
                for cmk in cm:
                    test_init = False
                    for ki, vi in list(dsi.items()):
                        if cmk == ki:
                            test_init = True
                            if not isinstance(vi, (float, int, tuple)):
                                aff += PrintInColor.error("Bad initialisation of " + str(cmk) + ".")
                    if not test_init:
                        print("Warning: the moment " + str(cmk) + " is not initialized.\n")
            # test if the conserved moments are initialized
            dsi = ds.get('init', None)
            if (inittype == 'moments') and (dsi is not None):
                for ki in list(dsi.keys()):
                    test_init = False
                    for cmk in cm:
                        if cmk == ki:
                            test_init = True
                    if not test_init:
                        print("Warning: the initialization of " + str(ki) + " is not valid.\n")
    return test, aff

def test_compatibility_bc(dico):
    test = True
    aff = ''
    dbox = dico.get('box', None)
    if dbox is not None:
        labels = dbox.get('label', [])
        if isinstance(labels, int):
            labels = [labels,]
        if len(labels) == 0:
            aff += PrintInColor.correct("No label given in the dictionary: default is -1 for periodic.\n")
        else:
            if any(l!=-1 for l in labels):
                dbc = dico.get('boundary_conditions', None)
                if dbc is None:
                    aff += PrintInColor.error("No boundary condition given in the dictionary.\n")
                    test = False
                for l in labels:
                    test_l = (l==-1) or any(k==l for k in list(dbc.keys()))
                    if not test_l:
                        test = False
                        aff += PrintInColor.error("The label {0} has no corresponding boundary condition.\n".format(l))
    return test, aff

def validate(dico, proto, test_comp = True):
    aff = "\n" + "*"*75
    aff += "\nTest of the dictionary\n"
    aff += "*"*75
    test, aff_d = test_dico_prototype(dico, proto)
    aff += aff_d
    if test and test_comp:
        test_c1, aff_c1 = test_compatibility_dim(dico)
        test_c2, aff_c2 = test_compatibility_schemes(dico)
        test_c3, aff_c3 = test_compatibility_bc(dico)
        test = test_c1 and test_c2 and test_c3
        if not test:
            aff += '\n' + '-'*60 + '\n'
            aff += aff_c1
            aff += aff_c2
            aff += aff_c3
            aff += '-'*60 + '\n'
        else:
            aff += '\n'
    else:
        aff += '\n'
    aff += "*"*75 + '\n'
    return test, aff

if __name__ == "__main__":

    rho, LA, X, Y = sp.symbols('rho, LA, X, Y')
    qx, qy = sp.symbols('qx, qy')
    rhoo, ux, uy = 1., 0.1, 0.2

    def fin(x):
        return x

    dico = {
        'box':{'x':(0., 1.), 'y':[0,1], 'label':[0, 'out', 0, 0]},
        'dim':1,
        'space_step':1.,
        'generator':pylbm.generator.CythonGenerator,
        'scheme_velocity':1.,
        'schemes':[{
            'velocities':list(range(1,5)),
            'conserved_moments':rho,
            'polynomials':[1, X, Y, X**2-Y**2, 2],
            'equilibrium':[rho, ux*rho, uy*rho, 0.],
            'relaxation_parameters':[0., 1., 1.],
            'init':{rho:1.,},
        }],
        'parameters':{LA:1.},
        'stability':{
            'linearization':{rho: rhoo,},
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
        'boundary_conditions':{
            0:{'method':{0:pylbm.bc.anti_bounce_back}, 'value':fin},
            'in':{'method':{0:pylbm.bc.neumann}, 'value':None},
        },
    }

    test, aff = validate(dico, pylbm.simulation.proto_simu)
    print(aff)
