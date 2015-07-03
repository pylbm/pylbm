import types
import sympy as sp
import numpy as np
import pyLBM


"""
TODO

- autoriser les string plutot que les symbols sympy
- tester que toutes les clés sont bien renseignées (sinon les mettre en rouge)
- faire les tests de compatiblités et mettre un message en rouge dans le cas contraire

"""
class PrintInColor:
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
    def unknown(cls, s, b):
        if b:
            return cls.correct(s)
        else:
            return cls.error(s)

def space(ntab):
    return "    "*ntab

def debut(b):
    if b:
        return PrintInColor.correct("\n|   ")
    else:
        return PrintInColor.error("\n|>>>")

def is_dico_generic(d, ltk, ltv, ntab=0):
    test = isinstance(d, types.DictionaryType)
    if test:
        ligne = ''
        for k, v in d.items():
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
            #if test_k:
            #    ligne += "\n|   "
            #else:
            #    ligne += "\n|>>>"
            #ligne += space(ntab) + ligne_k
    else:
        ligne = ''
    return test, ligne

def is_list_generic(l, lte, size=None):
    test = isinstance(l, (types.ListType, types.TupleType))
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
    return is_dico_generic(d, sp.Symbol, (types.IntType, types.FloatType), ntab=ntab)

def is_dico_int_func(d, ntab=0):
    return is_dico_generic(d, types.IntType, types.FunctionType, ntab=ntab)

def is_dico_box(d, ntab=0):
    return test_dico_prototype(d, proto_box, ntab=ntab)

def is_dico_bc(d, ntab=0):
    test = isinstance(d, types.DictionaryType)
    ligne = ''
    if test:
        for label, dico_bc_label in d.items():
            if not isinstance(label, (types.IntType, types.StringType)):
                test = False
                debut_l = debut(False) + space(ntab)
                ligne_l = PrintInColor.error(label) + ": "
            else:
                debut_l = debut(True) + space(ntab)
                ligne_l = PrintInColor.correct(label) + ": "
                if isinstance(dico_bc_label, types.DictionaryType):
                    test_lk, ligne_lk = test_dico_prototype(dico_bc_label, proto_bc, ntab=ntab+1)
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
    return is_dico_generic(d, (sp.Symbol, types.StringType), (types.TupleType, types.IntType, types.FloatType), ntab=ntab)

def is_dico_stab(d, ntab=0):
    return test_dico_prototype(d, proto_stab, ntab=ntab)

def is_list_sch(l, ntab=0):
    test = isinstance(l, (types.ListType, types.TupleType))
    ligne = ''
    if test:
        compt = 0
        for sch in l:
            if isinstance(sch, types.DictionaryType):
                test_l, ligne_l = test_dico_prototype(sch, proto_sch, ntab=ntab+1)
            else:
                test_l = False
                ligne_l = PrintInColor.error(sch)
            ligne += debut(test_l)
            ligne += space(ntab) + '{0}:'.format(compt) + ligne_l
            compt += 1
            test = test and test_l
    return test, ligne

def is_list_int(l, ntab=None):
    return is_list_generic(l, types.IntType)

def is_list_int_or_string(l, ntab=None):
    return is_list_generic(l, (types.IntType, types.StringType))

def is_list_float(l, ntab=None):
    return is_list_generic(l, (types.IntType, types.FloatType))

def is_2_list_int_or_float(l, ntab=None):
    return is_list_generic(l, (types.IntType, types.FloatType), size=2)

def is_generator(d, ntab=None):
    try:
        test = issubclass(d, pyLBM.generator.base.Generator)
    except:
        test = False
    return test, PrintInColor.unknown(d, test)

def is_list_sp(l, ntab=None):
    return is_list_generic(l, sp.Expr)

def is_list_sp_or_nb(l, ntab=None):
    return is_list_generic(l, (types.IntType, types.FloatType, sp.Expr))

def is_list_symb(l, ntab=None):
    return is_list_generic(l, sp.Symbol)

proto_simu = {
    'box':(is_dico_box,),
    'dim':(types.NoneType, types.IntType),
    'scheme_velocity':(types.IntType, types.FloatType),
    'parameters':(types.NoneType, is_dico_sp_float),
    'schemes':(is_list_sch,),
    'boundary_conditions':(types.NoneType, is_dico_bc),
    'generator':(types.NoneType, is_generator),
    'stability':(types.NoneType, is_dico_stab),
}

proto_box = {
    'x': (is_2_list_int_or_float,),
    'y': (types.NoneType, is_2_list_int_or_float,),
    'z': (types.NoneType, is_2_list_int_or_float,),
    'label': (types.NoneType, types.IntType, types.StringType, is_list_int_or_string),
}

proto_sch = {
    'velocities': (is_list_int,),
    'conserved_moments': (sp.Symbol, is_list_symb),
    'polynomials': (is_list_sp_or_nb,),
    'equilibrium': (is_list_sp_or_nb,),
    'relaxation_parameters': (is_list_float,),
    'init':(types.NoneType, is_dico_init),
}

proto_stab = {
    'linearization':(types.NoneType, is_dico_sp_float),
    'test_maximum_principle':(types.NoneType, types.BooleanType),
    'test_L2_stability':(types.NoneType, types.BooleanType),
}

proto_bc = {
    'method':(is_dico_int_func, ),
    'value':(types.NoneType, types.FunctionType),
}

def test_dico_prototype(dico, proto, ntab=0):
    test_g = True
    aff = ''
    for key, value in dico.items():
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
                    print "\n\n" + "*"*50 + "\nUnknown type\n" + "*"*50
        aff += debut(test_loc) + space(ntab) + aff_k
        test_g = test_g and test_loc
    return test_g, aff

if __name__ == "__main__":

    rho, LA, X, Y = sp.symbols('rho, LA, X, Y')
    qx, qy = sp.symbols('qx, qy')
    rhoo, ux, uy = 1., 0.1, 0.2

    def fin(x):
        return x

    dico = {
        'box':{'x':(0., 1.), 'y':[0,1,2], 'label':[0, 'out', 0, 0]},
        'dim':2,
        'generator':pyLBM.generator.CythonGenerator,
        'scheme_velocity':1.,
        'schemes':[{
            'velocities':range(1,5),
            'conserved_moments':[rho,1.],
            'polynomials':[1, X, Y, X**2-Y**2],
            'equilibrium':[rho, ux*rho, uy*rho, 0.],
            'relaxation_parameters':[0., 1., 1., 1.],
            'init':{rho:1.,},
        }],
        'parameters':{1.:LA},
        'stability':{
            'linearization':{rho: rhoo,},
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
        'boundary_conditions':{
            0:{'method':{0:pyLBM.bc.anti_bounce_back}, 'value':fin},
            'out':{'method':{0:pyLBM.bc.neumann}, 'value':None, 'toto':2},
        },
    }

    test, aff = test_dico_prototype(dico, proto_simu)
    if test:
        print PrintInColor.correct("The dictionary is valid.")
    else:
        print PrintInColor.error("The dictionary is not valid.")
    print aff
