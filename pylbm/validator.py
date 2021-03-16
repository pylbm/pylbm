# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
"""
This module validates the dictionary.
"""
#pylint: disable=invalid-name
import sys
import logging
import types
from cerberus import Validator, TypeDefinition
from colorama import init, Fore, Style
import sympy

from .elements.base import Element
from .boundary import BoundaryMethod
from .algorithm import BaseAlgorithm

log = logging.getLogger(__name__) #pylint: disable=invalid-name
init(autoreset=True)

symbol_type = TypeDefinition('symbol', (sympy.Symbol, sympy.IndexedBase), ())
Validator.types_mapping['symbol'] = symbol_type

expr_type = TypeDefinition('expr', (sympy.Expr,), ())
Validator.types_mapping['expr'] = expr_type

matrix_type = TypeDefinition('matrix', (sympy.Matrix,), ())
Validator.types_mapping['matrix'] = matrix_type

element_type = TypeDefinition('element', (Element,), ())
Validator.types_mapping['element'] = element_type

function_type = TypeDefinition('function', (types.FunctionType,), ())
Validator.types_mapping['function'] = function_type

valid_prompt = lambda indent: '   | ' + ' '*indent
error_prompt = lambda indent: Fore.RED + '>>>| ' + ' '*indent + Fore.RESET
bright_error = lambda error: Style.BRIGHT +  str(error) + Style.RESET_ALL
missing_value = lambda value: '%s%s: ???%s\n'%(Fore.MAGENTA, value, Fore.RESET)

class MyValidator(Validator):
    """
    New Validator to check boundary methods.
    """
    def _validate_isboundary(self, isboundary, field, value):
        """ Test if value is a subclass of BoundaryMethod.
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if isboundary:
            if not isinstance(value, type) or not issubclass(value, BoundaryMethod):
                self._error(field, "Must be a BoundaryMethod")

    def _validate_isalgorithm(self, isalgorithm, field, value):
        """ Test if value is a subclass of BaseAlgorithm.
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if isalgorithm:
            if not isinstance(value, type) or not issubclass(value, BaseAlgorithm):
                self._error(field, "Must be a BaseAlgorithm")

def rec_list(mylist, errors, indent=0):
    """
    visit the list recursively and record the elements
    and the errors (if there exist) in a string.

    Parameters
    ----------

    mylist: list
        The list to print
    errors: dict
        the possible errors in this list
    indent : int
        the indentation to add for printing

    Returns
    -------

    str
        the string with the elements of the list and the errors

    """
    if isinstance(errors, str):
        return bright_error(errors) + '\n'

    s = ''
    for il, l in enumerate(mylist):
        if il not in errors.keys():
            prompt = valid_prompt
            new_errors = {}
            error_message = ''
        else:
            prompt = error_prompt
            new_errors = errors[il][0]
            error_message = bright_error(errors[il][0])

        s += prompt(indent)
        s += '%s:\n'%il
        indent += 4
        if isinstance(l, list):
            s += rec_list(l, new_errors, indent)
        elif isinstance(l, dict):
            s += rec_dict(l, new_errors, indent)
        else:
            s += prompt(indent)
            s += '%s %s\n'%(l, error_message)
        indent -= 4
    return s

def rec_dict(dico, errors, indent=0):
    """
    visit the dictionary recursively and record the elements
    and the errors (if there exist) in a string.

    Parameters
    ----------

    dico: dict
        The dictionary to print
    errors: dict
        the possible errors in this list
    indent : int
        the indentation to add for printing

    Returns
    -------

    str
        the string with the elements of the dictionary and the errors

    """
    if isinstance(errors, str):
        return bright_error(errors) + '\n'

    s = ''
    for key, value in dico.items():
        if key not in errors:
            prompt = valid_prompt
            new_errors = {}
            error_message = ''
        else:
            prompt = error_prompt
            new_errors = errors[key][0]
            error_message = bright_error(errors[key][0])

        s += prompt(indent)
        s += '%s: '%key
        indent += 4
        if isinstance(value, list) and isinstance(value[0], dict):
            s += '\n'
            s += rec_list(value, new_errors, indent)
        elif isinstance(value, dict):
            s += '\n'
            s += rec_dict(value, new_errors, indent)
        else:
            s += '%s %s\n'%(value, error_message)
        indent -= 4

    for key, value in errors.items():
        if key not in dico.keys():
            s += error_prompt(indent)
            s += missing_value(key)
    return s

def validate(dico, name):
    """
    Validation of the dictionary used in pylbm to
    instanciate Stencil, Domain, Geometry, Scheme and Simulation

    Parameters
    ----------

    dico : dict
        The dictionary to validate
    name : str
        The class name to validate ('Stencil', 'Domain', ...)

    """
    scheme = {'velocities': {'type': 'list',
                             'schema': {'type': 'integer', 'min': 0},
                             'required': name in ['Domain', 'Scheme', 'Simulation', 'Stencil']
                            },
              'M': {'type': 'matrix',
                    'required': name in ['Scheme', 'Simulation'],
                    'excludes': 'polynomials',
                    },
              'polynomials': {'type': 'list',
                              'schema': {'anyof_type': ['number', 'expr']},
                              'required': name in ['Scheme', 'Simulation'],
                              'excludes': 'M',
                             },
              'relaxation_parameters': {'type': 'list',
                                        'schema': {'anyof_type': ['number', 'expr']},
                                        'required': name in ['Scheme', 'Simulation']
                                       },
              'equilibrium': {'type': 'list',
                              'schema': {'anyof_type': ['number', 'expr']},
                              'excludes': 'feq',
                              'required': name in ['Scheme', 'Simulation']
                             },
              'feq': {'type': 'list',
                      'items': [{'type': 'function'}, {'type': 'list'}],
                      'excludes': 'equilibrium',
                      'required': name in ['Scheme', 'Simulation']
                     },
              'conserved_moments': {'anyof': [{'type': 'symbol'},
                                              {'type': 'list',
                                               'schema': {'type': 'symbol'}}
                                             ],
                                    'required': name in ['Scheme', 'Simulation']
                                   },
              'source_terms': {'type': 'dict',
                               'keysrules': {'type': 'symbol'},
                               'valuesrules': {'anyof': [{'type': 'expr'},
                                                         {'type': 'number'}]},
                              },
             }

    boundary = {'method': {'type': 'dict',
                           'keysrules': {'type': 'integer'},
                           'valuesrules': {'isboundary': True}
                          },
                'value': {'anyof': [{'type': 'function'},
                                    {'type': 'list',
                                     'items': [{'type': 'function'}, {'type': 'list', 'nullable': True}],
                                    }]
                         },
                'time_bc': {'type': 'boolean',
                            'default': False
                           }
               }

    simulation = {'dim': {'type': 'integer',
                          'allowed': [1, 2, 3],
                          'excludes': 'box',
                          'required': name in ['Stencil', 'Scheme']
                         },
                  'box': {'type': 'dict',
                          'schema': {'x': {'type': 'list',
                                           'items': [{'type': 'number'},
                                                     {'type': 'number'}
                                                    ]
                                          },
                                     'y': {'type': 'list',
                                           'items': [{'type': 'number'},
                                                     {'type': 'number'}
                                                    ]
                                          },
                                     'z': {'type': 'list',
                                           'items': [{'type': 'number'},
                                                     {'type': 'number'}
                                                    ]
                                          },
                                     'label': {'anyof': [{'type': 'integer'},
                                                         {'type' : 'list',
                                                          'schema': {'type': 'integer'}}
                                                        ]
                                              }
                                    },
                          'excludes': 'dim',
                          'required': name in ['Domain', 'Geometry', 'Scheme', 'Simulation', 'Stencil']
                         },
                  'elements': {'type': 'list',
                               'schema': {'type': 'element'}
                              },
                  'space_step': {'type': 'number',
                                 'min': 0,
                                 'required': name in ['Domain', 'Simulation']
                                },
                  'scheme_velocity': {'anyof_type': ['number', 'symbol'],
                                      'required': name in ['Scheme', 'Simulation']
                                     },
                  'schemes': {'type': 'list',
                              'schema': {'type': 'dict',
                                         'schema': scheme
                                        },
                              'required': name in ['Domain', 'Scheme', 'Simulation', 'Stencil']
                             },
                  'parameters': {'type': 'dict',
                                 'keysrules': {'type': 'symbol'},
                                 'valuesrules': {'anyof': [{'type': 'expr'},
                                                           {'type': 'number'}]},
                                },
                  'init': {'type': 'dict',
                           'keysrules': {'type': 'symbol'},
                           'valuesrules': {'anyof': [{'type': 'number'},
                                                     {'type': 'function'},
                                                     {'type': 'list',
                                                      'items': [{'type': 'function'}, {'type': 'list'}]}
                                                    ]
                                          },
                            'required': name in ['Simulation']
                            },
                  'boundary_conditions': {'type': 'dict',
                                          'keysrules': {'type': 'integer'},
                                          'valuesrules': {'schema': boundary},
                                         },
                  'relative_velocity': {'type': 'list',
                                        'schema': {'anyof_type': ['number', 'expr']}
                                       },
                  'generator': {'type': 'string',
                                'allowed':['numpy', 'cython', 'loopy']
                               },
                  'codegen_option':{'type': 'dict',
                                    'schema': {'directory': {'type': 'string'},
                                               'generate': {'type': 'boolean'}
                                              },
                                   },
                  'lbm_algorithm': {'type': 'dict',
                                    'schema': {'name': {'isalgorithm': True},
                                               'settings': {'type': 'dict'}
                                              }
                                   },
                  'show_code': {'type': 'boolean'}
                 }

    v = MyValidator(simulation)
    is_valid = v.validate(dico)
    message = rec_dict(v.document, v.errors)

    if is_valid:
        log.info('Check the dictionary for %s class', name)
        log.info('%s', message)
    else:
        log.error('Check the dictionary for %s class', name)
        log.error(message)
        sys.exit()
