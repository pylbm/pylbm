# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from tokenize import (generate_tokens, untokenize, TokenError,
    NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN)
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from ..symbolic import set_order

def reorder(tokens, local_dict, global_dict):
    result = []
    tokens_ordered = set_order(tokens, local_dict['sorder'])
    for token in tokens_ordered:
        for t in token:
            result.append(t)
        result.append((OP, ','))
    return result

def transform_expr(tokens, local_dict, global_dict):
    result = []
    l = 0
    while l < len(tokens) - 1:
        token = tokens[l]
        nextTok = tokens[l+1]
        if token[0] == NAME and token[1] in local_dict['consm'].keys():
            if nextTok[0] == OP and nextTok[1] == '[':
                result.append((NAME, 'm'))
                result.append(nextTok)
                stacks = [[(NUMBER, '%s'%local_dict['consm'][token[1]])]]
                stacks.append([])
                l += 2
                token = tokens[l]
                while token[1] != ']':
                    if token[1] == ',':
                        stacks.append([])
                    else:
                        stacks[-1].append(token)
                    l += 1
                    token = tokens[l]
                result.extend(reorder(stacks, local_dict, global_dict))
                result.append(token)
            else:
                result.append((NAME, 'm'))
                result.append((OP, '['))
                default_index = [local_dict['consm'][token[1]]] + local_dict['default_index']
                if local_dict['sorder']:
                    index_ordered = set_order(default_index, local_dict['sorder'])
                else:
                    index_ordered = default_index
                for index in index_ordered:
                    if isinstance(index, sp.Idx):
                        result.append((NAME, '%s'%index))
                    else:
                        result.append((NUMBER, '%s'%index))
                    result.append((OP, ','))
                result.append((OP, ']'))
        else:
            result.append(token)
        l += 1
    while l < len(tokens):
        result.append(tokens[l])
        l += 1
    return result

def parse_expr(expr, user_local_dict):
    local_dict = user_local_dict
    for s in expr.atoms(sp.Symbol):
        local_dict[s.name] = s

    transformations = (standard_transformations + (transform_expr,))
    return sympy_parse_expr(expr.__str__(), local_dict=local_dict, transformations=transformations)