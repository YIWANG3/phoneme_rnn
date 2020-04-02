from __future__ import absolute_import
from .BaseLine import *
from .Eval import *
from .FourLstm import *
from .A_Base import *
from .A_Enhanced import *
from .A_Base_GRU import *
from .FourGRU import *
from .A_Base_K5 import *

__factory = {
    'baseline': BaseLine,
    'eval': Eval,
    'fourlstm': FourLstm,
    'a_base': A_Base,
    'a_enhanced': A_Enhanced,
    'a_base_gru': A_Base_GRU,
    'fourgru': FourGRU,
    'a_base_k5': A_Base_K5
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
