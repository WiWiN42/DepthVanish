# -*- encoding: utf-8 -*-
"""
@File    :   aanet_model.py
@Time    :   2025/04/28 21:26:23
@Author  :   yxing
"""


import os, sys
import torch, torchvision
from argparse import Namespace
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join('model', 'unimatch'))
sys.path.insert(0, os.path.join('model', 'unimatch', 'core'))
from unimatch.unimatch import UniMatch

class GMStereoModel(object):
    '''
    Wrapper class for IGEVStereo model
    '''