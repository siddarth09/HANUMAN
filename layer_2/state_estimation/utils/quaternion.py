# Math utilities for state estimation 

import numpy as np 
from typing import Tuple 


def skew_symmetric(v:np.ndarray)->np.ndarray:
    return np.array([
        [0, -v[2],v[1]],
        [v[2],0,-v[0]],
        [-v[1],v[0],0]
    ])