from typing import TypeVar
from enum import Enum
import numpy as np
import torch

_ArrT = TypeVar("_ArrT", torch.Tensor, np.ndarray)

class MatrixType(Enum):
    CSC = 0
    CSR = 1
    COO = 2

class CholeskySolverD:
    def __init__(self, n_rows: int, ii: _ArrT, jj: _ArrT, x: _ArrT, type: MatrixType):...
    def solve(self, b: _ArrT, x: _ArrT):...

class CholeskySolverF:
    def __init__(self, n_rows: int, ii: _ArrT, jj: _ArrT, x: _ArrT, type: MatrixType):...
    def solve(self, b: _ArrT, x: _ArrT):...