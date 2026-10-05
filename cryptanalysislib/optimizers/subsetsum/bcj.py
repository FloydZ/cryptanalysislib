
#!/usr/bin/env python3
""" 
"""

from math import inf
from cryptanalysislib.optimizers.helper import multiH, reps
from cryptanalysislib.optimizers.optimizers import Optimizer

class BCJ(Optimizer):
    """
    optimizer for the becker coron joux algorithm
    """

    def __init__(self, n: int, w: int, max_mem: int) -> None:
        super().__init__()

        self.n = n
        self.w = w
        self.max_mem = max_mem

    def opt(self):
        T = inf 
        # TODO
        return False, {}
