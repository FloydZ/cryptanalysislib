#!/usr/bin/env python3
""" simple test """

import os
from cryptanalysislib.optimizers.subsetsum import SubSetSumOptimizerD2
from cryptanalysislib.optimizers.optimizers import MetaOptimizer
from cryptanalysislib.optimizers.helper import Range

def test1():
    s = MetaOptimizer(SubSetSumOptimizerD2, 
                      [Range("n", 32, 40), 
                       Range("max_mem", 0, 32)])
    for v in s:
        print(v)

s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32)])
for v in s:
    print(v)
#s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 10)])
#s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 10, 2)])
#s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 10, 2), Range("max_mem", 20, 22)])
#s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 10, 2), Range("max_mem", 22, 19, 2)])
