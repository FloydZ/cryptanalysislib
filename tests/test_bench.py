#!/usr/bin/env python3
"""
"""

import pytest

from cryptanalysislib.helper import Range
from cryptanalysislib.builder import Cryptanalysislib
from cryptanalysislib.bench import Benchmarker
from cryptanalysislib.optimizers.subsetsum import SubSetSumOptimizerD2

@pytest.mark.integration
def test_bench():
    c = Cryptanalysislib()
    b = Benchmarker(c, SubSetSumOptimizerD2, 
                    "bench_subsetsum_tree", 
                    "bench/subsetsum/bench_subsetsum_tree", 
                    "./bench/subsetsum/params.h",
                    [Range("n", 22), Range("max_mem", 20, 22)]
                    )
