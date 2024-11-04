#!/usr/bin/env python3
"""
"""

from cryptanalysislib.builder import Cryptanalysislib 

def test1():
    c = Cryptanalysislib()
    assert c.has_error() == False
    
    
    c.run("bench_subsetsum_tree", 
          "bench/subsetsum/bench_subsetsum_tree")
    assert c.has_error() == False
