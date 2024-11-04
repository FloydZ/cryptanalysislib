#!/usr/bin/env python3
""" simple test """

from cryptanalysislib.optimizers.subsetsum import SubSetSumOptimizerD2
from cryptanalysislib.optimizers.optimizers import MetaOptimizer
from cryptanalysislib.optimizers.helper import Range

def test1():
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32)])
    l = list(s);
    assert len(l[0]) == 1
    assert l[0]["max_mem"] == 20


def test2():
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 20)])
    assert len(list(s)) == 9 # n=27,29,31, missing


def test3():
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 30,), Range("max_mem", 20)])
    l = list(s)
    assert len(l) == 1
    assert l[0]["max_mem"] == 20


def test4():
    i = 0
    c = 0
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 30, 33, 2), Range("max_mem", 20, 30)])
    for v in s:
        assert v["max_mem"] == 20 + c 
        i += 1
        if i == 2:
            c += 1 
            i = 0


def test5():
    i = 0
    c = 0
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 30, 33, 2), Range("max_mem", 30, 20)])
    for v in s:
            assert v["max_mem"] == 30 - c 
            i += 1
            if i == 2:
                c += 1 
                i = 0


def test6():
    i = 0
    c = 0
    s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 30, 33, 2), Range("max_mem", 30, 20, 2)])
    for v in s:
        print (v)
        assert v["max_mem"] == 30 - c 
        i += 1
        if i == 2:
            c += 2
            i = 0
