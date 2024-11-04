#!/usr/bin/env python3
""" simple test """

from cryptanalysislib.optimizers.helper import Range


def test1():
    r1 = Range("n", 32, 40)
    r2 = list(r1)
    assert r2 == [32,33,34,35,36,37,38,39]
    assert len(r2) == 8


def test2():
    r1 = Range("n", 32)
    r2 = list(r1)
    assert r2 == [32]
    assert len(r2) == 1


def test3():
    r1 = Range("n", 32, 40, 2)
    r2 = list(r1)
    assert r2 == [32,34,36,38]
    assert len(r2) == 4


def test4():
    r1 = Range("n", 40, 32)
    r2 = list(r1)
    assert r2 == [40,39,38,37,36,35,34,33]
    assert len(r2) == 8


def test5():
    r1 = Range("n", 40, 32, -1)
    r2 = list(r1)
    assert r2 == [40,39,38,37,36,35,34,33]
    assert len(r2) == 8


def test6():
    r1 = Range("n", 40, 32, 1)
    r2 = list(r1)
    assert r2 == [40,39,38,37,36,35,34,33]
    assert len(r2) == 8


def test7():
    r1 = Range("n", 40, 32, -2)
    r2 = list(r1)
    assert r2 == [40,38,36,34]
    assert len(r2) == 4


def test8():
    r1 = Range("n", 40, 32, 2)
    r2 = list(r1)
    assert r2 == [40,38,36,34]
    assert len(r2) == 4
