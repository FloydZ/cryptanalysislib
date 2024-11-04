#!/usr/bin/env python3
"""
"""

from cryptanalysislib.helper import dict2include, dict2str

def test1():
    s = (dict2str({"a": 1, "L1": 2})) 
    assert s == """#ifndef INCLUDE_PARAMS
    #define INCLUDE_PARAMS
    
    #define PARAM_a 1
    #define PARAM_L1 2
    
    #endif"""

#dict2include("./test.h", {"a": 1})
