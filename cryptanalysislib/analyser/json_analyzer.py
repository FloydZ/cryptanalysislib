#!/usr/bin/env python3 
"""
"""

from typing import List
import json
import math
from cryptanalysislib.analyser.analyser import Analyser

class json_Analyser(Analyser):
    """
    simple anayliser which marks 
        
    """
    def __init__(self, lines: str | List[str]):
        if isinstance(lines, List):
            lines = "[" + ",".join(lines) + "]"
        elif isinstance(lines, str):
            lines = lines.split("\n")
            lines = "[" + ",".join(lines) + "]"
        self.lines = lines
        self.j = json.loads(lines)
        if not isinstance(self.j, List):
            self.j = [self.j]
        
    def min(self, key: str):
        """
        """
        m = math.inf 
        for v in self.j:
            if v[key] < m:
                m = v[key]
        return m
            

    def max(self, key: str):
        """
        """
        m = -math.inf
        for v in self.j:
            if key in v and v[key] > m:
                m = v[key]
        return m

    def avg(self, key: str):
        """
        """
        m = 0
        ctr = 0
        for v in self.j:
            if key in v:
                m += v[key]
                ctr += 1
        return m/ctr

    def med(self, key: str):
        """
        computes the median
        """
        m = []
        for v in self.j:
            if key in v:
                m.append(v[key])
        m.sort()
        l = len(m)
        return m[l//2]



if __name__ == "__main__":
    lines1 = """{"a": 1, "b": 2}"""
    lines2 = ["""{"a": 1, "b": 2}""", """{"a": 3, "b": 4}"""]
    
    # { "name": "SubSetSumConfig", "d": 2, "n": 32, "q": 4294967296, "bp": 2, "l1": 10, "l2": 6, "l3": 0, "walk_len": 128, "flavour_q": 1021, "bit_pos": 0, "print_iterations": 512 }
    # { "rho_calls": 6383963.6, "collisions": 2341.2, "f_calls": 859628243.0, "avg_tree_iters": 1.271048, "avg_walk_len": 44.520920000000004, "seconds": 214.4 }
    
    # j1 = json_Analyser([lines1])
    lines3 = """{ "rho_calls": 12923590, "collisions": 4654, "f_calls": 1740176896, "avg_tree_iters": 1.26504, "avg_walk_len": 44.5504, "seconds": 430 }
    { "rho_calls": 43152, "collisions": 10, "f_calls": 5816187, "avg_tree_iters": 1.28046, "avg_walk_len": 44.5946, "seconds": 1 }
    { "rho_calls": 12846079, "collisions": 4866, "f_calls": 1727343811, "avg_tree_iters": 1.26624, "avg_walk_len": 44.4882, "seconds": 436 }
    { "rho_calls": 1556366, "collisions": 505, "f_calls": 207354551, "avg_tree_iters": 1.28137, "avg_walk_len": 44.0766, "seconds": 52 }
    { "rho_calls": 4550631, "collisions": 1671, "f_calls": 617449770, "avg_tree_iters": 1.26213, "avg_walk_len": 44.8948, "seconds": 153 }"""
    
    
    j = json_Analyser(lines3)
    print(j.avg("rho_calls"))
    print(j.avg("collisions"))
    print(j.avg("f_calls"))
    print(j.avg("avg_tree_iters"))
    print(j.avg("avg_walk_len"))
    print(j.avg("seconds"))
