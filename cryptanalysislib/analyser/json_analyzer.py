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



lines1 = """{"a": 1, "b": 2}"""
lines2 = ["""{"a": 1, "b": 2}""", """{"a": 3, "b": 4}"""]
# j1 = json_Analyser([lines1])
print(json_Analyser(lines2).max("a"))

