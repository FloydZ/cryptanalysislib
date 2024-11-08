#!/usr/bin/env python3
"""
"""

import logging
from typing import List

from cryptanalysislib.analyser.json_analyzer import json_Analyser
from cryptanalysislib.helper import Range, dict2include
from cryptanalysislib.optimizers.optimizers import MetaOptimizer

logging.basicConfig(format="%(filename)s:%(lineno)s:%(funcName)20s(): %(message)s", 
                    level=logging.DEBUG)


class Benchmarker:
    """
    Benchmarker:
    """
    def __init__(self, 
                 builder, 
                 optimizer,
                 target: str,
                 bin_path: str,
                 include_path: str,
                 ranges: List[Range],
                 *args, **kwargs) -> None:
        """
        :param builder:
        :param optimizer:
        :param target: cmake target name
        :param bin_path: actual binary path 
        :param include_path: path of the include header file to generate
        :param ranges:
        """
        self.builder = builder
        self.optimizer = optimizer
        self.target = target
        self.bin_path = bin_path
        self.include_path = include_path

        self.meta = MetaOptimizer(optimizer, ranges)

        self.iters = 100 # TODO
        outputs = []
        for param in self.meta:
            dict2include(self.include_path, param)
            for _ in range(self.iters):
                if self.builder.run(self.target, self.bin_path):
                    t = self.builder.run_output()
                    t = t[1:]
                    outputs.append(t[0])
                else:
                    # exit in case of error
                    break
        
        j = json_Analyser(outputs)
        print("rho_calls:", j.avg("rho_calls"))
        print("collisions:", j.avg("collisions"))
        print("f_calls:", j.avg("f_calls"))
        print("avg_tree_iters:", j.avg("avg_tree_iters"))
        print("avg_walk_len:", j.avg("avg_walk_len"))
        print("seconds:", j.avg("seconds"))
        print("pass_rho:", j.avg("pass_rho"))
        print("pass_function_selector:", j.avg("pass_function_selector"))
        print("pass_weight_check:", j.avg("pass_weight_check"))

