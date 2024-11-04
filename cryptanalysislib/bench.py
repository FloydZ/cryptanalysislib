#!/usr/bin/env python3
"""
"""

import logging
from typing import List

from cryptanalysislib.helper import Range, dict2include
from cryptanalysislib.optimizers.optimizers import MetaOptimizer


logging.basicConfig(format="%(filename)s:%(lineno)s:%(funcName)20s(): %(message)s", 
                    level=logging.DEBUG)




class Benchmarker:
    """
    """
    def __init__(self, 
                 builder, 
                 optimizer,
                 target: str,
                 bin_path: str,
                 include_path: str,
                 ranges: List[Range]) -> None:
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
        for param in self.meta:
            dict2include(self.include_path, param)
            if not self.builder.run(self.target, self.bin_path):
                # exit in case of error
                break
