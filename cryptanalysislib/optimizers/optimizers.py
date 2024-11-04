#!/usr/bin/env python3 

from typing import List
from cryptanalysislib.helper import Range


class Optimizer:
    """
    generic optimization class, just enforcing some abstract methods
    """
    def __init__(self) -> None:
        pass


class MetaOptimizer(Optimizer):
    """
    NOTE: this optimizer does not optimize parameters for any particular 
        problem. But instead it optimizes for different `n` another optimizer.

    TODO: check if the names of the ranges matches the names of the arguments
    """
    def __init__(self,
                 sub_problem_type,
                 parameters: Range|List[Range]) -> None:
        super().__init__()
        self.sub_problem_type = sub_problem_type
        self.parameters = parameters if isinstance(parameters, list) else [parameters]
        self.nr_params = len(self.parameters)
        self.finish = False

        # init all parameters
        for i in range(self.nr_params):
            next(self.parameters[i])
   
    def ranges2dict(self):
        """simply copies the current values into a new dict"""
        assert isinstance(self.parameters, list)
        ret = {}
        for r in self.parameters:
            ret[r.name] = r.current 
        return ret

    def __iter__(self):
        return self

    def __next__(self):
        """ NOTE: cannot yield in an iterator, whithou making it a generator"""
        if self.finish:
            raise StopIteration()

        d = self.ranges2dict()
        b, o = self.sub_problem_type(**d).opt()
        
        for ccp in range(0, self.nr_params):
            try:
                next(self.parameters[ccp])
                break
            except:
                self.parameters[ccp].reset()
                next(self.parameters[ccp])
                if ccp == (self.nr_params - 1):
                    self.finish = True

        if not b:
            return next(self)
        return o
