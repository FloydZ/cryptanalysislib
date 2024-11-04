#!/usr/bin/env python3 

from typing import List, Dict
from cryptanalysislib.optimizers.helper import Range


class Optimizer:
    """
    generic optimization class, just enforcing some abstract methods
    """
    def __init__(self) -> None:
        pass

    #def __iter__(self):
    #    return self
    
    #def __next__(self):
    #    pass

    #def opt(self):
    #    pass


class MetaOptimizer(Optimizer):
    """
    NOTE: this optimizer does not optimize parameters for any particular 
        problem. But instead it optimizes for different `n` another optimizer.
    """
    def __init__(self,
                 sub_problem_type,
                 parameters: Range|List[Range]) -> None:
        super().__init__()
        self.sub_problem_type = sub_problem_type
        self.parameters = parameters if isinstance(parameters, list) else [parameters]
        self.nr_params = len(self.parameters)
   
    def ranges2dict(self):
        """simply copies the current values into a new dict"""
        assert isinstance(self.parameters, list)
        ret = {}
        for r in self.parameters:
            ret[r.name] = r.current 
        return ret

    def __iter__(self):
        """ """
        run = True
        while run:
            d = self.ranges2dict()
            b, o = self.sub_problem_type(**d).opt()
            #print(d, o)
            # this check is rather important, as its insures that parameter 
            # configurations, which are not valid (due to too strict memory 
            # limits) are discarded
            if b:
                yield o

            for _ in self.parameters[0]:
                d = self.ranges2dict()
                b, o = self.sub_problem_type(**d).opt()
                #print(d, o)
                if b:
                    yield o
   
            for ccp in range(0, self.nr_params):
                try:
                    next(self.parameters[ccp])
                    break
                except:
                    self.parameters[ccp].reset()
                    if ccp == (self.nr_params - 1):
                        return
