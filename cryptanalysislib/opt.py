#!/usr/bin/env python3 

from math import comb
from typing import Dict, List


def binomH(n,k):
    """
    binomial coefficient
    """
    # if k/n not in ZZ:
    #     return -100
    if(n<=0):
        return 1.
    return comb(int(n),int(k))


def multiH(n,c):
    """
    multinomial coefficient
    """
    if sum(c)>n:
        return 1
    tot=1
    for i in c:
        tot*=binomH(n,i)
        n-=i
    return tot


def reps(p, m, d, l): 
    """
    representations of length-l vector with p ones and m minus ones
    two length-l vectors with p/2+d ones and m/2+d minus ones each.
    """
    if p <= 0.000001 or l == 0.:
        return 1
    if l < p or l - p -m < 2*d:
        return 1
    
    return binomH(p,p/2) * binomH(m,m/2) * multiH(l-p-m, [d,d]) 


class Range:
    """ just a wrapper around `range()` function with a name """
    def __init__(self, name: str, start: int, end: int = -1, step: int = 0) -> None:
        """
        :param name:
        :param start:
        :param end: -1 is the not passed symbol
        :param step:
        """
        assert (end != start)

        self.name = name
        self.start = start
        self.current = start
        
        if end != -1:
            if start < end:
                self.end = start + 1 if end == -1 else end
                self.step = 1 if step == 0 else step
            else:
                self.end = start - 1 if end == -1 else end
                self.step = -1 if step == 0 else step
                # make sure that setp is negative
                if self.step > 0:
                    self.step = -self.step
        else:
            self.end = start + 1
            self.step = 1
        
    def size(self):
        """ returns the number of value the Range can enumerate at most """
        return abs((abs(abs(self.end) - self.start + abs(self.step) - 1)) // self.step)
    
    def reset(self):
        self.current = self.start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += self.step
        if self.step > 0:
            if self.current >= self.end:
                raise StopIteration
        else: 
            if self.current <= self.end:
                raise StopIteration
        return self.current

class Optimizer:
    """
    generic optimization class, just enforcing some abstract methods
    """
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        pass

    def opt(self):
        for k in self:
            yield k

        return False, {}

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
        """
        """
        assert isinstance(self.parameters, list)
        ret = {}
        for r in self.parameters:
            ret[r.name] = r.current 
        return ret

    def opt(self):
        run = True
        while run:
            d = self.ranges2dict()
            b, o = self.sub_problem_type(**d).opt()
            #print(d, o)
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
                    if (ccp == (self.nr_params) - 1):
                        run = False

        return False, {}
