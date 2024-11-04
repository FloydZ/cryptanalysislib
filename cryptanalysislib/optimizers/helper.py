#!/usr/bin/env python3
""" just a collection of basic functions which are used all the time """

from math import comb

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
        
    def size(self) -> int:
        """ returns the number of value the Range can enumerate at most """
        return abs((abs(abs(self.end) - self.start + abs(self.step) - 1)) // self.step)
    
    def reset(self):
        """ resets the current state of the range"""
        self.current = self.start
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        val = self.current
        self.current += self.step
        if self.step > 0:
            if val >= self.end:
                raise StopIteration
        else: 
            if val <= self.end:
                raise StopIteration
        return val
