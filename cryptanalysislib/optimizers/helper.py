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


