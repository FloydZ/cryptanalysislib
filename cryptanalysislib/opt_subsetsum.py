#!/usr/bin/env python3

from math import comb, sqrt, inf, log2
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
    val=n
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
    def __init__(self, name: str, start: int, end: int = 0, step: int = 0) -> None:
        # TODO negative step value
        self.name = name
        self.start = start
        self.current = start
        self.end = start + 1 if end == 0 else end
        self.step = 1 if step == 0 else step

        assert (end > start)

    def size(self):
        """ returns the number of value the Range can enumerate at most """
        return (self.end - self.start + self.step - 1) // self.step
    
    def reset(self):
        self.current = self.start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += self.step
        if self.current >= self.end:
            raise StopIteration
        return self.current

class Optimizer:
    """
    TODO move somewhere usefull
    """
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        pass

    def opt(self) -> Dict:
        for k in self:
            print(k)

        return {}

class MetaOptimizer(Optimizer):
    """
    TODO move somewhere usefull

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

    def opt(self) -> Dict:
        run = True
        while run:
            d = self.ranges2dict()
            o = self.sub_problem_type(**d).opt()
            print(d, o)
            for _ in self.parameters[0]:
                d = self.ranges2dict()
                o = self.sub_problem_type(**d).opt()
                print(d, o)
   
            for ccp in range(0, self.nr_params):
                try:
                    next(self.parameters[ccp])
                    break
                except:
                    self.parameters[ccp].reset()
                    if (ccp == (self.nr_params) - 1):
                        run = False

        return {}

            
class SubSetSumOptimizerD2(Optimizer):
    """
    """
    def __init__(self, n: int, w: int = 0, max_mem: int = 0) -> None:
        """
        :param n: subset sum instance size
        :param w: subset sum weight
        :param max_mem: (logarithmic) max allowed memory for the optimization
                        process
        """
        super().__init__()
        self.n = n
        self.max_mem = n if max_mem == 0 else max_mem
        self.w = n/2 if w == 0 else w
        self.__w = self.w/self.n

        # optimization parameters
        self.d_1 = 0
        self.d_2 = 0

        self.l_1 = 0
        self.l_2 = 0

    def n1_0(self, g):
        return g*self.__w*self.n
    
    def nm1_0(self, g):
        return 0
    
    def n0_0(self, g):
        return g*self.n - self.n1_0(g)
    
    def n1_1(self, g):
        return self.n1_0(g)//2+self.d_1
    
    def nm1_1(self, g):
        return self.d_1
    
    def n0_1(self, g):
        return g*self.n - self.n1_1(g) - self.nm1_1(g)
    
    def n1_2(self, g):
        return self.n1_1(g)//2+self.d_2
    
    def nm1_2(self, g):
        return self.nm1_1(g)//2+self.d_2
    
    def n0_2(self, g):
        return g*self.n - self.n1_2(g) - self.nm1_2(g)
    
    def n1_3(self, g):
        return self.n1_2(g)//2
    
    def nm1_3(self, g):
        return self.nm1_2(g)//2
    
    def n0_3(self, g):
        return g*self.n - self.n1_3(g) - self.nm1_3(g)
    
    
    def R_1(self, g):
        return (reps(self.n1_0(g)/2,0,self.d_1/2,g*self.n/2))**2
    
    def R_2(self, g):
        return (reps(self.n1_1(g)/2,self.nm1_1(g)/2,self.d_2/2,g*self.n/2))**2
    
    
    def S_0(self, g):
        return (multiH(self.n/2,[self.__w*self.n/2]))**2
    
    def S_1(self, g):
        # multiH((1-g)*n/2,[(1-g)*n*w/2])
        return (multiH(g*self.n/2,[self.nm1_1(g)/2,self.n1_1(g)/2]))**2 * (2**((1-g)*self.n/2))  
    
    def S_2(self, g):
        # multiH((1-g)*n/4,[(1-g)*n*w/4])
        return (multiH(g*self.n/2,[self.nm1_2(g)/2,self.n1_2(g)/2]))**2 * (2**((1-g)*self.n/4))   
    
    def q_0(self, g):
        return (self.S_0(g)*self.R_1(g))/(self.S_1(g)*self.S_1(g))
    
    def q_1(self, g):
        return (self.S_1(g)*self.R_2(g))/(self.S_2(g)*self.S_2(g))
    
    def q_2(self, g):
        return 1
    
    def L_mitm(self, g):
        #multiH((1-g)*n/8,[(1-g)*n*w/8])
        return multiH(g*self.n*.5,[self.nm1_3(g),self.n1_3(g)]) * (2**((1-g)*self.n/8))  
    
    def L_2(self, g):
        return (self.L_mitm(g)*self.L_mitm(g))/(2**self.l_2)
    
    def L_1(self, g):
        return (self.L_2(g)**2)*(self.q_2(g)**2)/(2**self.l_1)
    
    def FL_2(self, g):
        return self.q_2(g)*self.L_2(g)
    
    def FL_1(self, g):
        return self.q_1(g)*self.L_1(g)
    
    def T_tree(self, g):    
        return self.L_2(g) + self.L_1(g) + self.L_mitm(g)
    
    def time(self, g):
        guess_s = 2**(self.l_1+self.l_2)/self.R_1(g)
        domain = 2**(self.l_2)
        one_collision = sqrt(domain)
        total_collisions = domain
        total_good_collisions = self.R_2(g)**2
        until_good = max(1,total_collisions / total_good_collisions)
        total = guess_s*until_good*one_collision*self.T_tree(g)
        return total
    
    def memory(self, g):
        max_of_FL_i = 2*self.L_mitm(g)+self.FL_2(g)+self.FL_1(g)
        return max_of_FL_i
    
    def opt(self) -> Dict:
        """ 
        """
        T = inf
        f = {"T": T}

        allocated_memory = 2**self.max_mem

        for self.d_1 in range(0,1):
            for j in range(0,101, 2):
                g = j*.01
                if(int(g*self.n)!=g*self.n):
                    continue

                for self.l_1 in range(1,self.n):
                    for self.l_2 in range(1,self.n):
                        if(self.l_1+self.l_2+(self.l_2)<=self.n and self.n0_1(g)>=0 and self.n0_2(g)>=0 and self.n0_3(g)>=0):
                            
                            if(self.memory(g)<allocated_memory and self.q_1(g)<=1 and self.q_1(g)>=0):
                                if(2**(self.l_1+self.l_2)>self.R_1(g) and 2**self.l_2>self.R_2(g)):
                                    
                                    if(self.FL_1(g) >= 1 and self.FL_1(g)<=2):
                                        t = log2(self.time(g))
                                        if(t<T):
                                            T = t
                                            f = {
                                                "T" : round(t, 3),
                                                "T_tree" : round(self.T_tree(g), 3),
                                                "M" : round(log2(self.memory(g)), 3),
                                                "mem_limit": self.max_mem,
                                                "g" : self.n*g,
                                                "L0" : self.L_mitm(g),
                                                "L1" : self.FL_1(g),
                                                "L2" : self.FL_2(g),
                                                "d1" : self.d_1,
                                                "d2" : self.d_2,
                                                "l1" : self.l_1,
                                                "l2" : self.l_2,
                                                # actual weight in the base lists
                                                "n1_3" : self.n1_3(g) + self.n//16,
                                                "nm1_3" : self.nm1_3(g),
                                                "n" : self.n,
                                            }
        return f

    #def __next__(self) -> Dict:
    #    T = inf
    #    f = {}
    #    while T == inf:
    #        f = self.opt()
    #        print("got")
    #        T = f["T"]
    #        print(T)
    #        self.current_max_mem += 1;
    #    
    #    print("kek", T)
    #    if self.current_max_mem > self.n:
    #        raise StopIteration()
    #    
    #    return f

# testing 

s = SubSetSumOptimizerD2(32)
#print(s.opt())
s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 40), Range("max_mem", 0, 32)])
#s = MetaOptimizer(SubSetSumOptimizerD2, [Range("n", 32, 40)])
s.opt()
