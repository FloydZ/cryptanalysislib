#!/usr/bin/env python3
import argparse
import math
from typing import Union

def HH(i: float):
    """
    binary entropy
    """
    if i == 1.0 or i == 0.0:
        return 0.0

    if i > 1.0 or i < 0.0:
        print("error: ", i)
        raise ValueError

    return -(i * math.log2(i) + (1 - i) * math.log2(1 - i))


def H1(value: float):
    """
    inverse binary entropy
    """
    if value == 1.0:
        return 0.5

    # approximate inverse binary entropy function
    steps = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.0000000001, 0.0000000000001, 0.000000000000001]
    r = 0.000000000000000000000000000000001
    if value < 0.:
        value = 0.

    for step in steps:
        i = r
        while (i + step < 1.0) and (HH(i) < value):
            i += step
            
        if step > i:
            break
        r = i - step

    return r


def calc_q_(n: int, w_: float, d_: float):
    """
    compute probability of any elements survives the filtering step
    """
    assert w_ < n
    assert d_ < n
    wm1 = 1 - w_
    return wm1 * HH((d_ - (w_ / 2)) / wm1)


def calc_q_2(k: int, gam: int, delt: int):
    t1 = math.comb(int(gam*k), int(gam*k/2))
    t2 = math.comb(int((1-gam)*k), int((delt-gam/2)*k))
    t2 /= 2**k
    return t1*t2


def NN_compute_optimal_params(n: int, lam: Union[int, float], gamma: Union[int, float], r=0):
    """
    taken from https://arxiv.org/pdf/2102.02597.pdf
    :var
        lambda = list size
        gamma = weight to diff
        delta = weight to match exactly on each bucket_window (relative to k)
        k = number of coords per window

    :input
        :n:
        :lam:
        :gamma:
        :r      if set to anything else than 0, this function computes its optimum

    :return
        r: number of sub-limbs
        N: number of leaves
        d: weight on each window
        k: window size
        q: probability that an element survives the filtering step
    """
    # convert to relative numbers
    if type(lam) == int or lam > 30:
        lam = math.log2(lam)/n
    
    if type(gamma) == int:
        if gamma >= n:
            return -1, -1, -1, -1
        gamma = gamma / n

    if gamma > 0.5:
        return -1, -1, -1, -1
    
    # only set r if explicitly wanted
    if r == 0:
        r = n / (math.log2(n))**2
    
    assert(lam <= 1.)

    k = n/r
    delta_star = H1(1. - lam)
    limit = 2.*delta_star * (1. - delta_star)
    if gamma > limit:
        d = 1/2. * (1. - math.sqrt(1. - 2. * gamma))
    else:
        d = delta_star

    q = calc_q_(k, gamma, d)
    # q = calc_q_2(int(k), gamma, d)
    assert q <= 1.

    N = int(n / q)
    return r, N, d*k, k, q


def NN_compute_time(n: int, lam: Union[int, float], gamma: Union[int, float], r=0, N=0, d=0):
    """
    compute the expected runtime in log2


    lambda = list size
    omega =
    gamma = weight to diff
    delta = weight to match exactly on each bucket_window (relative to k)
    k = number of coords per window
    """

    if type(lam) == int or lam > 30:
        lam = math.log2(lam)/n

    if type(gamma) == int:
        if gamma >= n:
            return -1
        gamma = gamma / n
    
    if gamma > 0.5:
        return -1

    delta_star = H1(1. - lam)
    w_star = 2*delta_star * (1. - delta_star)
    # print(w, w_star, lam)
    if gamma <= w_star:
        theta = (1. - gamma) * (1.0 - HH((delta_star - gamma / 2.) / (1. - gamma)))
    else:
        theta = 2 * lam + HH(gamma) - 1.

    return theta*n


def NN_compute_list_sizes(lam: Union[int, float], q: Union[int, float], r: Union[int, float], log=True):
    """
    computes the expected size of the intermediate lists. Additionally, it tries to estimate the best size for the
    bruteforce step over.
    :param lam: can be in logarithmic scale or normal scale
    :param q:
    :param r:   is rounded up
    :return: the expected list size on logarithmic scale
    """
    Ls = []

    if q > 1.:
        return []

    # expand the list size
    if lam < 30:
        lam = 2**lam

    if type(r) == float:
        r = math.ceil(r)

    Ls.append(lam)
    for i in range(r):
        lam = lam*q
        Ls.append(lam)

    if log:
        Ls = [math.log2(l) for l in Ls]
    return Ls


if __name__ == "__main__":
    n = 256
    lam = 1 << 20
    w = 16

    #n = 86
    #lam = math.comb(178, 2)
    #w = 6
    r, N, d, k, q = NN_compute_optimal_params(n, lam, w)
    print("r", r, "N", N, "d", d, "log2(lam)", math.log2(lam), "q", q)

    time = NN_compute_time(n, lam, w, r, N, d)
    print("theta", time, 2**time)

    Ls = NN_compute_list_sizes(lam, q, r, False)
    print("|L|", Ls)
    exit(1)

    parser = argparse.ArgumentParser(description='Mother of all NN algorithms.')
    parser.add_argument('-n', help='problem dimension',
                        type=int, required=True)
    parser.add_argument('--lam', help='list size, can also be log',
                        type=int, default=1000)
    parser.add_argument('-w', help='weight diff',
                        type=int, default=1)

