#!/usr/bin/env python3
import math


def HH(i: float):
    if i == 1.0 or i == 0.0:
        return 0.0

    if i > 1.0 or i < 0.0:
        print("error: ", i)
        raise ValueError

    return -(i * math.log2(i) + (1 - i) * math.log2(1 - i))


def H1(value: float):
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
    compute probability of
    """
    assert w_ < n
    assert d_ < n
    wm1 = 1 - w_
    return wm1 * HH((d_ - (w_ / 2)) / wm1)



def compute_optimal_params(n: int, lam, w, r=0):
    """
    taken from https://arxiv.org/pdf/2102.02597.pdf

    lambda = list size
    gamma = weight to diff
    delta = weight to match exactly on each bucket_window (relative to k)
    k = number of coords per window
    :return
        r: number of sublimbs
        N: number of leaves
        d: weight on each window
        k: window size

    """
    # convert to relative numbers
    if type(lam) == int or lam > 30:
        lam = math.log2(lam)/n
    
    if type(w) == int:
        w = w / n
    
    # only set r if explicity wanted
    if r == 0:
        r = n / (math.log2(n))**2
    
    assert(lam <= 1.)

    delta_star = H1(1. - lam)
    limit = 2.*delta_star * (1. - delta_star)
    if w > limit:
        d = 1/2. * (1. - math.sqrt(1. - 2.*w))
    else:
        d = delta_star

    q = calc_q_(n, w, d)
    assert q <= 1.

    N = int(n / q)

    k = n/r

    return r, N, d*k, k


def compute_time(n, lam, w, r, N, d):
    """
    compute the expected runtime in log2
    """

    if type(lam) == int or lam > 30:
        lam = math.log2(lam)/n

    w = w
    if type(w) == int:
        w = w / n

    delta_star = H1(1. - lam)
    w_star = 2*delta_star * (1. - delta_star)
    print(w, w_star, lam)
    if w <= w_star:
        theta = (1. - w) * (1.0 - HH((delta_star - w/2.)/(1. - w)))
    else:
        theta = 2*lam + HH(w) - 1.

    return theta*n


# TODO compute intermediate list size
n = 256
lam = 1 << 14
w = 16
r, N, d, k = compute_optimal_params(n, lam, w)
print("r", r, "N", N, "d", d)

time = compute_time(n, lam, w, r, N, d)
print("theta", time, 2**time)
