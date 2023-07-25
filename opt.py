#!/usr/bin/env python3
import argparse
import math
import os
import re
import time
from subprocess import Popen, PIPE, STDOUT
from typing import Union


optimisations = " -O3 -Wall -march=native -mavx2 -mavx -mbmi -mbmi2 -ftree-vectorize -funroll-loops -fomit-frame-pointer " \
                "-Wno-unused-function  -Wno-unused-variable -Wno-unused-value -std=gnu++20 -DUSE_AVX2"


def parse_run_output(data):
    regex_int = r"(\d+)"
    regex_float = r"(\d+\.\d+)"

    time = -1.0
    sols = 0

    for line in data:
        if line.__contains__("sols:"):
            match = re.findall(regex_int, line)
            assert(len(match))
            sols = int(match[0])

        if line.__contains__("time:"):
            match = re.findall(regex_float, line)
            assert(len(match))
            time = float(match[0])

    return time, sols


def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)
    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            #os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.kill()
            raise RuntimeError("Process timed out")

        time.sleep(interval)


def rebuild(n: int, lam: int, gamma: int, k: int, delta: int, N: int, bf: int, iters=5):
    """

    :param n:
    :param lam:
    :param gamma:
    :param k:
    :param delta:
    :param N:
    :param bf:
    :param iters:
    :return:
    """
    target = "bench_nn_opt"
    path = "./cmake-build-release"

    # first clean everything
    p = Popen(["make", "clean"], stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=path)
    p.wait()
    
    lam = int(math.log2(lam))
    cmd = ["make", "CXX_FLAGS +=  -DBENCH_n=" + str(n) + " -DBENCH_N=" + str(N) + " -DBENCH_LS=" + str(lam) + \
           " -DBENCH_R="+str(r) + " -DBENCH_K="+str(k) + " -DBENCH_GAMMA="+str(gamma) + " -DBENCH_DELTA="+str(delta) +
           " -DBENCH_BF="+str(bf) + " -DBENCH_ITERS="+str(iters) + optimisations, "-B", target, "-j1"]

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=path)
    p.wait()
    data = p.stdout.read()
    if p.returncode != 0:
        print(data)
        return p.returncode, data

    return p.returncode, data


def run():
    """
    runs `./dinur_opt` in `cmake-build-release`
    :return: returncode, time, success prob
    """
    target = "bench_nn_opt"
    cmd = ["./" + target]
    path = "./cmake-build-release/bench/nn"
    seconds = 60
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
              preexec_fn=os.setsid, cwd=path)
    try:
        wait_timeout(p, seconds)

        data = p.stdout.readlines()
        data = [str(a).replace("b'", "")
                .replace("\\n'", "")
                .lstrip() for a in data]
        
        timee, sols = parse_run_output(data)
        print("runtime successfull")
        print(data)
        print(timee, sols)
        return p.returncode, timee, sols
    except Exception as e:
        print("error",e)
        data = p.stdout.readlines()
        print(data)
        return -1, -1, -1


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


def calc_p(k: int, delta: int, below=False) -> float:
    """
    computes the probability that any element survives the filtering step.
    :param k: number of coordinates to match on (most likely 32, 64)
    :param delta: number of coordinates/bits to be unequal
    :param below: if set: sum up the probabilities for d in range(delta + 1)
    :return:
    """
    if below:
        return sum([math.comb(k, d) / 2**k for d in range(delta + 1)])
    return math.comb(k, delta) / 2**k


def calc_q(k: int, gam: int, delt: int):
    """
    computes the probability that the golden element survives the filtering step
    :param k: bits/coordinates to match on
    :param gam: difference of the golden elements on all coordinates
    :param delt: differences on the k coordinates
    :return:
    """
    assert k > delt
    assert k > gam
    assert delt > gam // 2

    t1 = math.comb(gam, gam // 2)
    t2 = math.comb(k-gam, delt - (gam // 2))
    return t1*t2/2**k


def NN_compute_time(n: int, lam: int, gamma: int, r=0, N=0, d=0):
    """
    :param n: number of bits/coordinates
    :param lam: list size in int and not in logarithmic form
    :param gamma: weight difference
    :param r: number of limbs
    :param N: number of leaves
    :param d: weight diff on each limb
    :return:
    """
    lam_ = math.log2(lam)/n
    gamma_ = gamma/n

    delta_star = H1(1. - lam_)
    gamma_star = 2.*delta_star * (1. - delta_star)
    #print(lam_, gamma_, delta_star, gamma_star, lam)

    if gamma_ <= gamma_star:
        theta = (1. - gamma_) * (1. - HH((delta_star - gamma_ / 2.) / (1. - gamma_)))
    else:
        theta = 2 * lam_ + HH(gamma_) - 1.

    P = math.log2((n+1)**(n/math.log2(n)**2))
    print(theta, P)
    return theta*n


def NN_compute_list_sizes(n: int, r: int, lam: int, delta: int, below=False, logscale=True,max_switch=0):
    """
    computes the expected size of the intermediate lists. Additionally, it tries to estimate the best size for the
    bruteforce switch over.
    :param n: number of bits/coordinates per limb
    :param r: number of limbs
    :param lam: list size, not in logarithmic scale
    :param delta: weight difference which is allowed for each element in the list
    :param below: if set to true, elements with weight <= delta are also accepted
    :param logscale: return the list sizes in logarithmic scale
    :return:
    """
    k = int(n // r)
    p = calc_p(k, delta, below)
    Ls = [lam]

    for i in range(r):
        lam = lam*p
        Ls.append(lam)

        if max_switch and lam <= max_switch:
            break

    if logscale:
        Ls = [math.log2(l) for l in Ls]
    return Ls


def NN_compute_optimal_params(n: int, lam: int, gamma: int, r:int=0):
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
        theta: runtime
        listsizes: listsizes in each level
    """
    # convert to relative numbers
    lam_ = math.log2(lam)/n
    assert(lam_ <= 1.)

    gamma_ = gamma/n
    assert(gamma_ <= 1.)
    if gamma_ > 0.5:
        return -1, -1, -1, -1, -1

    # only set r if explicitly wanted
    if r == 0:
        r = int(n / (math.log2(n))**2)

    k = int(n//r)
    delta_star = H1(1. - lam_)
    limit = 2.*delta_star * (1. - delta_star)
    if gamma_ > limit:
        d = 1/2. * (1. - math.sqrt(1. - 2. * gamma_))
    else:
        d = delta_star

    delta = int(d*k)
    print(k, gamma, delta)
    q = calc_q(k, gamma, delta)
    assert q <= 1.

    N = int(1 / q)
    return r, N, delta, k, q, NN_compute_time(n, lam, gamma, r, N, delta), NN_compute_list_sizes(n, r, lam, delta)


def NN_practical_opt(n: int, lam: int, gamma: int, set_k: int = 0, set_delta: int = 0, set_N: int = 0, set_below=-1, compile: bool = False):
    """

    :param compile:
    :param n:
    :param lam:
    :param gamma:
    :param set_k:
    :param set_delta:
    :param set_N:
    :param set_below:
    :return:
    """
    max_switch_size = 1024
    iters = 10
    min_cost = math.inf
    params = {}
    for below in [True, False]:
        if set_below != -1 and below != set_below:
            continue

        for k in [32, 64]:
            if set_k and k != set_k:
                continue

            r = n // k
            for delta in range(gamma//2 + 1, k//2+1):
                if set_delta and delta != set_delta:
                    continue
                
                # compute the probability that the golden element survives the
                # filtering step
                q = calc_q(k, gamma, delta)
                # its inverse is the number of iterations per level
                N = int(1./q)
                if set_N:
                    N = set_N
                # computes the expected list sizes until the threshold of `max_switch`
                # is reached
                lss = NN_compute_list_sizes(n, r, lam, delta, below=below, logscale=False, max_switch=max_switch_size)

                # if set to true, we do not compute the theoretic optimal, but rather
                # try all values, to find the practical best.
                if compile:
                    print(below, k, r, delta, N)
                    c, data = rebuild(n, lam, gamma, k, delta, N, max_switch_size, iters)
                    if c:
                        print("rebuild error", data)
                        continue

                    _, timee, sols = run()

                    if timee == -1:
                        print('running time error')
                        continue
                    
                    #
                    if sols >= 0.8*iters:
                        if timee < min_cost:
                            min_cost = timee
                            params = {"k": k, "r": r, "N": N, "delta": delta, "lists": lss, "below": below}

                    continue


                cost = 0
                for i, l in enumerate(lss):
                    cost += N**(i+1) * l

                cost = math.log2(cost)
                if cost < min_cost:
                    min_cost = cost
                    params = {"k": k, "r": r, "N": N, "delta": delta, "lists": lss, "below": below}

    return min_cost, params


if __name__ == "__main__":
    n = 256
    lam = 1 << 20
    gamma = 14
    delta = 10
    r = 8
    k = n // r
    N = 129

    #print(calc_p(32, delta, True))
    #print(calc_q(32, delta, gamma))
    #print(NN_compute_time(n, lam, gamma, r, N, delta))
    #print(NN_compute_list_sizes(n, r, lam, delta))
    #print(NN_compute_optimal_params(n, lam, gamma))
    # print(NN_practical_opt(n, lam, gamma, set_k=32, set_delta=10, set_N=150, set_below=True))
    print(NN_practical_opt(n, lam, gamma, compile=True))


    exit(1)
    #n = 86
    #lam = math.comb(178, 2)
    #w = 6
    r, N, d, k, q = NN_compute_optimal_params(n, lam, delta)
    print("r", r, "N", N, "d", d, "log2(lam)", math.log2(lam), "q", q)

    time = NN_compute_time(n, lam, delta, r, N, d)
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

