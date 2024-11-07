#!/usr/bin/env python3
""" 
"""

import argparse
import pprint
from typing import Dict, List

from cryptanalysislib.bench import Benchmarker
from cryptanalysislib.helper import Range
from cryptanalysislib.builder import Cryptanalysislib
from cryptanalysislib.optimizers.optimizers import MetaOptimizer
from cryptanalysislib.optimizers.subsetsum.rho import SubSetSumOptimizerD2
from cryptanalysislib.optimizers.subsetsum.hgj import HGJ
from cryptanalysislib.analyser.json_analyzer import json_Analyser
    
# TODO generate this dict from files from a subfolder: so the whole thing 
# is more dynamic
algos = [
    {
        "name": "subsetsum_rho",
        "description": "pollard rho meets tree algorihms",
        # 
        "class": SubSetSumOptimizerD2,
        # cmake build/run target
        "target": "bench_subsetsum_rho", # TODO maybe name all targets: bench_${algoname}
        # cmake output target == binary to exec
        "bin_path": "bench/subsetsum/bench_subsetsum_rho",
        # header file to write the parameters to
        "include_path": "../bench/subsetsum/params.h", # TODO path
        #
        "analyser": json_Analyser,
        # TODO explain params
        "parameters": [
            { "n": { "help": "length ofthe instanc3", "default": 32}}, 
            { "w": { "help": "weight of the subsetsum solution", "default": 16}}, 
            { "l1": { "help": "base list matching", "default": 10}}, 
            { "l2": { "help": "", "default": 6}}, 
            { "n1": { "help": "number of 1 in the baselist", "default": 0}}, 
            { "nm1": { "help": "number of -1 in the baselist", "default": 2}}, 
        ]
    },
    {
        "name": "subsetsum_hgj",
        "description": "reps",
        # 
        "class": HGJ,
        # cmake build/run target
        "target": "bench_subsetsum_hgj",
        # cmake output target == binary to exec
        "bin_path": "bench/subsetsum/bench_subsetsum_hgj",
        # header file to write the parameters to
        "include_path": "bench/subsetsum/params.h",
        # 
        "analyser": "",
        #
        "parameters": [
            { "n": { "help": "length ofthe instanc3", "default": 32}}, 
            { "w": { "help": "weight of the subsetsum solution", "default": 16}}, 
            { "l1": { "help": "base list matching", "default": 10}}, 
            { "l2": { "help": "", "default": 6}}, 
            { "n1": { "help": "number of 1 in the baselist", "default": 0}}, 
            { "nm1": { "help": "number of -1 in the baselist", "default": 2}}, 
        ]
    },
]


def algos_array_to_dict(algos: List):
    """
    """
    ret = {}
    for k in algos:
        ret[k["name"]] = k
    return ret 


def create_subparsers(subparsers, algos):
    """
    :param subparsers: `argparser` subparser, generated via: 
        subparsers = parser.add_subparsers(dest="algorithm", help='algorithms')
        subparsers.required = True
    :param algos: just the algos dictionary
    """
    for algo in algos:
        sp = subparsers.add_parser(algo["name"], help=algo["description"])
        for param in algo["parameters"]:
            k = list(param.keys())[0]
            help = param[k]["help"]
            sp.add_argument("-" + k, help=help, dest="algo_param", type=str,
                            action=Range)


def subparser_params_to_dict(param):
    """
    :param param: translates the outpout of the `argparse` Subparsers of the 
            form: 
                [Range(a, b, d, param_name), ...]
            to:
                {
                    param_name: Range(a,b,c),
                    ...
                }
    """
    ret = {}
    for k in param:
        ret[k.dest] = k
    return ret


def main():
    parser = argparse.ArgumentParser(description='Mother of all crypto algorithms.')
    parser.add_argument('--seed', help='prng seed', type=int, default=0)

    subparsers = parser.add_subparsers(dest="algorithm", help='algorithms')
    subparsers.required = True
    create_subparsers(subparsers, algos)
    args = parser.parse_args()
    param = subparser_params_to_dict(args.algo_param)
    # pprint.pprint(param)
    # pprint.pprint(args.algo_param)

    a = args.algorithm
    a1 = algos_array_to_dict(algos)
    a2 = a1[a]

    problemOptimizer = a2["class"]
    # metaOptimizer = MetaOptimizer(problemOptimizer, args.algo_param)
    builder = Cryptanalysislib(False, args.seed)
    bencher = Benchmarker(builder, problemOptimizer, a2["target"], 
                          a2["bin_path"], a2["include_path"], args.algo_param)

if __name__ == "__main__":
    main()
