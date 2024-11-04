#!/usr/bin/env python3
""" 
"""

import argparse
import pprint

from cryptanalysislib.helper import Range
# TODO generate this dict from files from a subfolder: so the whole thing 
# is more dynamic
algos = [
    {
        "name": "subsetsum_rho",
        "description": "kek",
        # TODO class. header und so
        "parameters": [
            { "n": { "help": ""}}, 
            { "w": { "help": ""}}, 
            { "l1": { "help": ""}}, 
            { "l2": { "help": ""}}, 
            { "bp": { "help": "base p"}}, 
        ]
    },
]

def create_subparsers(subparsers, algos):
    for algo in algos:
        sp = subparsers.add_parser(algo["name"], help=algo["description"])
        for param in algo["parameters"]:
            k = list(param.keys())[0]
            help = param[k]["help"]
            sp.add_argument("-" + k, help=help, dest="algo_param", type=str,
                            action=Range)

def subparser_params_to_dict(param):
    ret = {}
    for k in param:
        ret[k.dest] = k
    return ret


def main():
    parser = argparse.ArgumentParser(description='Mother of all crypto algorithms.')

    subparsers = parser.add_subparsers(dest="algorithm", help='algorithms')
    subparsers.required = True
    create_subparsers(subparsers, algos)
    args = parser.parse_args()
    param = subparser_params_to_dict(args.algo_param)
    print(args.algorithm)
    pprint.pprint(param)
    print(param["n"].current)


if __name__ == "__main__":
    main()
