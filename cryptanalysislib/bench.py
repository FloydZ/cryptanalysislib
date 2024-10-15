#!/usr/bin/env python3
"""
"""

import os
import random
import logging
import json
import pathlib
from typing import List, Dict
from subprocess import Popen, PIPE, STDOUT

from opt_subsetsum import SubSetSumOptimizerD2
from opt import MetaOptimizer, Range


class Cryptanalysislib:
    compilers = ["g++", "clang++"]
    search_path = ["/usr/bin", "/usr/bin/env"]
    debug_flags = ["-g", "-Og", "-DDEBUG", "-fopenmp"]
    release_flags = ["-DNDEBUG", "-O3", "-march=native", "-fopenmp"]
    cmake_executable = ["/usr/bin/env", "cmake"]

    def __init__(self, debug:bool=False, seed:int=0):
        """
        :param debug: if true debug binaries will be compiled
        :param seed: no idea
        """
        random.seed(seed)

        # path of the build output
        self.tmp_build_dir = "/tmp/cryptanalysislib"
        # path of the source. Mainly needed by `cmake`
        self.source_dir = os.path.dirname(os.path.realpath(__file__))

        # if true: debug binaries will be compiled
        self.__debug = debug

        self.__create_build_env()


    def __create_build_env(self) -> bool:
        """ preparse the build environment
        runs `cmake -B` command.
        """
        t = "Debug" if self.__debug else "Release"
        cmd = Cryptanalysislib.cmake_executable + ["-B", self.tmp_build_dir, 
               "-DCMAKE_BUILD_TYPE={t}".format(t=t), "-S", self.source_dir]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        if p.returncode != 0 and p.returncode is not None:
            assert p.stdout
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            return False
            
        return True

    def __build(self, target: str) -> bool:
        """ builds the target """
        if not self.__create_build_env():
            return False
        cmd = Cryptanalysislib.cmake_executable + \
            ["--build", self.tmp_build_dir, "--target", target]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            print("couldn't execute: %s", " ".join(cmd))
            print("error msg:")
            print(p.stdout.read())
            return False
        
        self.__build_output = p.stdout.readlines()
        self.__build_output = [d.decode("utf-8").strip("\n") for d in self.__build_output]
        return True

    def run(self,
            build_target: str,
            target: str,
            args: List[str]) -> bool:
        """ only exported fuction. Simply runs a target 
        :param build_target
        :param target actual binary to ru
        :param args cmd arguments passed to the binary
        """
        if not self.__build(build_target):
            return False

        cmd = [self.tmp_build_dir + "/" + target] + args
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            return False
            
        self.__run_output = p.stdout.readlines()
        self.__run_output = [d.decode("utf-8").strip("\n") for d in self.__run_output]
        return True


def dict2str(d: Dict):
    """
    """
    a = "\n".join(f"#define PARAM_{k} {v}" for k,v in d.items())
    ret = "#ifndef INCLUDE_PARAMS\n#define INCLUDE_PARAMS\n\n"
    ret += a
    ret += "\n\n#endif"
    return ret


def dict2include(file: pathlib.Path | str, d: Dict):
    """
    """
    a = dict2str(d)
    if isinstance(file, str):
        file = pathlib.Path(file)

    with file.open("w", encoding ="utf-8") as f:
        f.write(a)


class Benchmarker:
    """
    """
    def __init__(self, 
                 builder, 
                 optimizer,
                 target: str,
                 bin_path: str,
                 include_path: str,
                 ranges: List[Range]) -> None:
        """
        :param builder:
        :param optimizer:
        :param target: cmake target name
        :param bin_path: actual binary path 
        :param include_path: path of the include header file to generate
        """
        self.builder = builder
        self.optimizer = optimizer
        self.target = target
        self.bin_path = bin_path
        self.include_path = include_path

        self.meta = MetaOptimizer(optimizer, ranges)
        for param in self.meta.opt():
            print(param)
            dict2include(self.include_path, param)


# test code 
print(dict2str({"a": 1, "L1": 2}))
dict2include("./test.h", {"a": 1})
exit(1)
c = Cryptanalysislib()
b = Benchmarker(c, SubSetSumOptimizerD2, 
                "bench_subsetsum_tree", 
                "cmake-build-release/bench/subsetsum_tree", 
                "./bench/subsetsum/params.h",
                [Range("n", 32), Range("max_mem", 0, 32)]
                )
