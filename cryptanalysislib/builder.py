#!/usr/bin/env python3
"""
"""

import os
import random
import logging
import pathlib
from typing import List
from subprocess import Popen, PIPE, STDOUT

from cryptanalysislib.optimizers import *

logging.basicConfig(format="%(filename)s:%(lineno)s:%(funcName)20s(): %(message)s", 
                    level=logging.DEBUG)



class Cryptanalysislib:
    """
    Builder and Runner class 
    """

    debug_flags = ["-g", "-Og", "-DDEBUG", "-fopenmp"]
    release_flags = ["-DNDEBUG", "-O3", "-march=native", "-fopenmp"]
    cmake_executable = ["/usr/bin/env", "cmake"]

    clean_befor_build = False

    def __init__(self, debug:bool=False, seed:int=0, compiler="clang++"):
        """
        :param debug: if true debug binaries will be compiled
        :param seed: no idea
        :param compiler
        """
        random.seed(seed)

        # path of the build output
        self.tmp_build_dir = "/tmp/cryptanalysislib"

        # path of the source. Mainly needed by `cmake`
        self.source_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"

        # if true: debug binaries will be compiled
        self.__debug = debug
       
        # get set to `True`, if somewhere an error happens. Only used for 
        # debugging and testing.
        self.__error = False

        # to be able to specify the compiler from the outside
        self.compiler = compiler

        self.__create_build_env()


    def __create_build_env(self) -> bool:
        """ preparse the build environment
        runs `cmake -B` command.
        """
        tt = pathlib.Path(self.tmp_build_dir)
        if Cryptanalysislib.clean_befor_build and tt.is_dir():
            tt.rmdir()

        t = "Debug" if self.__debug else "Release"
        cmd = Cryptanalysislib.cmake_executable + ["-B", self.tmp_build_dir, 
               "-DCMAKE_BUILD_TYPE={t}".format(t=t), "-S", self.source_dir]
        cmd += ["-DCMAKE_CXX_COMPILER={t}".format(t=self.compiler)]
        logging.debug(cmd)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        if p.returncode != 0 and p.returncode is not None:
            assert p.stdout
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            self.__error = True
            return False
            
        return True

    def __build(self, target: str) -> bool:
        """ builds the target """
        if not self.__create_build_env():
            return False
        cmd = Cryptanalysislib.cmake_executable + \
            ["--build", self.tmp_build_dir, "--target", target]
        logging.debug(cmd)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            print("couldn't execute: %s", " ".join(cmd))
            print("error msg:")
            print(p.stdout.read())
            self.__error = True
            return False
        
        self.__build_output = [d.decode("utf-8").strip("\n") for d in p.stdout.readlines()]
        logging.debug(self.__build_output)
        return True

    def reset(self):
        """ reset the internal error state.
        NOTE: only used for debugging and testing
        """
        self.__error = False

    def has_error(self):
        """NOTE: only used for debugging and testing
        :return: true if no error is occured"""
        return self.__error != False
    def run(self,
            build_target: str,
            target: str,
            args: List[str] | None = None) -> bool:
        """ only exported fuction. Simply runs a target 
        :param build_target
        :param target actual binary to ru
        :param args cmd arguments passed to the binary
        """
        if not self.__build(build_target):
            return False

        cmd = [self.tmp_build_dir + "/" + target]
        if args:
            cmd += args

        logging.debug(cmd)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            self.__error = True
            return False
            
        self.__run_output = p.stdout.readlines()
        self.__run_output = [d.decode("utf-8").strip("\n") for d in self.__run_output]
        logging.debug(self.__run_output)
        return True
