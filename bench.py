import os
import random
import logging
import json
import pathlib
from typing import List
from subprocess import Popen, PIPE, STDOUT


def read_google_benchmark_data(args):
    """parse a csv/json output of a google benchmark run """
    extension = pathlib.Path(args.file.name).suffix
    data = []
    try:
        if extension == ".csv":
            # data = pd.read_csv(args.file, usecols=["name", args.metric])
            print("error not impl") 
        elif extension == ".json":
            json_data = json.load(args.file)
            data = json_data["benchmarks"]
        else:
            logging.error("Unsupported file extension '{}'".format(extension))
            exit(1)
    except ValueError:
        logging.error('Could not parse the benchmark data.')
        return 0

    for d in data:
        # TODO probably pack this into a try catch block
        d["label"] = d["name"].split("/")[0]
        d["size"] = d["name"].split("/")[1]
    return data


class Cryptanalysislib:
    compilers = ["g++", "clang++"]
    search_path = ["/usr/bin", "/usr/bin/env"]
    debug_flags = ["-g", "-Og", "-DDEBUG", "-fopenmp"]
    release_flags = ["-DNDEBUG", "-O3", "-march=native", "-fopenmp"]
    cmake_executable = ["/usr/bin/env", "cmake"]

    def __init__(self, debug:bool=False, seed:int=0):
        """
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

    def run(self, target: str, args: List[str]) -> bool:
        """ """
        if not self.__build(target):
            return False

        # TODO: this is not correct in general
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