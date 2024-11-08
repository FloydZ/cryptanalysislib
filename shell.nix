with import <nixpkgs> { };
{ pkgs ? import <nixpkgs> { } }:
let 
  myPython = pkgs.python312;
  pythonPackages = pkgs.python312Packages;
  pythonWithPkgs = myPython.withPackages (pythonPkgs: with pythonPkgs; [
    ipython
    pip
    setuptools
    virtualenvwrapper
    wheel
    black
    prophet
  ]);

  # add the needed packages here
  extraBuildInputs = with pkgs; [
    pythonPackages.numpy
    pythonPackages.pytest
    cmake
    git
    libtool
    autoconf
    automake
    autogen
    gnumake
    python3
    # codelldb
    cmake
    lldb
    clang_17
    clang-tools_17
    llvm_17
    llvmPackages_17.libcxx
    llvmPackages_17.openmp
    gcc
    gtest
    gbenchmark
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
    flamegraph
    gdb
    linuxKernel.packages.linux_6_6.perf
    pprof
    valgrind
    massif-visualizer

    # opencl stuff (intel)
    # needed for libOpenCL.so
    clang
    ocl-icd
    clinfo
    #intel-compute-runtime
    #intel-ocl
    # needed for <CL/cl.h>
    opencl-headers
    opencl-clhpp
    # needed for <Gl/gl.h>
    libGL
    libGLU

    # opencl cuda
    cudaPackages.cuda_opencl
    cudaPackages.cudatoolkit
    cudaPackages.cuda_cudart
  ]));
in
import ./python-shell.nix { 
 extraBuildInputs=extraBuildInputs; 
 myPython=myPython;
 pythonWithPkgs=pythonWithPkgs;
}
