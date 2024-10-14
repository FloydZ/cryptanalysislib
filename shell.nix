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
    clang_18
    clang-tools_18
    llvm_18
    llvmPackages_18.libcxx
    llvmPackages_18.openmp
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
  ]));
in
import ./python-shell.nix { 
 extraBuildInputs=extraBuildInputs; 
 myPython=myPython;
 pythonWithPkgs=pythonWithPkgs;
}
