with import <nixpkgs> { };
{ pkgs ? import <nixpkgs> { } }:

stdenv.mkDerivation {
  name = "cryptanalysislib";
  src = ./.;

  buildInputs = [
    cmake
    git
    libtool
    autoconf
    automake
    autogen
    gnumake
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
    openssl     # needed for libcoro
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
    flamegraph
    gdb
    linuxKernel.packages.linux_6_6.perf
    pprof
    valgrind
    massif-visualizer
  ]));
}
