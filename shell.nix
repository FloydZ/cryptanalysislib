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
