with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

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
	cmake 
    clang_13
	llvmPackages_13.openmp
    # clang_17
	# clang-tools_17
	# llvm_17
	# llvmPackages_17.libcxx
	# llvmPackages_17.openmp
	gcc
	gtest
	gbenchmark 
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
	flamegraph
	gdb
    # linuxKernel.packages.linux_6_5.perf
	pprof
	valgrind
	massif-visualizer
  ]));
}
