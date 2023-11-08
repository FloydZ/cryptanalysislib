with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "cryptanalysislib";
  src = ./.;

  buildInputs = [ 
  	git 
	libtool 
	autoconf 
	automake 
	autogen 
	gnumake 
	cmake 
	clang_16
	clang-tools_16
	llvm_16
	llvmPackages_16.libcxx
	llvmPackages_16.openmp
	gcc
	gtest
	gbenchmark 
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
	flamegraph
	gdb
    linuxKernel.packages.linux_6_5.perf
	pprof
	valgrind
	massif-visualizer
  ]));
}
