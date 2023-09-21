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
	gtest
	gbenchmark 
	gmp 
	tbb 
	libpng 
	mpfr 
	fplll 
	ninja
	ripgrep
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
	flamegraph
	gdb
    linuxKernel.packages.linux_6_4.perf
	pprof
	valgrind
	massif-visualizer
  ]));
}
