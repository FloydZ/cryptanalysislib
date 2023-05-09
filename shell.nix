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
	clang
	clang-tools
	#cudaPackages_11_0.cudatoolkit 
	gtest 
	gbenchmark 
	gmp 
	tbb 
	libpng 
	mpfr 
	fplll 
	ninja
	ripgrep
	flamegraph
	gdb
    linuxKernel.packages.linux_6_0.perf
	pprof
	valgrind
	massif-visualizer
  ];
}
