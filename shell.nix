with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "cryptanalysislib";
  src = ./.;

  buildInputs = [ git libtool autoconf automake autogen gnumake cmake clang ];

  # buildPhase = "c++ -o main main.cpp -lPocoFoundation -lboost_system";

  # installPhase = ''
  #  mkdir -p $out/bin
  #  cp main $out/bin/
  # '';
}
