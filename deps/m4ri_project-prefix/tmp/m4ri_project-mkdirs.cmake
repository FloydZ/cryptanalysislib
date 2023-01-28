# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src/m4ri_project-build"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/tmp"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src/m4ri_project-stamp"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src"
  "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src/m4ri_project-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src/m4ri_project-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/duda/Downloads/crypto/decoding/decoding/deps/cryptanalysislib/deps/m4ri_project-prefix/src/m4ri_project-stamp${cfgdir}") # cfgdir has leading slash
endif()
