with import <nixpkgs> { };
{ pkgs ? import <nixpkgs> { } }:
let 
  myPython = pkgs.python3;
  pythonPackages = pkgs.python3Packages;
  pythonWithPkgs = myPython.withPackages (pythonPkgs: with pythonPkgs; [
    ipython
    pip
    setuptools
    virtualenvwrapper
    wheel
    black
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
    cmake
    lldb
    clang
    clang-tools
    llvm
    llvmPackages.libcxx
    llvmPackages.openmp
    gcc
    gtest
    gbenchmark
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
      #flamegraph
      #gdb
      #linuxKernel.packages.linux_6_6.perf
      #pprof
      #valgrind
      #massif-visualizer

      ## opencl stuff (intel)
      ## needed for libOpenCL.so
      #clang
      #ocl-icd
      #clinfo
      ##intel-compute-runtime
      ##intel-ocl
      ## needed for <CL/cl.h>
      #opencl-headers
      #opencl-clhpp
      ## needed for <Gl/gl.h>
      #libGL
      #libGLU

    # opencl cuda
    #cudaPackages.cuda_opencl
    #cudaPackages.cudatoolkit
    #cudaPackages.cuda_cudart

    # cuda
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvcc
      cudaPackages.cuda_cccl
      cudaPackages.cuda_gdb
      cudaPackages.cuda_cupti
      cudaPackages.nsight_compute
      cudaPackages.libcurand
      linuxPackages.nvidia_x11

  ]));
in
let
  b = ''
    # NVIDIA Driver and CUDA setup
    export NVIDIA_VISIBLE_DEVICES=all
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export CUDA_VISIBLE_DEVICES=0


    # Library paths with specific NVIDIA driver
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    export LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LIBRARY_PATH

    # OpenGL driver path
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH

    echo "CUDA C/C++ development environment ready"
    echo "GCC version in use:"
    gcc --version
    echo "NVCC version:"
    nvcc --version
  '';


  buildInputs  = with pkgs; [
      clang
      llvmPackages.bintools
      rustup
  ] ++ extraBuildInputs;
  lib-path = with pkgs; lib.makeLibraryPath buildInputs;
  shell = pkgs.mkShell {
    buildInputs = [
       # my python and packages
        pythonWithPkgs
        
        # other packages needed for compiling python libs
        pkgs.readline
        pkgs.libffi
        pkgs.openssl
  
        # unfortunately needed because of messing with LD_LIBRARY_PATH below
        pkgs.git
        pkgs.openssh
        pkgs.rsync
    ] ++ extraBuildInputs;
    shellHook = ''
        # Allow the use of wheels.
        SOURCE_DATE_EPOCH=$(date +%s)
        # Augment the dynamic linker path
        export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
        # Setup the virtual environment if it doesn't already exist.
        VENV=.venv
        if test ! -d $VENV; then
          virtualenv $VENV
        fi
        source ./$VENV/bin/activate
        export PYTHONPATH=$PYTHONPATH:`pwd`/$VENV/${myPython.sitePackages}/
        # pip install -e .
    '' + pkgs.lib.optionalString pkgs.stdenv.isLinux b;
  };
in shell
