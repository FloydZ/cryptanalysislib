# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /nix/store/k7lm30wld0jhdks4maz47v7ak8ydv2g6-cmake-3.22.3/bin/cmake

# The command to remove a file.
RM = /nix/store/k7lm30wld0jhdks4maz47v7ak8ydv2g6-cmake-3.22.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/duda/Downloads/crypto/lib/cryptanalysislib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug

# Include any dependencies generated for this target.
include bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/compiler_depend.make

# Include the progress variables for this target.
include bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/progress.make

# Include the compile flags for this target's objects.
include bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/flags.make

bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o: bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/flags.make
bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o: ../bench/mem/malloc_free.cpp
bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o: bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o -MF CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o.d -o CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o -c /home/duda/Downloads/crypto/lib/cryptanalysislib/bench/mem/malloc_free.cpp

bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.i"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/lib/cryptanalysislib/bench/mem/malloc_free.cpp > CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.i

bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.s"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/lib/cryptanalysislib/bench/mem/malloc_free.cpp -o CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.s

# Object files for target bench_b63_mem_malloc_free
bench_b63_mem_malloc_free_OBJECTS = \
"CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o"

# External object files for target bench_b63_mem_malloc_free
bench_b63_mem_malloc_free_EXTERNAL_OBJECTS =

bench/mem/bench_b63_mem_malloc_free: bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/malloc_free.cpp.o
bench/mem/bench_b63_mem_malloc_free: bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/build.make
bench/mem/bench_b63_mem_malloc_free: bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_b63_mem_malloc_free"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_b63_mem_malloc_free.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/build: bench/mem/bench_b63_mem_malloc_free
.PHONY : bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/build

bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/clean:
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem && $(CMAKE_COMMAND) -P CMakeFiles/bench_b63_mem_malloc_free.dir/cmake_clean.cmake
.PHONY : bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/clean

bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/depend:
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/lib/cryptanalysislib /home/duda/Downloads/crypto/lib/cryptanalysislib/bench/mem /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bench/mem/CMakeFiles/bench_b63_mem_malloc_free.dir/depend

