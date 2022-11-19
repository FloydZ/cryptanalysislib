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
CMAKE_BINARY_DIR = /home/duda/Downloads/crypto/lib/cryptanalysislib/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/flags.make

tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o: tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/flags.make
tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o: ../tests/test_ConstNonBlockingLinkedList.cpp
tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o: tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o -MF CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o.d -o CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o -c /home/duda/Downloads/crypto/lib/cryptanalysislib/tests/test_ConstNonBlockingLinkedList.cpp

tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.i"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/lib/cryptanalysislib/tests/test_ConstNonBlockingLinkedList.cpp > CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.i

tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.s"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/lib/cryptanalysislib/tests/test_ConstNonBlockingLinkedList.cpp -o CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.s

# Object files for target test_ConstNonBlockingLinkedList
test_ConstNonBlockingLinkedList_OBJECTS = \
"CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o"

# External object files for target test_ConstNonBlockingLinkedList
test_ConstNonBlockingLinkedList_EXTERNAL_OBJECTS =

tests/test_ConstNonBlockingLinkedList: tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/test_ConstNonBlockingLinkedList.cpp.o
tests/test_ConstNonBlockingLinkedList: tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/build.make
tests/test_ConstNonBlockingLinkedList: tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ConstNonBlockingLinkedList"
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ConstNonBlockingLinkedList.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/build: tests/test_ConstNonBlockingLinkedList
.PHONY : tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/build

tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/clean:
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_ConstNonBlockingLinkedList.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/clean

tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/depend:
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/lib/cryptanalysislib /home/duda/Downloads/crypto/lib/cryptanalysislib/tests /home/duda/Downloads/crypto/lib/cryptanalysislib/build /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests /home/duda/Downloads/crypto/lib/cryptanalysislib/build/tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_ConstNonBlockingLinkedList.dir/depend

