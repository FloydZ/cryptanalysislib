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
include CMakeFiles/small_secret_lwe_library.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/small_secret_lwe_library.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/small_secret_lwe_library.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/small_secret_lwe_library.dir/flags.make

CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o: CMakeFiles/small_secret_lwe_library.dir/flags.make
CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o: ../src/config/config.cpp
CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o: CMakeFiles/small_secret_lwe_library.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o"
	/nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o -MF CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o.d -o CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o -c /home/duda/Downloads/crypto/lib/cryptanalysislib/src/config/config.cpp

CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.i"
	/nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/lib/cryptanalysislib/src/config/config.cpp > CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.i

CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.s"
	/nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/lib/cryptanalysislib/src/config/config.cpp -o CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.s

# Object files for target small_secret_lwe_library
small_secret_lwe_library_OBJECTS = \
"CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o"

# External object files for target small_secret_lwe_library
small_secret_lwe_library_EXTERNAL_OBJECTS =

libsmall_secret_lwe_library.a: CMakeFiles/small_secret_lwe_library.dir/src/config/config.cpp.o
libsmall_secret_lwe_library.a: CMakeFiles/small_secret_lwe_library.dir/build.make
libsmall_secret_lwe_library.a: CMakeFiles/small_secret_lwe_library.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libsmall_secret_lwe_library.a"
	$(CMAKE_COMMAND) -P CMakeFiles/small_secret_lwe_library.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/small_secret_lwe_library.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/small_secret_lwe_library.dir/build: libsmall_secret_lwe_library.a
.PHONY : CMakeFiles/small_secret_lwe_library.dir/build

CMakeFiles/small_secret_lwe_library.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/small_secret_lwe_library.dir/cmake_clean.cmake
.PHONY : CMakeFiles/small_secret_lwe_library.dir/clean

CMakeFiles/small_secret_lwe_library.dir/depend:
	cd /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/lib/cryptanalysislib /home/duda/Downloads/crypto/lib/cryptanalysislib /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug /home/duda/Downloads/crypto/lib/cryptanalysislib/cmake-build-debug/CMakeFiles/small_secret_lwe_library.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/small_secret_lwe_library.dir/depend

