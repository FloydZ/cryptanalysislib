cmake_minimum_required(VERSION 3.20)  		# 3.20 is needed because of C++23
project(CryptanalysisLib LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 23)

# options
option(USE_TCMALLOC "Enable tcmalloc" OFF)
option(USE_SANITIZER "Enable memory sanitizer" OFF)
option(USE_NOINLINE "Enable -fno-inline" OFF)
option(USE_LTO "Enable -flto" OFF)

# 
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# needed for nvim autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# import needed  
include(FetchContent)
include(ExternalProject)
include(CheckCXXCompilerFlag) 	# check for compiler optimization flags
include(CheckIPOSupported)  	# check for -flto
include(CheckLanguage) 			# needed for `check_language(CUDA)`

# local 
include("deps/cmake_optimize_for_architecture/OptimizeForArchitecture.cmake")
OptimizeForArchitecture()
message(STATUS _available_vector_units_list)

# only for the CI
include_directories("benchmark/include")
link_directories("benchmark/build/src")

check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag(-fno-inline COMPILER_SUPPORTS_NO_INLINE)
check_cxx_compiler_flag(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS) # TODO support not recognized
check_cxx_compiler_flag(-fsanitize=memory COMPILER_SUPPORTS_SANITIZE_MEMORY)
check_cxx_compiler_flag(-fconstexpr-steps=91 COMPILER_SUPPORTS_CONSTEXPR_COST)
check_ipo_supported(RESULT lto_supported OUTPUT error)
check_cxx_compiler_flag(-latomic LINKER_SUPPORTS_ATOMIC) # TODO

# find needed packages: the good thing only is: we only have a single external
# dependencies. 
enable_testing()
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest) 
find_package(benchmark)
find_package(Git QUIET)

# init the submodules
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule sync
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update sync failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()


# this is my try to enforce a builder script which runs with the: `cmake ..`
# command. TODO currently not working.
add_custom_target (
	CRYPTANALYSISLIB_BUILD_SCRIPT
	COMMAND ./${CMAKE_SOURCE_DIR}/build.sh
)

# global include directories
include_directories("deps/b63/include/b63")
include_directories("src")
include_directories("/usr/local/include")

# global link directories
link_directories("/usr/local/lib")

# two possible options. 'tcmalloc' is the standard google malloc implementation
if(USE_TCMALLOC)
	message(STATUS "Using tcmalloc")
	set(MALLOC_LIBRARY_FLAGS "tcmalloc")
endif()

if(USE_SANITIZER AND COMPILER_SUPPORTS_SANITIZE_ADDRESS)
	message(STATUS "Using address sanitize")
	# possible sanitizers = -fsanitize=[address,leak,thread,memory,undefined]
	set(SANITIZER_FLAGS "-fsanitize=address -fsanitize=pointer-compare -fno-omit-frame-pointer")
endif()

if(USE_NOINLINE AND COMPILER_SUPPORTS_NO_INLINE)
	message(STATUS "not inlining")
	set(INLINE_FLAGS "-fno-inline")
endif()

if (COMPILER_SUPPORTS_CONSTEXPR_COST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=99999999")
endif()

if(lto_supported AND USE_LTO)
  message(STATUS "IPO / LTO enabled")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(ALLOWED_WARNINGS "  -Wno-vla-extension -Wno-unused-function -Wno-overlength-strings -Wno-unknown-pragmas -Wno-unused-local-typedefs -Wno-invalid-constexpr")
set(GCC_HELPER_FLAGS "-fvar-tracking-assignments")

message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
# sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
# for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
# -fsized-deallocation -fno-builtin-malloc -fno-builtin-free
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") 
	message(STATUS "ARM CPU detected")
    include_directories(/opt/homebrew/opt/libomp/include)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/opt/libomp/link)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/opt/homebrew/lib)

    set(ADDITIONAL_FLAGS "-DUSE_ARM -march=native -flax-vector-conversions -march=native")

	set(DEBUG_FLAGS "-g -Wall -Wextra -Wpedantic -fpermissive -DDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-O3 -DNDEBUG -fno-exceptions ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZE_FLAG} ${ADDITIONAL_FLAGS}")
else()

if(_available_vector_units_list AND "avx2" IN_LIST _available_vector_units_list)
	message(STATUS "AVX2 CPU detected")
	set(AVX2_FLAGS "-DUSE_AVX2 -mavx2 -mavx -mbmi2 -march=native")
endif()
# TODO checking the CI
#if("avx512f" IN_LIST _available_vector_units_list)
#    set(AVX512_FLAGS "-DUSE_AVX512 -mavx512f -mavx512dq  -mavx512bf16 -mavx512cd  -mavx512bw -mavx512vl -mavx512vbmi -mavx512ifma -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq")
#endif()

    set(DEBUG_FLAGS "-g -O0 -Wall -Wextra -Wpedantic -DDEBUG ${INLINE_FLAGS} ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
	set(OPTIMIZE_FLAGS "-g -march=native -DNDEBUG -fno-exceptions ${INLINE_FLAGS} ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZE_FLAGS} ${ADDITIONAL_FLAGS}")
endif()

set(LINK_TEST_FLAGS 
	GTest::gtest 
	GTest::gtest_main 
	Threads::Threads 
	OpenMP::OpenMP_CXX
)

# Set additionally our own debug/optimization compiler/linkage flags. Maybe not 
# he nicest way, but by far the easiest.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")


# now check if we have to add the cuda runtime
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # global include path for dependencies
    # maybe a little ugly, but the easiest way to include `cuda_runtime.h`
    # as far as i know this is only needed for arch linux.
    include_directories(/opt/cuda/samples/common/inc)
    include_directories(/opt/cuda/include)
    include_directories(deps/cuda/common)

    #for ubuntu
    include_directories(/usr/local/cuda/samples/common/inc)

    # set the cuda standard
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED True)
    endif()


    # fall back, if one does not directly have a CUDA ready card, but rather a 
	# optimus (internal laptop) card.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif()

    #add_executable(test_cuda tests/test_cuda.cu)
    #target_link_libraries(test_cuda ${LINK_TEST_FLAGS})
endif()

# add sub directories
add_subdirectory(src)

if (benchmark_FOUND)
	add_subdirectory(bench)
endif()
if (GTest_FOUND)
	add_subdirectory(tests)
endif()

