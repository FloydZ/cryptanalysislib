cmake_minimum_required(VERSION 3.20)  		# 3.20 is needed because of C++20
project(CryptanalysisLib LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(USE_TCMALLOC "Enable tcmalloc" OFF)
option(USE_SANITIZER "Enable memory sanitizer" OFF)
option(USE_NOINLINE "Enable -fno-inline" OFF)
option(USE_LTO "Enable -flto" OFF)
option(USE_PROFILE "Enable profile guided optmization" OFF)
option(USE_POLLY "Enable polly loop optimization" OFF)
option(USE_BOLT "Enable bolt binary optmization" OFF)
option(USE_TRACY "Enable tracy profiling" OFF)

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

# local cmake includes
include("deps/cmake_optimize_for_architecture/Optimizations.cmake")

# only for the CI
include_directories("/opt/homebrew/include/")
include_directories("benchmark/include")
link_directories("/opt/homebrew/lib")
link_directories("benchmark/build/src")

check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag(-fno-inline COMPILER_SUPPORTS_NO_INLINE)
check_cxx_compiler_flag(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS) # TODO support not recognized
check_cxx_compiler_flag(-fsanitize=memory COMPILER_SUPPORTS_SANITIZE_MEMORY)
check_cxx_compiler_flag(-fconstexpr-steps=91 COMPILER_SUPPORTS_CONSTEXPR_COST)
check_cxx_compiler_flag(-fno-exceptions COMPILER_SUPPORTS_NO_EXCEPTION)
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

if(${USE_TRACY})
	FetchContent_Declare(
		tracy
		GIT_REPOSITORY 	https://github.com/wolfpld/tracy
		GIT_TAG 		master
		GIT_SHALLOW 	TRUE
		GIT_PROGRESS 	TRUE
	)
	FetchContent_MakeAvailable(tracy)
endif ()

# global include directories
include_directories("deps/b63/include/b63")
include_directories("src")
include_directories("/usr/local/include")

# global link directories
link_directories("/usr/local/lib")

# two possible options. 'tcmalloc' is the standard google malloc implementation
if(${USE_TCMALLOC})
	message(STATUS "Using tcmalloc")
	set(MALLOC_LIBRARY_FLAGS "tcmalloc")
endif()

if(${USE_SANITIZER} AND COMPILER_SUPPORTS_SANITIZE_ADDRESS)
	message(STATUS "Using address sanitize")
	# possible sanitizers = -fsanitize=[address,leak,thread,memory,undefined]
	set(SANITIZER_FLAGS "-fsanitize=address -fsanitize=pointer-compare -fno-omit-frame-pointer")
endif()

if(${USE_NOINLINE} AND COMPILER_SUPPORTS_NO_INLINE)
	message(STATUS "not inlining")
	set(INLINE_FLAGS "-fno-inline")
endif()

if (COMPILER_SUPPORTS_CONSTEXPR_COST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=99999999")
endif()

if(${USE_LTO} AND lto_supported)
  message(STATUS "IPO / LTO enabled")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# -Wno-zero-length-array: needed, because in the simd layer we sometimes need to allocate stuff like
#   simd512_type v512[simd512_limbs];
# and if the type is < 512 bits, then this must be zero
# -Wno-unused-function: needed, because we have alot of helper function, which are not always used.
# -Wno-unknown-pragmas: needed for `#pragma unroll`
set(ALLOWED_WARNINGS " -Wno-c11-extensions -Wno-unknown-pragmas -Wno-gnu-label-as-value -Wno-zero-length-array -Wno-unused-function -Wno-invalid-constexpr")

message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
# for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
# -fsized-deallocation -fno-builtin-malloc -fno-builtin-free
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") 
	message(STATUS "ARM CPU detected")
    set(ADDITIONAL_FLAGS "-DUSE_ARM -flax-vector-conversions -march=native")
	set(DEBUG_FLAGS "-g -Wall -Wextra -DDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-g -O3 -DNDEBUG  ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${ADDITIONAL_FLAGS}")
else()

    set(DEBUG_FLAGS "-g -O0 -Wall -Wextra -Wpedantic -DDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
	set(OPTIMIZE_FLAGS "-g -march=native -DNDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${ADDITIONAL_FLAGS}")
endif()

set(LINK_TEST_FLAGS 
	GTest::gtest 
	GTest::gtest_main 
	Threads::Threads 
	OpenMP::OpenMP_CXX
    # TODO think about if this is really needed
	# atomic
)

set(LINK_BENCH_FLAGS 
	benchmark::benchmark 
	# atomic
)

if(${USE_TRACY})
	message(STATUS "Using tracy")
	list(APPEND LINK_TEST_FLAGS TracyClient)
	list(APPEND LINK_BENCH_FLAGS TracyClient)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TRACY")
endif()

# Set additionally our own debug/optimization compiler/linkage flags. Maybe not
# he nicest way, but by far the easiest.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")

# on apple we need to add standard library???
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}     -stdlib=libc++")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
	elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}     -stdlib=libstdc++")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libstdc++")
	endif()
endif()

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

