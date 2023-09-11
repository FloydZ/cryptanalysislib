cmake_minimum_required(VERSION 3.11)
project(CryptanalysisLib LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)


# needed for nvim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

include(FetchContent)
include(ExternalProject)
set(CMAKE_VERBOSE_MAKEFILE ON)

include("deps/cmake_optimize_for_architecture/OptimizeForArchitecture.cmake")
OptimizeForArchitecture()

# include the external projects 'fplll', 'g6k', 'abseil' and 'tcmalloc'
#ExternalProject_Add(fplll_project
#        SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/fplll
#        CONFIGURE_COMMAND cd <SOURCE_DIR> && ./autogen.sh
#        COMMAND           cd <SOURCE_DIR> && ./configure
#        BUILD_COMMAND     cd <SOURCE_DIR> && make -j8
#        INSTALL_COMMAND   ""
#        TEST_COMMAND      ""
#)

ExternalProject_Add(m4ri_project
        SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/m4ri
        CONFIGURE_COMMAND cd <SOURCE_DIR> && autoreconf --install
        COMMAND           cd <SOURCE_DIR> && ./configure
        BUILD_COMMAND     cd <SOURCE_DIR> && make -j8
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)

#FetchContent_Declare(
#	tracy
#	GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#	GIT_TAG master
#	GIT_SHALLOW TRUE
#	GIT_PROGRESS TRUE
#)
#
#FetchContent_MakeAvailable (tracy)

add_custom_target (
        CRYPTANALYSISLIB_BUILD_SCRIPT
        COMMAND ./${CMAKE_SOURCE_DIR}/build.sh
)


# only for the CI
include_directories("benchmark/include")
link_directories("benchmark/build/src")

# include directories
include_directories("deps/fplll/fplll")
include_directories("deps/fplll")
include_directories("deps/b63/include/b63")
include_directories("deps/m4ri")
include_directories("deps/ska_sort")
include_directories("deps/ska_sort")
include_directories("deps/Static-Sort/include")
include_directories("src")
include_directories("/usr/local/include")

link_directories("deps/fplll/fplll/.libs")
link_directories("deps/m4ri/.libs")
link_directories("/usr/local/lib")

# some helper variables
# possible values are: [-DVALUE_KARY, -DVALUE_BINARY]
# the first enables a k-ary implementation of the class 'Value'
# the second sets the class 'Value' on a binary vector
# set(VALUE_FLAG -DVALUE_BINARY)
# set(VALUE_FLAG -DVALUE_KARY)
set(VALUE_FLAG "")

# two possible options. 'tcmalloc' is the standard google malloc implementation
#set(MALLOC_LIBRARY tcmalloc)
set(MALLOC_LIBRARY "")

# possible sanitzers = -fsanitize=[address,leak,thread,memory,undefined]
#set(SANITIZE_FLAG "-fsanitize=address")
set(SANITIZE_FLAGS "")

#set(INLINE_FLAGS "-fno-inline")
set(INLINE_FLAGS "")

set(ALLOWED_WARNINGS " -Wno-unused-function -Wno-overlength-strings")
#set(SANITIZER_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
set(GCC_HELPER_FLAGS "-fvar-tracking-assignments")

# sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
# for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
# -fsized-deallocation -fno-builtin-malloc -fno-builtin-free
if(APPLE)
    include_directories(/opt/homebrew/opt/libomp/include)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/opt/libomp/link)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/opt/homebrew/lib)

    set(ADDITIONAL_FLAGS "-DUSE_ARM")
    set(DEBUG_FLAGS "-g  -DDEBUG ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-O3 -flto -DNDEBUG -fno-exceptions ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAG} ${ADDITIONAL_FLAGS}")
else()

if("avx2" IN_LIST _available_vector_units_list)
	set(AVX2_FLAGS "-DUSE_AVX2 -mavx2 -mavx -mbmi2 ")
endif()

if("avx512f" IN_LIST _available_vector_units_list)
    set(AVX512_FLAGS "-DUSE_AVX512 -mavx512f -mavx512dq  -mavx512bf16 -mavx512cd  -mavx512bw -mavx512vl -mavx512vbmi -mavx512ifma -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq")
endif()

    set(DEBUG_FLAGS "-g -Og -DDEBUG -march=native ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
	set(OPTIMIZE_FLAGS "-g -march=native -mavx2 -mavx -mbmi2 -O3 -flto -DNDEBUG -fno-exceptions ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAGS} ${ADDITIONAL_FLAGS}")
endif()

# note that fplll is missing
set(LINK_TEST_FLAGS -lgtest -pthread gmpxx libm4ri.a m png)
set(LINK_BENCH_FLAGS -lgtest -pthread gmpxx libm4ri.a m png tbb -lbenchmark)


if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # if using GCC
    set(DEBUG_FLAGS "${DEBUG_FLAGS} ${GCC_HELPER_FLAGS} -O0 ")
endif()

# Set additionally our own debug/optimisation compiler/linkage flags. Maybe not he nicest way, but by far the easiest.
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")

# MAIN SOURCE FILES
set(SRCS src/config/config.cpp)

# add_library(small_secret_lwe_library STATIC ${SRCS})
# add_dependencies(small_secret_lwe_library fplll_project m4ri_project)
# target_compile_options(small_secret_lwe_library PRIVATE ${VALUE_FLAG})

# add_library(small_secret_lwe_binary_library STATIC ${SRCS})
# add_dependencies(small_secret_lwe_binary_library fplll_project)
# target_compile_options(small_secret_lwe_binary_library PRIVATE -DVALUE_BINARY)

# now check if we have to add the cuda runtime
include(CheckLanguage)
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


    # fall back, if one does not directly have a CUDA rdy card, but rather a optimus (internal laptop) card.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif()

    #add_executable(test_cuda tests/test_cuda.cu)
    #target_link_libraries(test_cuda ${LINK_TEST_FLAGS} fplll m4ri gmpxx gmp)
endif()

# add subdirectories
add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(tests)

