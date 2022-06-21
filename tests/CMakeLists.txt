#TESTS Flags
set(LINK_TEST_FLAGS -lgtest -pthread gmp fplll gmpxx libm4ri.a m png tbb -lbenchmark)
set(COMPILE_TEST_FLAGS -g -pthread -march=native)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(${file} ${testfile})
    target_compile_options(${file} PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(${file} ${LINK_TEST_FLAGS})
    #add_dependencies(${file} m4ri_project fplll_project)
endforeach(testfile ${TEST_SOURCES})

add_subdirectory(binary)
if(USE_CUDA)
add_subdirectory(cuda)
endif()

add_subdirectory(popcount)