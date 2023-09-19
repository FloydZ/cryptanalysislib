file(GLOB BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(benchfile ${BENCH_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${benchfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(bench_b63_sort_${file} ${benchfile} )
    # NOTE: the benchmark is needed here
    target_link_libraries(bench_b63_sort_${file} ${MALLOC_LIBRARY} ${LINK_TEST_FLAGS} benchmark)
endforeach( benchfile ${BENCH_SOURCES})
