file(GLOB BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(benchfile ${BENCH_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${benchfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(bench_b63_search_${file} ${benchfile} )
    target_link_libraries(bench_b63_search_${file} m -lgtest -pthread -lbenchmark )
	add_test(AllBenchs bench_b63_search_${file})
endforeach( benchfile ${BENCH_SOURCES})
