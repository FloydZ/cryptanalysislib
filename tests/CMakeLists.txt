#TESTS Flags
set(COMPILE_TEST_FLAGS -g -pthread)
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(${file} ${testfile})
    target_link_libraries(${file} ${LINK_TEST_FLAGS})
	add_test(AllTests ${file})
endforeach(testfile ${TEST_SOURCES})

#TODO osx failed kp, add_subdirectory(alloc)
add_subdirectory(atomic)
add_subdirectory(binary)
add_subdirectory(container)
add_subdirectory(crypto)
if(USE_CUDA)
add_subdirectory(cuda)
endif()
add_subdirectory(hash)
add_subdirectory(list)
add_subdirectory(math)
add_subdirectory(matrix)
#TODO add_subdirectory(nn)
add_subdirectory(popcount)
add_subdirectory(search)
add_subdirectory(simd)
add_subdirectory(sort)
add_subdirectory(thread)
