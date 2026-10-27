file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(test_nn_${file} ${testfile})
    target_compile_options(test_nn_${file} PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(test_nn_${file} ${LINK_TEST_FLAGS})
	add_test(nn_${file} test_nn_${file})
endforeach(testfile ${TEST_SOURCES})

if(CMAKE_CUDA_COMPILER)
    add_executable(test_nn_cuda cuda.cu)
	add_test(nn_cuda test_nn_cuda)

    add_executable(test_nn_tensor tensor.cu)
    target_link_libraries(test_nn_tensor GTest::gtest)
	add_test(nn_tensor test_nn_tensor)

    add_executable(test_nn_full_length full_length.cu)
    target_link_libraries(test_nn_full_length GTest::gtest)
	add_test(nn_full_length test_nn_full_length)

    add_executable(test_nn_tree tree.cu)
    target_link_libraries(test_nn_tree GTest::gtest)
	add_test(nn_tree test_nn_tree)

    set_tests_properties(nn_cuda nn_tensor nn_full_length nn_tree PROPERTIES LABELS cuda)
    add_dependencies(cuda_tests test_nn_cuda test_nn_tensor test_nn_full_length test_nn_tree)
endif()
