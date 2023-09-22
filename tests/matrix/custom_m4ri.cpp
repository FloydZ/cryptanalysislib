#include <gtest/gtest.h>

#include "../test.h"
#include "glue_m4ri.h"
#include "matrix/custom_m4ri.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace std;

TEST(TestEcho, matrix_echolonize_partial){
	int nc = 1024, nr = 1024;
	size_t k = 6;

	customMatrixData *matrix_data = init_matrix_data(nc);

	mzd_t *A = matrix_init(nr, nc);
	m4ri_random_full_rank(A);

	matrix_echelonize_partial(A, k, nr, matrix_data);



	for(uint64_t i = 0; i < nc; i++){
		for(uint64_t j = 0; j < nr; j++){
			if(i == j){
				ASSERT_EQ(1, A->rows[j][i/64] >> (i%64) & 1);
			}
			else{
				ASSERT_EQ(0, A->rows[j][i/64] >> (i%64) & 1);
			}
		}
	}

	free_matrix_data(matrix_data);
	mzd_free(A);
}



TEST(CustomMatrix, Test) {
	mzd_t *M = matrix_init(100, 150);
	customMatrixData *data = init_matrix_data(150);
	mzd_randomize(M);

	auto t = matrix_echelonize_partial(M, 4, 99, data);
	std::cout << t << "\n";

	free_matrix_data(data);
	mzd_free(M);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
