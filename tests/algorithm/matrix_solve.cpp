#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "algorithm/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// tridiagonal system a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i with a known solution
TEST(matrix, solve_tridiagonal) {
	using namespace cryptanalysislib;
	for (const int n : {2, 3, 10, 100}) {
		double x[100], A[100], B[100], C[100], D[100], X[100];
		for (int i = 0; i < n; ++i) { x[i] = 1.0 + 0.5 * i; }
		for (int i = 0; i < n; ++i) {
			A[i] = (i == 0) ? 0. : -1.;
			C[i] = (i == n - 1) ? 0. : -1.;
			B[i] = 4.;
			D[i] = B[i] * x[i] + (i ? A[i] * x[i - 1] : 0.) + (i < n - 1 ? C[i] * x[i + 1] : 0.);
		}
		solve<double>(X, A, B, C, D, n);
		for (int i = 0; i < n; ++i) { EXPECT_NEAR(X[i], x[i], 1e-9) << n << " " << i; }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
