#include <gtest/gtest.h>

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u << 14u;

TEST(NearestNeighborAVX, avx2_sort_nn_on64) {
	constexpr static NN_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	const uint64_t z = fastrandombytes_uint64();
	size_t e1 = algo.simd_sort_nn_on64_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.simd_sort_nn_on64<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 = algo.simd_sort_nn_on64_simple<1>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on64<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 = algo.simd_sort_nn_on64_simple<2>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on64<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 =algo.simd_sort_nn_on64_simple<3>(LS, z, algo.L1);
	e2 =algo.simd_sort_nn_on64<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on_double64) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	NN<config> algo1{};
	NN<config> algo2{};
	algo1.generate_random_instance();
	// once needed
	algo2.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS);
	memcpy(algo2.L2, algo1.L2, LS);

	const uint64_t z = fastrandombytes_uint64();
	size_t e11 = algo1.simd_sort_nn_on64_simple<0>(LS, z, algo1.L1);
	size_t e12 = algo1.simd_sort_nn_on64_simple<0>(LS, z, algo1.L2);
	size_t e21=0, e22=0;
	algo2.simd_sort_nn_on_double64<0, 8>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS);
	memcpy(algo2.L2, algo1.L2, LS);
	e21=0, e22=0;

	e11 = algo1.simd_sort_nn_on64_simple<1>(LS, z, algo1.L1);
	e12 = algo1.simd_sort_nn_on64_simple<1>(LS, z, algo1.L2);
	algo2.simd_sort_nn_on_double64<1, 2>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS);
	memcpy(algo2.L2, algo1.L2, LS);
	e21=0, e22=0;

	e11 = algo1.simd_sort_nn_on64_simple<2>(LS, z, algo1.L1);
	e12 = algo1.simd_sort_nn_on64_simple<2>(LS, z, algo1.L2);
	algo2.simd_sort_nn_on_double64<2, 4>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on32) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 8, 320, 32, LS, 10, 8, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	uint32_t z = fastrandombytes_uint64();
	size_t e1 = algo.simd_sort_nn_on32_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.simd_sort_nn_on32<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 = algo.simd_sort_nn_on32_simple<1>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on32<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 = algo.simd_sort_nn_on32_simple<2>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on32<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 =algo.simd_sort_nn_on32_simple<3>(LS, z, algo.L1);
	e2 =algo.simd_sort_nn_on32<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on_double32) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 8, 320, 32, LS, 10, 12, 0, 512};
	NN<config> algo1{};
	NN<config> algo2{};
	algo1.generate_random_instance();
	// once needed
	algo2.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS);
	memcpy(algo2.L2, algo1.L2, LS);

	const uint64_t z = fastrandombytes_uint64();
	size_t e11 = algo1.simd_sort_nn_on32_simple<0>(LS, z, algo1.L1);
	size_t e12 = algo1.simd_sort_nn_on32_simple<0>(LS, z, algo1.L2);
	size_t e21=0, e22=0;
	algo2.simd_sort_nn_on_double32<0, 4>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS);
	memcpy(algo2.L2, algo1.L2, LS);
	e21=0, e22=0;

	e11 = algo1.simd_sort_nn_on32_simple<1>(LS, z, algo1.L1);
	e12 = algo1.simd_sort_nn_on32_simple<1>(LS, z, algo1.L2);
	algo2.simd_sort_nn_on_double32<1, 2>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);
	e21=0, e22=0;

	e11 = algo1.simd_sort_nn_on32_simple<2>(LS, z, algo1.L1);
	e12 = algo1.simd_sort_nn_on32_simple<2>(LS, z, algo1.L2);
	algo2.simd_sort_nn_on_double32<2, 4>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, simd_sort_nn_on_double32_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 12u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 8, 1, 32, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint32_t z;
	size_t e1=LS, e2=LS, new_e1=0, new_e2=0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<0, 1>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<0, 2>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<0, 4>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);

	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<1, 1>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<1, 2>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.simd_sort_nn_on_double32<1, 4>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
}

TEST(NearestNeighborAVX, simd_sort_nn_on_32_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 12u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 8, 1, 32, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint32_t z;
	size_t e1=LS,new_e1=0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	ASSERT(cryptanalysislib::popcount::popcount(z) == dk);
	new_e1 = algo1.simd_sort_nn_on32<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);

	algo1.L1[fastrandombytes_uint64()%LS][0] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][1] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][2] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][3] = -1ull;

	z = fastrandombytes_weighted<uint32_t>(dk);
	ASSERT(cryptanalysislib::popcount::popcount(z) == dk);
	new_e1 = algo1.simd_sort_nn_on32<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.simd_sort_nn_on32<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
}

TEST(NearestNeighborAVX, simd_sort_nn_on_64_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 14u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 4, 1, 64, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint64_t z;
	size_t e1=LS,new_e1=0;
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);

	algo1.L1[fastrandombytes_uint64()%LS][0] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][1] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][2] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][3] = -1ull;

	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.simd_sort_nn_on64<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
}

// k is not a multiple of 32
TEST(NearestNeighborAVX, simd_sort_nn_on32_k) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 8, 320, 28, LS, 8, 4, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	uint32_t z = fastrandombytes_uint64();
	size_t e1 = algo.simd_sort_nn_on32_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.simd_sort_nn_on32<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 = algo.simd_sort_nn_on32_simple<1>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on32<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 = algo.simd_sort_nn_on32_simple<2>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on32<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	z = fastrandombytes_uint64();
	e1 =algo.simd_sort_nn_on32_simple<3>(LS, z, algo.L1);
	e2 =algo.simd_sort_nn_on32<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, simd_sort_nn_on64_k) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 4, 320, 58, LS, 18, 16, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	const uint64_t z = fastrandombytes_uint64();
	size_t e1 = algo.simd_sort_nn_on64_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.simd_sort_nn_on64<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 = algo.simd_sort_nn_on64_simple<1>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on64<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 = algo.simd_sort_nn_on64_simple<2>(LS, z, algo.L1);
	e2 = algo.simd_sort_nn_on64<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j){
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS);

	e1 =algo.simd_sort_nn_on64_simple<3>(LS, z, algo.L1);
	e2 =algo.simd_sort_nn_on64<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}


#ifdef USE_AVX512
TEST(Bruteforce, avx512_32_8x8) {
	constexpr static NN_Config config{32, 1, 1, 1, LS, 2, 4, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx512_32_16x16(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}
#endif

int main(int argc, char **argv) {
	random_seed(0);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
