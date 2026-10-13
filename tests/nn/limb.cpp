#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <vector>

#include "algorithm/min.h"
#include "nn/limb.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t w = 10;
using N = NNLimb<uint32_t, w>;
using P = std::pair<uint32_t, uint32_t>;

static std::set<P> to_set(const uint32_t (*out)[2], const size_t n) {
	std::set<P> s;
	for (size_t i = 0; i < n; ++i) { s.insert({out[i][0], out[i][1]}); }
	return s;
}

TEST(NNLimb, simd_equals_naive) {
	for (const size_t el : {0ul, 1ul, 7ul, 33ul, 100ul}) {
		for (const size_t er : {0ul, 5ul, 32ul, 97ul, 200ul}) {
			std::vector<uint32_t> L(el), R(er);
			for (auto &x : L) { x = cryptanalysislib::rng<uint32_t>(); }
			for (auto &x : R) { x = cryptanalysislib::rng<uint32_t>(); }
			// plant some solutions
			for (size_t i = 0; i < cryptanalysislib::min(el, er); i += 3) { R[i] = L[i] ^ 0x3FFu; }

			const size_t max = el * er + 1;
			std::unique_ptr<uint32_t[][2]> o1(new uint32_t[max][2]{}), o2(new uint32_t[max][2]{});
			const size_t n1 = N::nn(o1.get(), L.data(), R.data(), el, er, max);
			const size_t n2 = N::nn_bruteforce_simd_F2_32<uint32_t, 2, 3>(o2.get(), L.data(), R.data(), el, er, max);
			EXPECT_EQ(n1, n2);
			EXPECT_GE(n1, cryptanalysislib::min(el, er) / 3);
			EXPECT_EQ(to_set(o1.get(), n1), to_set(o2.get(), n2));

			// brute force reference
			size_t ref = 0;
			for (size_t i = 0; i < el; ++i) {
				for (size_t j = 0; j < er; ++j) {
					ref += cryptanalysislib::popcount::popcount(L[i] ^ R[j]) == w;
				}
			}
			EXPECT_EQ(n1, ref);

			// `max` is respected
			if (n1 > 1) {
				EXPECT_EQ(N::nn_bruteforce_simd_F2_32(o2.get(), L.data(), R.data(), el, er, 1), 1u);
				EXPECT_EQ(N::nn(o1.get(), L.data(), R.data(), el, er, 1), 1u);
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
