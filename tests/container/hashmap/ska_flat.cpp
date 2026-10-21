#include <gtest/gtest.h>
#include <functional>
#include <map>
#include <utility>

#include "container/hashmap.h"
#include "random.h"

using ::testing::InitGoogleTest;

using V = std::pair<uint64_t, uint64_t>;
struct KS {
	using key_type = uint64_t;
	const uint64_t &operator()(const V &v) const noexcept { return v.first; }
};
struct VS {
	using value_type = uint64_t;
	uint64_t &operator()(V &v) const noexcept { return v.second; }
	const uint64_t &operator()(const V &v) const noexcept { return v.second; }
};
template<typename GrowthPolicy = cryptanalysislib::hh::fibonacci_growth_policy>
using Table = sherwood_v3_table<V, KS, VS, std::hash<uint64_t>, std::equal_to<uint64_t>,
                                std::allocator<V>, GrowthPolicy>;

/// insert, operator[], find (also with a precomputed hash), erase against std::map
template<typename Table>
void random_vs_std_map() {
	for (const uint64_t mod : {50ull, 5000ull, 1ull << 40}) {
		Table t;
		std::map<uint64_t, uint64_t> e;
		for (size_t i = 0; i < 20000; ++i) {
			const uint64_t k = cryptanalysislib::rng<uint64_t>() % mod;
			switch (cryptanalysislib::rng() % 4) {
				case 0: t[k] += i; e[k] += i; break;
				case 1: EXPECT_EQ(t.insert(V{k, i}).second, e.insert({k, i}).second); break;
				case 2: EXPECT_EQ(t.erase(k), e.erase(k)); break;
				default: {
					const auto it = t.find(k, std::hash<uint64_t>{}(k));
					const auto ie = e.find(k);
					ASSERT_EQ(it == t.end(), ie == e.end());
					if (ie != e.end()) {
						EXPECT_EQ(it->second, ie->second);
						EXPECT_EQ(it.value(), ie->second);
					}
				}
			}
		}
		ASSERT_EQ(t.size(), e.size());
		for (const auto &[k, v] : e) {
			const auto it = t.find(k);
			ASSERT_NE(it, t.end());
			EXPECT_EQ(it->second, v);
		}

		t.clear();
		EXPECT_EQ(t.size(), 0u);
		EXPECT_EQ(t.find(1), t.end());
	}
}

TEST(ska_flat, random_vs_std_map) {
	random_vs_std_map<Table<>>();
}

/// the other growth policies
TEST(ska_flat, growth_policies) {
	random_vs_std_map<Table<cryptanalysislib::hh::power_of_two_growth_policy<2>>>();
	random_vs_std_map<Table<cryptanalysislib::hh::mod_growth_policy<>>>();
	random_vs_std_map<Table<cryptanalysislib::hh::prime_growth_policy>>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	cryptanalysislib::rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
