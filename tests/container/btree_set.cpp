#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <set>

#include "container/btree_set.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using std::size_t;
using std::uniform_int_distribution;
// Random number generation global variables
std::default_random_engine randGen((std::random_device())());
std::uniform_real_distribution<double> realDist;

template <typename E>
static bool contains(std::set<E> &set, const E &val) {
	return set.find(val) != set.end();
}

TEST(BTreeSet, SmallRandom) {
	const long TRIALS = 1000;
	const long OPERATIONS = 100;
	const int RANGE = 1000;
	uniform_int_distribution<int> valueDist(0, RANGE - 1);
	uniform_int_distribution<int> degreeDist(2, 6);

	for (long i = 0; i < TRIALS; i++) {
		std::set<int> set0;
		BTreeSet<int> set1(degreeDist(randGen));
		for (long j = 0; j < OPERATIONS; j++) {
			// Add/remove a random value
			int val = valueDist(randGen);
			if (realDist(randGen) < 0.001) {
				set0.clear();
				set1.clear();
			} else if (realDist(randGen) < 0.5) {
				set0.insert(val);
				set1.insert(val);
			} else {
				if (set0.erase(val) != set1.erase(val))
					throw std::runtime_error("Erase mismatch");
			}

			// Check size and check element membership over entire range
			if (set0.empty() != set1.empty())
				throw std::runtime_error("Empty mismatch");
			if (set0.size() != set1.size())
				throw std::runtime_error("Size mismatch");
			for (int k = -4; k < RANGE + 4; k++) {
				int val = k;
				if (contains(set0, val) != set1.contains(val))
					throw std::runtime_error("Contain test mismatch");
			}
		}
	}
}

TEST(BTreeSet, InsertRandom) {
	const long TRIALS = 100;
	const long OPERATIONS = 10'000;
	const long RANGE = 100'000;
	const long CHECKS = 10;
	uniform_int_distribution<long> valueDist(0, RANGE - 1);

	for (long i = 0; i < TRIALS; i++) {
		std::set<long> set0;
		BTreeSet<long> set1(2);
		for (long j = 0; j < OPERATIONS; j++) {
			// Add a random value
			long val = valueDist(randGen);
			set0.insert(val);
			set1.insert(val);
			//if (realDist(randGen) < 0.003)
			//	set1.checkStructure();

			// Check size and random element membership
			if (set0.size() != set1.size())
				throw std::runtime_error("Size mismatch");
			for (long k = 0; k < CHECKS; k++) {
				long val = valueDist(randGen);
				if (contains(set0, val) != set1.contains(val))
					throw std::runtime_error("Contain test mismatch");
			}
		}
	}
}

TEST(BTreeSet, LargeRandom) {
	const long TRIALS = 100;
	const long OPERATIONS = 30'000;
	const long RANGE = 100'000;
	const long CHECKS = 10;
	uniform_int_distribution<long> valueDist(0, RANGE - 1);
	uniform_int_distribution<int> degreeDist(2, 6);

	for (long i = 0; i < TRIALS; i++) {
		std::set<long> set0;
		BTreeSet<long> set1(degreeDist(randGen));
		for (long j = 0; j < OPERATIONS; j++) {
			// Add/remove a random value
			long val = valueDist(randGen);
			if (realDist(randGen) < 0.5) {
				set0.insert(val);
				set1.insert(val);
			} else {
				if (set0.erase(val) != set1.erase(val))
					throw std::runtime_error("Erase mismatch");
			}
			// if (realDist(randGen) < 0.001)
			// 	set1.checkStructure();

			// Check size and random element membership
			if (set0.size() != set1.size())
				throw std::runtime_error("Size mismatch");
			for (long k = 0; k < CHECKS; k++) {
				long val = valueDist(randGen);
				if (contains(set0, val) != set1.contains(val))
					throw std::runtime_error("Contain test mismatch");
			}
		}
	}
}

TEST(BTreeSet, RemoveRandom) {
	const long TRIALS = 100;
	const long LIMIT = 10'000;
	const long RANGE = 100'000;
	const long CHECKS = 10;
	uniform_int_distribution<long> valueDist(0, RANGE - 1);
	uniform_int_distribution<int> degreeDist(2, 6);

	for (long i = 0; i < TRIALS; i++) {
		std::set<long> set0;
		BTreeSet<long> set1(degreeDist(randGen));
		for (long j = 0; j < LIMIT; j++) {
			long val = valueDist(randGen);
			set0.insert(val);
			set1.insert(val);
		}

		// Incrementally remove each value
		std::vector<long> temp(set0.begin(), set0.end());
		std::shuffle(temp.begin(), temp.end(), randGen);
		for (long val : temp) {
			if (set0.erase(val) != set1.erase(val))
				throw std::runtime_error("Erase mismatch");
			// if (realDist(randGen) < 1.0 / std::min(std::max(set1.size(), static_cast<size_t>(1)), static_cast<size_t>(1000)))
			// 	set1.checkStructure();
			if (set0.size() != set1.size())
				throw std::runtime_error("Size mismatch");
			for (long k = 0; k < CHECKS; k++) {
				long val = valueDist(randGen);
				if (contains(set0, val) != set1.contains(val))
					throw std::runtime_error("Contain test mismatch");
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
