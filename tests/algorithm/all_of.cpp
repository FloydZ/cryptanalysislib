#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/all_of.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(all_of, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::all_of(in.begin(), in.end(), [](const T &a) {
    	return a == 1;
    });

    EXPECT_EQ(d, true);
}

TEST(any_of, int_) {
	constexpr static size_t s = 100;
	using T = int;
	std::vector<T> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);

	const auto d1 = cryptanalysislib::any_of(in.begin(), in.end(), [](const T &a) {
		return a == 1;
	});

	EXPECT_EQ(d1, true);

	const auto d2 = cryptanalysislib::any_of(in.begin(), in.end(), [](const T &a) {
		return a == 0;
	});

	EXPECT_EQ(d2, false);
}

TEST(none_of, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d1 = cryptanalysislib::none_of(in.begin(), in.end(), [](const T &a) {
    	return a == 1;
    });

    EXPECT_EQ(d1, false);

	const auto d2 = cryptanalysislib::none_of(in.begin(), in.end(), [](const T &a) {
    	return a == 0;
    });

    EXPECT_EQ(d2, true);
}

TEST(all_of, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::all_of(par_if(true), in.begin(), in.end(), [](const T &a) {
	    return a == 1;
    });
    EXPECT_EQ(d, true);
}

TEST(any_of, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d1 = cryptanalysislib::any_of(par_if(true), in.begin(), in.end(), [](const T &a) {
    	return a == 1;
    });

    EXPECT_EQ(d1, true);

	const auto d2 = cryptanalysislib::any_of(par_if(true), in.begin(), in.end(), [](const T &a) {
    	return a == 0;
    });

    EXPECT_EQ(d2, false);
}

TEST(none_of, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d1 = cryptanalysislib::none_of(par_if(true), in.begin(), in.end(), [](const T &a) {
    	return a == 1;
    });

    EXPECT_EQ(d1, false);

	const auto d2 = cryptanalysislib::none_of(par_if(true), in.begin(), in.end(), [](const T &a) {
    	return a == 0;
    });

    EXPECT_EQ(d2, true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
