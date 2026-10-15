#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib;

TEST(log, simple) {
    // NOTE: qualified, `log` alone is ambiguous with ::log from <cmath>
    cryptanalysislib::log << "test1";
    cryptanalysislib::log << logging::debug << "test2";
    cryptanalysislib::log(logging::debug) << "test3";
    logging::debug << "tes4"; 
}

/// redirects `std::cout` into a string for the lifetime of the object
struct CaptureCout {
    std::stringstream ss;
    std::streambuf *old = std::cout.rdbuf(ss.rdbuf());
    ~CaptureCout() { std::cout.rdbuf(old); }
};

TEST(log, default_config) {
    CaptureCout c;
    logging::logger<> l;
    l.debug("a", 1);
    l.error("b");
    EXPECT_EQ(c.ss.str(), "a1\nb\n");
}

constexpr static logging::LoggerConfig warn_config{
    .min_level = logging::LoggingLevel::warning,
    .flush = false,
    .print_level = true,
};

TEST(log, config) {
    CaptureCout c;
    logging::logger<warn_config> l;
    l.debug("dropped");
    l.info("dropped");
    l.warning("w", 2);
    l.error("e");
    l(logging::info) << "dropped";
    l << logging::error << "x";
    EXPECT_EQ(c.ss.str(), "[warning] w2\n[error] e\n[error] x\n");
}

TEST(log, data_logger) {
    CaptureCout c;
    const uint32_t data1 = 5;
    const std::vector<uint8_t> data2{1, 2, 255};
    cryptanalysislib::data_logger << LOG_DATA(data1) << LOG_DATA(data2) << cryptanalysislib::endl;
    cryptanalysislib::data_logger << cryptanalysislib::endl;
    EXPECT_EQ(c.ss.str(), "{\"data1\": 5, \"data2\": [1, 2, 255]}\n{}\n");
}

TEST(log, data_logger_types) {
    std::stringstream ss;
    logging::DataLogger dl{ss};
    const std::string s = "a\"b\\\n";
    const double d = 0.1;
    const bool b = true;
    const int8_t i = -3;
    const std::vector<std::vector<int>> v{{1}, {}, {2, 3}};
    dl << LOG_DATA(s) << LOG_DATA(d) << LOG_DATA(b) << LOG_DATA(i)
       << LOG_DATA(v) << logging::named("sum", i + 1) << cryptanalysislib::endl;
    EXPECT_EQ(ss.str(), "{\"s\": \"a\\\"b\\\\\\n\", \"d\": 0.10000000000000001, \"b\": true, "
                        "\"i\": -3, \"v\": [[1], [], [2, 3]], \"sum\": -2}\n");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
